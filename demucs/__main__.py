# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import torch as th
from torch import distributed, nn
from torch.utils.data import ConcatDataset
from torch.nn.parallel.distributed import DistributedDataParallel

from .augment import FlipChannels, FlipSign, Remix, Scale, Shift
from .compressed import get_compressed_datasets
from .model import Demucs
from .parser import get_name, get_parser
from .raw import Rawset
from .repitch import RepitchedWrapper
from .pretrained import load_pretrained, SOURCES
from .tasnet import ConvTasNet
from .test import evaluate
from .train import train_model, validate_model
from .utils import (human_seconds, load_model, save_model, get_state,
                    save_state, sizeof_fmt, get_quantizer)
from .wav import get_wav_datasets, get_musdb_wav_datasets

# added by Holik Viliam
from .loss import (LogL1, LogL2, SISDR, FreqL1, FreqMSE,
                   FreqLogL1, FreqLogL2, FreqSISDR)
##############################


@dataclass
class SavedState:
    metrics: list = field(default_factory=list)
    last_state: dict = None
    best_state: dict = None
    optimizer: dict = None
    counter: int = 0    # added


def main():
    parser = get_parser()
    args = parser.parse_args()
    name = get_name(parser, args)
    print(f"Experiment {name}")

    if args.musdb is None and args.rank == 0:
        print(
            "You must provide the path to the MusDB dataset with the --musdb flag. "
            "To download the MusDB dataset, see https://sigsep.github.io/datasets/musdb.html.",
            file=sys.stderr)
        sys.exit(1)

    eval_folder = args.evals / name
    eval_folder.mkdir(exist_ok=True, parents=True)
    args.logs.mkdir(exist_ok=True)
    metrics_path = args.logs / f"{name}.json"
    eval_folder.mkdir(exist_ok=True, parents=True)
    args.checkpoints.mkdir(exist_ok=True, parents=True)
    args.models.mkdir(exist_ok=True, parents=True)

    if args.device is None:
        device = "cpu"
        if th.cuda.is_available():
            device = "cuda"
    else:
        device = args.device

    th.manual_seed(args.seed)
    # Prevents too many threads to be started when running `museval` as it can be quite
    # inefficient on NUMA architectures.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    if args.world_size > 1:
        if device != "cuda" and args.rank == 0:
            print("Error: distributed training is only available with cuda device", file=sys.stderr)
            sys.exit(1)
        th.cuda.set_device(args.rank % th.cuda.device_count())
        distributed.init_process_group(backend="nccl",
                                       init_method="tcp://" + args.master,
                                       rank=args.rank,
                                       world_size=args.world_size)

    checkpoint = args.checkpoints / f"{name}.th"
    checkpoint_tmp = args.checkpoints / f"{name}.th.tmp"
    if args.restart and checkpoint.exists() and args.rank == 0:
        checkpoint.unlink()

    if args.test or args.test_pretrained:
        args.epochs = 1
        args.repeat = 0
        if args.test:
            model = load_model(args.models / args.test)
        else:
            model = load_pretrained(args.test_pretrained)
    elif args.tasnet:
        model = ConvTasNet(audio_channels=args.audio_channels,
                           samplerate=args.samplerate, X=args.X,
                           N=args.N, L=args.L, B=args.B,
                           H=args.H, P=args.P, R=args.R,
                           segment_length=4 * args.samples,
                           sources=SOURCES)
    else:
        model = Demucs(
            audio_channels=args.audio_channels,
            channels=args.channels,
            context=args.context,
            depth=args.depth,
            glu=args.glu,
            growth=args.growth,
            kernel_size=args.kernel_size,
            lstm_layers=args.lstm_layers,
            rescale=args.rescale,
            rewrite=args.rewrite,
            stride=args.conv_stride,
            resample=args.resample,
            normalize=args.normalize,
            samplerate=args.samplerate,
            segment_length=4 * args.samples,
            sources=SOURCES,
        )
    model.to(device)
    if args.init:
        model.load_state_dict(load_pretrained(args.init).state_dict())

    if args.show:
        print(model)
        size = sizeof_fmt(4 * sum(p.numel() for p in model.parameters()))
        print(f"Model size {size}")
        return

    try:
        saved = th.load(checkpoint, map_location='cpu')
    except IOError:
        saved = SavedState()

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

    quantizer = None
    quantizer = get_quantizer(model, args, optimizer)

    if saved.last_state is not None:
        model.load_state_dict(saved.last_state, strict=False)
    if saved.optimizer is not None:
        optimizer.load_state_dict(saved.optimizer)

    model_name = f"{name}.th"
    if args.save_model:
        if args.rank == 0:
            model.to("cpu")
            assert saved.best_state is not None, "model needs to train for 1 epoch at least."
            model.load_state_dict(saved.best_state)
            save_model(model, quantizer, args, args.models / model_name)
        return
    elif args.save_state:
        model_name = f"{args.save_state}.th"
        if args.rank == 0:
            model.to("cpu")
            model.load_state_dict(saved.best_state)
            state = get_state(model, quantizer)
            save_state(state, args.models / model_name)
        return

    if args.rank == 0:
        done = args.logs / f"{name}.done"
        if done.exists():
            done.unlink()

    augment = [Shift(args.data_stride)]
    if args.augment:
        augment += [FlipSign(), FlipChannels(), Scale(),
                    Remix(group_size=args.remix_group_size)]
    augment = nn.Sequential(*augment).to(device)
    print("Agumentation pipeline:", augment)

    if args.mse:
        criterion = nn.MSELoss()

    # added by Holik Viliam
    elif args.logL1:
        criterion = LogL1()
    elif args.logL2:
        criterion = LogL2()
    elif args.SISDR:
        criterion = SISDR()
    elif args.freqL1:
        criterion = FreqL1()
    elif args.freqMSE:
        criterion = FreqMSE()
    elif args.freqLogL1:
        criterion = FreqLogL1()
    elif args.freqLogL2:
        criterion = FreqLogL2()
    elif args.freqSISDR:
        criterion = FreqSISDR()
    ##############################

    else:
        criterion = nn.L1Loss()

    # Setting number of samples so that all convolution windows are full.
    # Prevents hard to debug mistake with the prediction being shifted compared
    # to the input mixture.
    samples = model.valid_length(args.samples)
    print(f"Number of training samples adjusted to {samples}")
    samples = samples + args.data_stride
    if args.repitch:
        # We need a bit more audio samples, to account for potential
        # tempo change.
        samples = math.ceil(samples / (1 - 0.01 * args.max_tempo))

    args.metadata.mkdir(exist_ok=True, parents=True)
    if args.raw:
        train_set = Rawset(args.raw / "train",
                           samples=samples,
                           channels=args.audio_channels,
                           streams=range(1, len(model.sources) + 1),
                           stride=args.data_stride)

        valid_set = Rawset(args.raw / "valid", channels=args.audio_channels)
    elif args.wav:
        train_set, valid_set = get_wav_datasets(args, samples, model.sources)

        if args.concat:
            if args.is_wav:
                mus_train, mus_valid = get_musdb_wav_datasets(args, samples, model.sources)
            else:
                mus_train, mus_valid = get_compressed_datasets(args, samples)
            train_set = ConcatDataset([train_set, mus_train])
            valid_set = ConcatDataset([valid_set, mus_valid])
    elif args.is_wav:
        train_set, valid_set = get_musdb_wav_datasets(args, samples, model.sources)
    else:
        train_set, valid_set = get_compressed_datasets(args, samples)
    print("Train set and valid set sizes", len(train_set), len(valid_set))

    if args.repitch:
        train_set = RepitchedWrapper(
            train_set,
            proba=args.repitch,
            max_tempo=args.max_tempo)

    best_loss = float("inf")
    for epoch, metrics in enumerate(saved.metrics):
        print(f"Epoch {epoch:03d}: "
              f"train={metrics['train']:.8f} "
              f"valid={metrics['valid']:.8f} "
              f"best={metrics['best']:.4f} "
              f"ms={metrics.get('true_model_size', 0):.2f}MB "
              f"cms={metrics.get('compressed_model_size', 0):.2f}MB "
              f"duration={human_seconds(metrics['duration'])}")
        best_loss = metrics['best']

    if args.world_size > 1:
        dmodel = DistributedDataParallel(model,
                                         device_ids=[th.cuda.current_device()],
                                         output_device=th.cuda.current_device())
    else:
        dmodel = model

    counter = saved.counter         # added

    for epoch in range(len(saved.metrics), args.epochs):
        ### added scection
        break_out = False
        for g in optimizer.param_groups:
            ac_lr = g['lr']
            if ac_lr <= 1e-6:
                break_out = True
        if break_out:
            break
        ### end

        begin = time.time()
        model.train()
        train_loss, model_size = train_model(
            epoch, train_set, dmodel, criterion, optimizer, augment,
            quantizer=quantizer,
            batch_size=args.batch_size,
            device=device,
            repeat=args.repeat,
            seed=args.seed,
            diffq=args.diffq,
            workers=args.workers,
            world_size=args.world_size)
        model.eval()
        valid_loss = validate_model(
            epoch, valid_set, model, criterion,
            device=device,
            rank=args.rank,
            split=args.split_valid,
            overlap=args.overlap,
            world_size=args.world_size)

        ms = 0
        cms = 0
        if quantizer and args.rank == 0:
            ms = quantizer.true_model_size()
            cms = quantizer.compressed_model_size(num_workers=min(40, args.world_size * 10))

        duration = time.time() - begin
        if valid_loss < best_loss and ms <= args.ms_target:
            counter = 0     # added
            best_loss = valid_loss
            saved.best_state = {
                key: value.to("cpu").clone()
                for key, value in model.state_dict().items()
            }
        ### added section
        else:
            counter += 1
            if counter >= 8:
                for g in optimizer.param_groups:
                    lr = g['lr']
                    lr *= 0.5
                    print("Actual LR = ", lr)
                    g['lr'] = lr
                counter = 0
        ### end

        saved.metrics.append({
            "train": train_loss,
            "valid": valid_loss,
            "best": best_loss,
            "duration": duration,
            "model_size": model_size,
            "true_model_size": ms,
            "compressed_model_size": cms,
        })
        if args.rank == 0:
            json.dump(saved.metrics, open(metrics_path, "w"))

        saved.last_state = model.state_dict()
        saved.optimizer = optimizer.state_dict()
        saved.counter = counter     # added
        if args.rank == 0 and not args.test:
            th.save(saved, checkpoint_tmp)
            checkpoint_tmp.rename(checkpoint)

        print(f"Epoch {epoch:03d}: "
              #f"LR={lr:.8f} "    # added
              f"train={train_loss:.8f} valid={valid_loss:.8f} best={best_loss:.4f} ms={ms:.2f}MB "
              f"cms={cms:.2f}MB "
              f"duration={human_seconds(duration)}")

    if args.world_size > 1:
        distributed.barrier()

    del dmodel
    model.load_state_dict(saved.best_state)
    if args.eval_cpu:
        device = "cpu"
        model.to(device)
    model.eval()
    evaluate(model, args.musdb, eval_folder,
             is_wav=args.is_wav,
             rank=args.rank,
             world_size=args.world_size,
             device=device,
             save=args.save,
             split=args.split_valid,
             shifts=args.shifts,
             overlap=args.overlap,
             workers=args.eval_workers)
    model.to("cpu")
    if args.rank == 0:
        if not (args.test or args.test_pretrained):
            save_model(model, quantizer, args, args.models / model_name)
        print("done")
        done.write_text("done")


if __name__ == "__main__":
    main()
