import torch
from torch import nn

EPS = 1e-8

def make_freq(predictions, target,
		len = 2048,
		hop = 512,
		pad = "constant"):
    """
    This function converts time domain to frequency domain via STFT,
    needs predictions and target and return converted ones.
    Input  time domain      3D/4D - [(batch,) sources, stereo, samples]
    Output frequency domain 4D/5D - [(batch,) sources, stereo, frames, freq bin]
    """
    win = torch.hann_window(len, device="cuda")

    # STFT input must be either 1D or 2D, flatten 3D/4D tensor to acceptable shape
    if target.dim() == 4:	# TRAIN cycle [batch, sources, stereo, samples]
        pr = torch.flatten(predictions, start_dim=0, end_dim=2)
        tr = torch.flatten(target, start_dim=0, end_dim=2)
        tensor_dims = [target.size(dim=0), target.size(dim=1), target.size(dim=2)]
    else:			# VALID cycle [sources, stereo, samples]
        pr = torch.flatten(predictions, start_dim=0, end_dim=1)
        tr = torch.flatten(target, start_dim=0, end_dim=1)
        tensor_dims = [target.size(dim=0), target.size(dim=1)]

    # STFT
    pred = torch.stft(pr, len, hop_length=hop, window=win, pad_mode=pad, return_complex=True)
    tar = torch.stft(tr, len, hop_length=hop, window=win, pad_mode=pad, return_complex=True)

    # reshape tensor back
    freq = list(tar.size())
    size = [*tensor_dims, freq[-2], freq[-1]]
    pred = torch.reshape(pred, size)
    tar = torch.reshape(tar, size)
    return pred, tar


class LogL1(nn.Module):
    """Logarithmic L1 loss function in time domain"""
    def __init__(self):
        super(LogL1, self).__init__()

    def forward(self, predictions, target):
        if predictions.dim() == 4:	# TRAIN
            dim_list = (1, 2, 3)
        else:				# VALID
            dim_list = (0, 1, 2)

        diff = torch.abs(predictions - target)
        diff_mean = torch.mean(diff, dim=dim_list)
        log = torch.log10(diff_mean + EPS)
        loss_value = 10 * torch.mean(log)
        return loss_value


class FreqLogL1(nn.Module):
    """Logarithmic L1 loss function in frequency domain"""
    def __init__(self):
        super(FreqLogL1, self).__init__()

    def forward(self, predictions, target):
        predictions, target = make_freq(predictions, target)
        if predictions.dim() == 5:	# TRAIN cycle [batch, sources, stereo, frames, freq bin]
            dim_list = (1, 2, 3, 4)
        else:				# VALID cycle [sources, stereo, frames, freq bin]
            dim_list = (0, 1, 2, 3)

        diff = torch.abs(torch.abs(predictions) - torch.abs(target))
        diff_mean = torch.mean(diff, dim=dim_list)
        log = torch.log10(diff_mean + EPS)
        loss_value = 10 * torch.mean(log)
        return loss_value


class FreqL1(nn.Module):
    """L1 loss function in frequency domain"""
    def __init__(self):
        super(FreqL1, self).__init__()

    def forward(self, predictions, target):
        pred, tar = make_freq(predictions, target)
        diff = torch.abs(torch.abs(pred) - torch.abs(tar))
        loss_value = torch.mean(diff)
        return loss_value


class FreqMSE(nn.Module):
    """L2/MSE loss function in frequency domain"""
    def __init__(self):
        super(FreqMSE, self).__init__()

    def forward(self, predictions, target):
        pred, tar = make_freq(predictions, target)
        diff = torch.abs(torch.abs(pred) - torch.abs(tar))
        loss_value = torch.mean(torch.square(diff))
        return loss_value


class LogL2(nn.Module):
    """Logarithmic L2 loss function in time domain"""
    def __init__(self):
        super(LogL2, self).__init__()

    def forward(self, predictions, target):
        if target.dim() == 4:		# TRAIN
            dim_list = (1, 2, 3)
        else:				# VALID
            dim_list = (0, 1, 2)

        diff_squared = torch.square(predictions - target)
        diff_mean = torch.mean(diff_squared, dim=dim_list)
        log = torch.log10(diff_mean + EPS)
        loss_value = 10 * torch.mean(log)
        return loss_value


class FreqLogL2(nn.Module):
    """Logarithmic L2 loss function in frequency domain"""
    def __init__(self):
        super(FreqLogL2, self).__init__()

    def forward(self, predictions, target):
        predictions, target = make_freq(predictions, target)
        if predictions.dim() == 5:      # TRAIN cycle with batch
            dim_list = (1, 2, 3, 4)
        else:                           # VALID cycle without batch
            dim_list = (0, 1, 2, 3)

        diff = torch.abs(predictions) - torch.abs(target)
        diff_mean = torch.mean(torch.square(diff), dim=dim_list)
        log = torch.log10(diff_mean + EPS)
        loss_value = 10 * torch.mean(log)
        return loss_value


class SISDR(nn.Module):
    """SI-SDR loss function in time domain"""
    def __init__(self):
        super(SISDR, self).__init__()

    def forward(self, predictions, target):
        if predictions.dim() == 4:	# TRAIN
            dim_list = (1, 2, 3)
            last = 3
        else:				# VALID
            dim_list = (0, 1, 2)
            last = 2

        # function for normalization
        def normalize(a, b=None):
            b = a if b is None else b
            dot = torch.matmul(torch.unsqueeze(a, last), torch.unsqueeze(b, last+1))
            return torch.squeeze(dot, last)

        alpha = normalize(target, predictions)
        target_norm = normalize(target)
        s_target = (alpha / (target_norm + EPS)) * target
        e_noise = s_target - predictions

        numerator = torch.square(s_target)
        denominator = torch.square(e_noise)
        log = torch.log10(torch.mean(numerator / (denominator + EPS), dim=dim_list) + EPS)
        loss_value = 10 * torch.mean(log)
        return -loss_value

    def old(self, predictions, target):
        alpha_1 = torch.sum(torch.mul(predictions, target), dim=last, keepdim=True)
        alpha = torch.div(alpha_1, torch.sum(torch.square(target), dim=last, keepdim=True))
        root = torch.mul(alpha, target)
        numerator = torch.sum(torch.square(torch.abs(root)), dim=dim_list, keepdim=True)
        denominator = torch.sum(torch.square(torch.abs(root - predictions)), dim=dim_list, keepdim=True)
        log = torch.log10(torch.mean(torch.div(numerator, denominator + EPS), dim=dim_list) + EPS)
        loss_value = 10 * torch.mean(log)
        return -loss_value


class FreqSISDR(nn.Module):
    """SI-SDR loss function in frequency domain"""
    def __init__(self):
        super(FreqSISDR, self).__init__()
        self.crit = SISDR()

    def forward(self, predictions, target):
        predictions, target = make_freq(predictions, target)
        predictions = torch.flatten(torch.abs(predictions), start_dim=-2)
        target = torch.flatten(torch.abs(target), start_dim=-2)
        return self.crit(predictions, target)

    def old(self, predictions, target):
        predictions, target = make_freq(predictions, target)
        predictions = torch.abs(predictions)
        target = torch.abs(target)
        if predictions.dim() == 5:	# TRAIN
            dim_list = (1, 2, 3, 4)
            last_2 = (3, 4)
        else:				# VALID
            dim_list = (0, 1, 2, 3)
            last_2 = (2, 3)

        alpha_1 = torch.sum(torch.mul(predictions, target), dim=last_2, keepdim=True)
        alpha = torch.div(alpha_1, torch.sum(torch.square(target), dim=last_2, keepdim=True))
        root = torch.mul(alpha, target)
        numerator = torch.sum(torch.square(torch.abs(root)), dim=dim_list, keepdim=True)
        denominator = torch.sum(torch.square(torch.abs(root - predictions)), dim=dim_list, keepdim=True)
        log = torch.log10(torch.mean(torch.div(numerator, denominator + EPS), dim=dim_list) + EPS)
        loss_value = 10 * torch.mean(log)
        return -loss_value
