import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchlibrosa import Spectrogram, LogmelFilterBank, SpecAugmentation
from utils import normalize, getfminfmax
from einops import rearrange


def _one_sample_positive_class_precisions(scores, truth):
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)

    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)

    retrieved_classes = np.argsort(scores)[::-1]

    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)

    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True

    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)

    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


def lwlrap(truth, scores):
    truth = (truth > 0.5).astype(int)
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(scores[sample_num, :], truth[sample_num, :])
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = precision_at_hits

    labels_per_class = np.sum(truth > 0.5, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))

    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    return (per_class_lwlrap*weight_per_class).sum()


def kaggle_metric(preds, labels):
    return lwlrap(labels, preds)


def patch_first_conv(model, in_channels):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            break

    # change input channels for first conv
    module.in_channels = in_channels
    weight = module.weight.detach()
    reset = False

    if in_channels == 1:
        weight = weight.sum(1, keepdim=True)
    elif in_channels == 2:
        weight = weight[:, :2] * (3.0 / 2.0)
    else:
        reset = True
        weight = torch.Tensor(
            module.out_channels,
            module.in_channels // module.groups,
            *module.kernel_size
        )

    module.weight = nn.parameter.Parameter(weight)
    if reset:
        module.reset_parameters()


def criterion(pred, true, masks, focal=0., pos_weight=1., lsoft=0., lsep=0):
    if lsep:
        return lsep_loss(pred, true)
    if lsoft > 0:
        with torch.no_grad():
            tmp_true = (1 - lsoft) * true + lsoft * torch.sigmoid(pred)
        true = torch.where(masks == 0, tmp_true, true)
    loss = nn.BCEWithLogitsLoss(reduction='none')(pred, true)
    if pos_weight != 1:
        loss = torch.where(true >= 0.5, pos_weight * loss, loss)
    if focal > 0:
        probas = torch.sigmoid(pred)
        loss = torch.where(true >= 0.5, (1. - probas) ** focal * loss, probas ** focal * loss)
    if lsoft == 1:
        loss = (loss*masks).sum(dim=-1)/masks.sum(dim=-1)
    return loss.mean()


def lsep_loss(input, target, average=True):

    differences = input.unsqueeze(1) - input.unsqueeze(2)
    where_different = (target.unsqueeze(1) < target.unsqueeze(2)).float()

    exps = differences.exp() * where_different
    lsep = torch.log(1 + exps.sum(2).sum(1))
    if average:
        return lsep.mean()
    else:
        return lsep


class RFCXModel(nn.Module):
    def __init__(self, cfg):
        super(RFCXModel, self).__init__()

        self.cfg = cfg
        self.bn = nn.BatchNorm2d(1)
        self.encoder = timm.create_model(cfg.encoder, pretrained=not bool(cfg.init_random), num_classes=0, global_pool='')
        patch_first_conv(self.encoder, 1)
        dim = self.encoder.conv_head.out_channels

        # self.head = nn.Sequential(
        #     nn.LayerNorm(dim*24),
        #     nn.Dropout(cfg.dropout),
        #     nn.Linear(dim*24, 24)
        # )
        self.heads = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(dim, 1)
        ) for _ in range(24)])

        self.spectrogram_extractor = Spectrogram(
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=None,
            window='hann',
            center=True,
            pad_mode='reflect',
            freeze_parameters=False
        )

        self.logmel_extractors = []
        for i in range(24):
            fmin, fmax = getfminfmax(i, cfg.fminfmax_offset)
            self.logmel_extractors.append(
                LogmelFilterBank(
                    sr=self.cfg.sr,
                    n_fft=self.cfg.n_fft,
                    n_mels=self.cfg.n_mels,
                    fmin=fmin,
                    fmax=fmax,
                    is_log=True,
                    ref=1.0,
                    amin=1e-10,
                    top_db=80.0,
                    freeze_parameters=True
                )
            )
        self.logmel_extractors = nn.ModuleList(self.logmel_extractors)

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=32,
            time_stripes_num=1,
            freq_drop_width=2,
            freq_stripes_num=1
        )

    def forward(self, x):
        # to melspec
        x = self.spectrogram_extractor(x)  # (batch_size, 1, time_steps, freq_bins)
        logmelspecs = []
        for i in range(24):
            logmelspecs.append(self.logmel_extractors[i](x))
        x = torch.cat(logmelspecs, dim=0)
        if self.training and self.cfg.aug:
            x = self.spec_augmenter(x)
        x = self.bn(x)
        x = self.encoder(x)
        b, c, h, w = x.shape
        # features = x.view(b, c, -1).max(dim=-1)[0]
        mask = (x > 0).float()
        features = (x*mask).sum(dim=(2, 3))/(torch.maximum(mask.sum(dim=(2, 3)), torch.tensor(1e-8).to(mask.device)))
        features = features.reshape(24, -1, c)
        # logits = self.head(rearrange(features, '(l b) c -> b (l c)', l=24))
        logits = []
        for i, head in enumerate(self.heads):
            logits.append(head(features[i]))
        logits = torch.cat(logits, dim=1)
        return logits

