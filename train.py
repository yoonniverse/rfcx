import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
import joblib
import os
import warnings
from time import time
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from torchlibrosa import LogmelFilterBank

from model import RFCXModel, kaggle_metric, criterion
from data import get_loader
from utils import make_reproducible, mkdir, CFG, getfminfmax, sigmoid
from cam import FeatureHook, get_cam, blend_cam

warnings.filterwarnings('ignore')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:3456', rank=rank, world_size=world_size)


def main_worker(rank, world_size, cfg):

    verbose_cond = (rank == 0) and cfg.verbose
    if cfg.verbose: print(f'Launched gpu {rank}')
    setup(rank, world_size)
    cfg.batch_size = cfg.batch_size // world_size
    cfg.infer_batch_size = cfg.infer_batch_size // world_size
    cfg.num_workers = cfg.num_workers // world_size

    # for reproduciblity
    make_reproducible(42) if cfg.resume else make_reproducible(cfg.seed)

    # create logdir
    mkdir(cfg.logdir)

    # create dataloader
    rid_by_sid = joblib.load('rid_by_sid.jl')
    trn_loader = get_loader(cfg, mode='train')
    val_loader = get_loader(cfg, mode='valid')
    if verbose_cond: print(f'# Train Samples: {len(trn_loader.dataset)} | # Val Samples: {len(val_loader.dataset)}')

    # define model
    model = RFCXModel(cfg).to(rank)
    n_parameters = sum(p.numel() for p in model.parameters())
    if verbose_cond: print(f'# Parameters: {n_parameters}')

    # define optimizer
    kwargs = []
    for name, params in model.named_parameters():
        tmp = {'params': params}
        if any([x in name for x in ['bias', 'bn']]):
            tmp['weight_decay'] = 0
        else:
           tmp['weight_decay'] = cfg.weight_decay
        # if ('encoder' in name) and ('conv_stem' not in name):
        #     tmp['lr'] = cfg.lr
        # else:
        #     tmp['lr'] = cfg.lr * cfg.raw_param_lr_ratio
        kwargs.append(tmp)

    optimizer = torch.optim.AdamW(kwargs, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # define lr scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, pct_start=0.01, max_lr=cfg.lr, steps_per_epoch=len(trn_loader), epochs=cfg.epochs,
        final_div_factor=1e1 if cfg.focal else 1e4
    )

    # mixed precision
    amp_scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # teacher
    if cfg.teacher_paths is not None:
        teacher_models = []
        for path in cfg.teacher_paths.split(','):
            checkpoint = torch.load(path, map_location='cpu')
            teacher_model = RFCXModel(CFG(checkpoint['cfg'])).to(rank).eval()
            _ = teacher_model.load_state_dict(checkpoint['model'], strict=False)
            teacher_models.append(teacher_model)
            if verbose_cond:
                print(_)
                print(f'Loaded Teacher Model from {path}')

    # pretrained
    if cfg.pretrained_path is not None:
        checkpoint = torch.load(cfg.pretrained_path, map_location='cpu')
        _ = model.load_state_dict(checkpoint['model'], strict=False)
        if verbose_cond:
            print(_)
            print(f'Loaded Pretrained Weights from {cfg.pretrained_path}')

    if cfg.resume:
        checkpoint = torch.load(os.path.join(cfg.logdir, 'last.pth'), map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_val_metric = torch.load(os.path.join(cfg.logdir, 'best.pth'), map_location='cpu')['val_metric']
        amp_scaler.load_state_dict(checkpoint['amp_scaler'])
        if verbose_cond: print(f'Resume from {cfg.logdir}/last.pth. Best val metric: {best_val_metric}')
    else:
        start_epoch = 0
        best_val_metric = -np.inf

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    t0 = time()
    cams = [defaultdict(list) for _ in range(24)]
    # loop
    for epoch in range(start_epoch, cfg.epochs):

        t00 = time()

        # check current learning rates
        current_lrs = [x["lr"] for x in optimizer.param_groups]
        if verbose_cond: print(f'EPOCH: {epoch} | LRs: {set(current_lrs)}')

        # train
        model.train()
        train_loss = 0.0
        if verbose_cond & (epoch == 0):
            iterator = tqdm(enumerate(trn_loader), total=len(trn_loader))
        else:
            iterator = enumerate(trn_loader)
        for i, data in iterator:
            for k in data.keys():
                data[k] = data[k].to(rank)

            # mixup
            # if cfg.mixup > 0:
            #     lam = np.random.beta(cfg.mixup, cfg.mixup)
            #     index = torch.randperm(data['audio'].size()[0]).to(rank)
            #     data['audio'] = lam * data['audio'] + (1 - lam) * data['audio'][index]
            #     data['labels'] = torch.maximum(data['labels'], data['labels'][index])
            #     data['masks'] = torch.maximum(data['masks'], data['masks'][index])
            if np.random.rand() < cfg.mixup:
                index = torch.randperm(data['audio'].size()[0]).to(rank)
                data['audio'] = data['audio'] * 0.5 + data['audio'][index] * 0.5
                data['labels'] = torch.maximum(data['labels'], data['labels'][index])
                data['masks'] = torch.maximum(data['masks'], data['masks'][index])

            if cfg.teacher_paths is not None:
                with torch.no_grad():
                    teacher_probs = [torch.sigmoid(teacher_model(data['audio']))+cfg.teacher_adder for teacher_model in teacher_models]
                    teacher_probs = torch.stack(teacher_probs, dim=0).mean(dim=0).clamp(0, 1)
                    if cfg.teacher_pred_thresh:
                        teacher_probs = (teacher_probs > cfg.teacher_pred_thresh).float()
                    data['labels'] = torch.where(data['masks'] == 1, data['labels'], teacher_probs)

            with torch.cuda.amp.autocast(enabled=cfg.amp):
                logits = model(data['audio'])
                loss = criterion(logits, data['labels'], data['masks'], focal=cfg.focal, pos_weight=cfg.pos_weight,
                                 lsoft=cfg.lsoft, lsep_weight=cfg.lsep_weight, default_label=cfg.default_label)

            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            # gather all
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            # add batch loss and metric
            if not torch.isfinite(loss):
                joblib.dump((data, logits, loss, grad_norm), 'error.jl')
                print('LOSS NOT FINITE')
                exit()
            train_loss += loss.item() * len(data['audio']) / (len(trn_loader.dataset))
        str_train_loss = np.round(train_loss, 6)
        if verbose_cond:
            print(f'(trn) LOSS: {str_train_loss}', end=' || ')

        # visualize cam
        # with torch.no_grad():
        #     dataset = val_loader.dataset
        #     for sid in range(24):
        #         rid_lst = rid_by_sid[sid]
        #         rid_lst = [x for x in rid_lst if x in dataset.df.index][:3]
        #         for rid in rid_lst:
        #             idx = np.argwhere(dataset.df.index == rid).flatten()[0]
        #             cur = dataset[idx]
        #             feature_hook = FeatureHook(model.module.encoder.act2)
        #             model = model.eval()
        #             audio = cur['audio'].unsqueeze(0).to(rank)
        #             pred = model(audio)[0, sid]
        #             feature_hook.remove()
        #             fc_weights = np.squeeze(list(model.module.heads[sid].parameters())[2].cpu().numpy())
        #             feature_map = feature_hook.features
        #             b, c, h, w = feature_map.shape
        #             cam = get_cam(feature_map.reshape(24, 1, c, h, w)[sid], fc_weights)
        #             spectrogram = model.module.logmel_extractors[sid](model.module.spectrogram_extractor(audio))[0, 0].cpu().numpy()
        #             cams[sid][rid].append((pred.item(), blend_cam(spectrogram, cam, (1000, 100))))

        # validate
        if (epoch % cfg.val_freq == 0) and (not cfg.full):
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            val_masks = []
            with torch.no_grad():
                iterator = tqdm(val_loader) if verbose_cond & (epoch == 0) else val_loader
                for data in iterator:
                    for k in data.keys():
                        data[k] = data[k].to(rank)

                    # if cfg.teacher_paths is not None:
                    #     teacher_probs = [torch.sigmoid(teacher_model(data['audio'])) for teacher_model in teacher_models]
                    #     teacher_probs = torch.stack(teacher_probs, dim=0).mean(dim=0)
                    #     data['labels'] = torch.where(data['masks'] == 1, data['labels'], teacher_probs)

                    logits = model(data['audio'])
                    # gather all
                    g_cls_logits = [torch.zeros(logits.shape).to(rank) for _ in range(world_size)]
                    dist.all_gather(g_cls_logits, logits)
                    g_cls_logits = torch.cat(g_cls_logits, dim=0)
                    g_labels = [torch.zeros(data['labels'].shape).to(rank) for _ in range(world_size)]
                    dist.all_gather(g_labels, data['labels'])
                    g_labels = torch.cat(g_labels, dim=0)
                    g_masks = [torch.zeros(data['masks'].shape).to(rank) for _ in range(world_size)]
                    dist.all_gather(g_masks, data['masks'])
                    g_masks = torch.cat(g_masks, dim=0)
                    # add batch preds, targets, loss
                    val_preds += g_cls_logits.cpu().tolist()
                    val_targets += g_labels.cpu().tolist()
                    val_masks += g_masks.cpu().tolist()
                    loss = criterion(g_cls_logits, g_labels, g_masks, lsep_weight=cfg.lsep_weight, default_label=cfg.default_label)
                    val_loss += loss.item() * len(data['audio']) * world_size / len(val_loader.dataset)

            # compute validation metric
            val_preds, val_targets, val_masks = np.array(val_preds), np.array(val_targets), np.array(val_masks)
            rel_index = np.argwhere(val_masks)
            tmp = tuple(rel_index.T)
            val_metric = kaggle_metric(np.expand_dims(val_preds[tmp], 0), np.expand_dims(val_targets[tmp], 0))
            str_val_metric = np.round(val_metric, 6)
            str_val_loss = np.round(val_loss, 6)
        else:
            val_loss, val_metric, val_preds, str_val_loss, str_val_metric = None, None, None, None, None

        # add log
        tmp_info = {
            'epoch': epoch,
            'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'amp_scaler': amp_scaler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'learning_rates': current_lrs,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metric': val_metric,
            'val_preds': val_preds,
            'cfg': cfg.__dict__
        }
        if val_metric is not None:
            if val_metric > best_val_metric:
                # save info when improved
                best_val_metric = val_metric
                if rank == 0: torch.save(tmp_info, f'{cfg.logdir}/best.pth')
        if rank == 0:
            with open(f'{cfg.logdir}/log.txt', 'a') as f:
                f.write(
                    f'{epoch} - {str_train_loss} - {str_val_loss} - {str_val_metric} - {set(current_lrs)}\n')

            torch.save(tmp_info, f'{cfg.logdir}/last.pth')
        if verbose_cond: print(f'(val) LOSS: {str_val_loss} | METRIC: {str_val_metric} || Runtime: {int(time() - t00)}')
        if verbose_cond and (epoch % cfg.val_freq == 0) and (not cfg.full):
            lst = []
            for i in range(24):
                tmp = roc_auc_score(np.round(val_targets[:, i]), val_preds[:, i])
                lst.append(np.round(tmp, 4))
                tmp = np.round(sigmoid(val_preds).mean(axis=0), 4).tolist()
            print(np.round(np.mean(tmp), 4), tmp)
            print(np.round(np.mean(lst), 4), lst)
    # if rank == 0:
    #     joblib.dump(cams, f'{cfg.logdir}/cams.jl')

    if verbose_cond:
        runtime = int(time() - t0)
        print(f'Best Val Score: {best_val_metric}')
        print(f'Runtime: {runtime}')
        # save current argument settings and result to file
        if os.path.exists('history.csv'):
            history = pd.read_csv('history.csv')
        else:
            history = pd.DataFrame(columns=list(cfg.__dict__.keys()))

        info = cfg.__dict__
        info['score'] = best_val_metric
        info['runtime'] = runtime
        info['n_parameters'] = n_parameters
        history = history.append(info, ignore_index=True)

        # rearrange columns
        cols = history.columns.tolist()
        front_cols = ['logdir', 'score', 'runtime']
        for c in front_cols:
            cols.remove(c)
        cols = front_cols + cols
        history = history[cols]

        history.to_csv('history.csv', index=False)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--infer_batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--encoder', type=str)
    parser.add_argument('--win_seconds', type=int)
    parser.add_argument('--min_overlap_ratio', type=float)
    parser.add_argument('--default_label', type=float, default=0.)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--data_frac', type=float)
    parser.add_argument('--amp', type=int)
    parser.add_argument('--gpu_numbers', type=str)
    parser.add_argument('--pretrained_path', type=str)
    parser.add_argument('--data_dir', type=str, default='../../input/kaggle/')
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--init_random', type=bool, default=False)
    parser.add_argument('--focal', type=float, default=0.)
    parser.add_argument('--pos_weight', type=float, default=1.)
    parser.add_argument('--aug', type=int, default=1)
    parser.add_argument('--pseudo_data_dir', type=str)
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--sr', type=int, default=48000)
    parser.add_argument('--hop_length', type=int, default=256)
    parser.add_argument('--n_mels', type=int, default=32)
    parser.add_argument('--mixup', type=float, default=0.)
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--use_fp', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--uncertain_weight', type=float, default=1.)
    parser.add_argument('--full', type=float, default=0.)
    parser.add_argument('--lsoft', type=float, default=0.)
    parser.add_argument('--lsep_weight', type=float, default=0)
    parser.add_argument('--random_sample_prob', type=float, default=0)
    parser.add_argument('--teacher_paths', type=str)
    parser.add_argument('--fminfmax_offset', type=float, default=100)
    parser.add_argument('--pseudo', type=int, default=0)
    parser.add_argument('--teacher_adder', type=float, default=0)
    parser.add_argument('--multicrop', type=int, default=1)
    parser.add_argument('--teacher_pred_thresh', type=float)
    parser.add_argument('--freq_bn', type=int, default=0)

    args = parser.parse_args()
    args.amp = bool(args.amp)
    cfg = CFG(vars(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_numbers
    make_reproducible(cfg.seed)

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main_worker, nprocs=world_size, args=(world_size, cfg))
