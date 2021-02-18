import numpy as np
import pandas as pd
import torch
import argparse
import joblib
import os
from tqdm import tqdm

from model import RFCXModel, kaggle_metric
from data import get_loader
from train import CFG
from utils import sigmoid
from scipy.stats import rankdata


def load_model(path, device, cfg):
    ckp = torch.load(path, map_location='cpu')
    print(path)
    ckp['cfg']['init_random'] = True
    print(ckp['cfg'])
    model = RFCXModel(CFG(ckp['cfg']))
    print(model.load_state_dict(ckp['model'], strict=False))
    model.to(device)
    model.eval()
    if cfg.dataparallel:
        model = torch.nn.DataParallel(model)
    print(ckp['val_metric'])
    print('')
    return model, ckp['cfg']


def infer(cfg):
    # define device
    device = 'cpu' if (cfg.cpu or (not torch.cuda.is_available())) else 'cuda'
    print(device)

    # load models
    models = []
    fnames = os.listdir(cfg.model_dir)
    for j, fname in enumerate(fnames):
        path = os.path.join(cfg.model_dir, fname)
        model, ckp_cfg = load_model(path, device, cfg)
        models.append(model)
        if j == 0:
            existing_keys = cfg.__dict__.keys()
            for k, v in ckp_cfg.items():
                if k not in existing_keys:
                    cfg.__setattr__(k, v)

    if cfg.model_per_species:
        assert len(models) % 24 == 0

    # create loader
    if cfg.fold is None:
        tmp_fnames = sorted(os.listdir(os.path.join(cfg.data_dir, 'test')))
        uids = ['.'.join(x.split('.')[:-1]) for x in tmp_fnames]
        paths = [os.path.join(cfg.data_dir, x) for x in tmp_fnames]
    else:
        folds = joblib.load(cfg.folds_path)
        uids = folds[cfg.fold][1]
        paths = [os.path.join(cfg.data_dir, x+'.flac') for x in uids]
    print(f'# uids: {len(paths)}')
    loader = get_loader(cfg, mode='test' if cfg.fold is None else 'valid', distributed=False)

    if cfg.weights is None:
        weights = np.ones(len(models))
    else:
        weights = np.array(cfg.weights.split('-'), dtype=np.float32)
    weights = weights/weights.sum()
    print('Ensemble weights:', weights)

    # infer
    preds = []
    trues = []
    masks = []
    with torch.no_grad():
        for data in tqdm(loader):
            if cfg.fold is not None:
                trues.append(data['labels'].numpy())
                masks.append(data['masks'].numpy())
            batch_preds = np.zeros((len(data['audio']), 24))
            for model, weight, fname in zip(models, weights, fnames):
                x = data['audio'].to(device)
                if cfg.full_clip:
                    pred = model(x).cpu().numpy()
                else:
                    b, n, c = x.shape
                    per_window_pred = model(x.view(b*n, c)).view(b, n, -1)
                    pred = per_window_pred.max(dim=1)[0].cpu().numpy()
                if cfg.rank_mean: pred = rankdata(pred, axis=1)
                # pred = (pred-pred.min(axis=1, keepdims=True))/(pred.max(axis=1, keepdims=True)-pred.min(axis=1, keepdims=True))
                # pred = np.array([(one[:, np.newaxis] - one).mean(axis=1) for one in pred])
                if cfg.model_per_species:
                    sid = int(fname[1:])
                    batch_preds[:, sid] += pred[:, 0] * weight
                else:
                    batch_preds += pred * weight
            preds.append(batch_preds)
    preds = np.concatenate(preds, axis=0)
    if len(trues) > 0: trues = np.concatenate(trues, axis=0)
    if len(masks) > 0: masks = np.concatenate(masks, axis=0)
    return uids, preds, trues, masks


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--infer_batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--cpu', type=int, default=0)
    parser.add_argument('--folds_path', type=str, default='folds.jl')
    parser.add_argument('--train_df_path', type=str, default='../../input/kaggle/train.csv')
    parser.add_argument('--dataparallel', type=int, default=0)
    parser.add_argument('--slide_offset', type=int)
    parser.add_argument('--model_per_species', type=int)
    parser.add_argument('--full_clip', type=int, default=1)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--rank_mean', type=int, default=0)
    args = parser.parse_args()

    cfg = CFG(vars(args))

    uids, preds, trues, masks = infer(cfg)

    if args.fold is not None:
        rel_index = np.argwhere(masks)
        tmp = tuple(rel_index.T)
        val_metric = kaggle_metric(np.expand_dims(preds[tmp], 0), np.expand_dims(trues[tmp], 0))
        print(f'Fold{args.fold} SCORE', val_metric)

    # make submission csv
    label_cols = [f's{i}' for i in range(24)]
    submission = pd.DataFrame(preds, index=uids, columns=label_cols)
    submission.index = submission.index.rename('recording_id')
    submission.to_csv(os.path.join(args.out_dir, 'submission.csv'), index=True)


