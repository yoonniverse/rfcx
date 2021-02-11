from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import torch
import soundfile
import augmentation as aug
import os
import joblib
from audiomentations import Normalize


class RFCXDataset(Dataset):

    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        if mode == 'test':
            uids = sorted(os.listdir(os.path.join(cfg.data_dir, 'test')))
            self.df = pd.DataFrame(index=['.'.join(x.split('.')[:-1]) for x in uids])
        else:
            train_tp = pd.read_csv(f'{cfg.data_dir}/train_tp.csv', index_col='recording_id')
            train_tp['is_tp'] = True
            if self.cfg.use_fp:
                train_fp = pd.read_csv(f'{cfg.data_dir}/train_fp.csv', index_col='recording_id')
                train_fp['is_tp'] = False
                self.df = pd.concat((train_tp, train_fp), axis=0)
            else:
                self.df = train_tp
            if not cfg.full:
                folds = joblib.load('folds.jl')
                uids = folds[cfg.fold][0] if mode == 'train' else folds[cfg.fold][1]
                self.df = self.df.loc[self.df.index.unique().intersection(uids)]
            # self.df = self.df[self.df['species_id']==cfg.species_id]
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        uid = self.df.index[idx]
        base_path = os.path.join(self.cfg.data_dir, 'test' if self.mode == 'test' else 'train')
        path = os.path.join(base_path, str(uid)+'.flac')
        out = {}
        if self.mode in ['train', 'valid']:
            if self.mode == 'train':
                length = self.cfg.win_seconds * self.cfg.sr
                win_seconds = np.random.uniform(self.cfg.win_seconds*self.cfg.min_overlap_ratio, self.cfg.win_seconds)
            else:
                length = 60 * self.cfg.sr
                win_seconds = 60
            out['labels'] = np.ones(24, dtype=np.float32) * self.cfg.default_label
            out['masks'] = np.zeros(24, dtype=np.float32)
            if (self.mode == 'train') and (np.random.rand() < self.cfg.random_sample_prob):
                uid = np.random.choice(self.df.index)
                cur_df = self.df.loc[[uid]]
                start = np.random.randint(0, (60 - win_seconds) * self.cfg.sr)
                cur_df_len_thresh = 0
            else:
                cur_df = self.df.loc[[uid]]
                row = self.df.iloc[idx]
                # crop
                k, t0, t1, is_tp = row[['species_id', 't_min', 't_max', 'is_tp']].values
                min_overlap = min(t1-t0, win_seconds*self.cfg.min_overlap_ratio)
                if self.mode == 'train':
                    start = np.random.randint(
                        int(max(0, (t0-win_seconds+min_overlap)*self.cfg.sr)),
                        int((t1-min_overlap)*self.cfg.sr)
                    )
                else:
                    start = int(max(0, ((t0+t1-win_seconds)/2) * self.cfg.sr))
                out['labels'][k] = 1 if is_tp else 0
                out['masks'][k] = 1
                cur_df_len_thresh = 1

            # assign other possible labels
            if len(cur_df) > cur_df_len_thresh:
                for i, row in cur_df.iterrows():
                    if out['masks'][row['species_id']] == 1:
                        continue
                    min_overlap = min(row['t_max'] - row['t_min'], win_seconds*self.cfg.min_overlap_ratio)
                    if min(row['t_max'], start/self.cfg.sr + win_seconds) - max(row['t_min'], start/self.cfg.sr) > min_overlap-0.01:
                        k = row['species_id']
                        out['labels'][k] = 1 if row['is_tp'] else 0
                        out['masks'][k] = 1

            y, sr = soundfile.read(path, start=start, stop=min(start+length, self.cfg.sr*60))
            assert sr == self.cfg.sr
            if self.cfg.aug and self.mode == 'train':
                transform = aug.Compose([
                    aug.OneOf([
                        aug.GaussianNoiseSNR(min_snr=5., max_snr=20.),
                        aug.PinkNoiseSNR(min_snr=5., max_snr=20.)
                    ]),
                    aug.PitchShift(max_steps=2, sr=self.cfg.sr),
                    aug.TimeStretch(rate=0.1),
                    aug.TimeShift(sr=sr, max_shift_second=1),
                    aug.OneOf([
                        aug.VolumeControl(db_limit=10, mode="uniform"),
                        aug.VolumeControl(db_limit=10, mode="sine"),
                        aug.VolumeControl(db_limit=10, mode="cosine"),
                    ])
                ])
                y = transform(y).astype(np.float32)
            tmp = np.zeros(length, dtype=np.float32)
            y = y[:length]
            tmp[:len(y)] = y
            out['audio'] = tmp
        else:
            length = 60*self.cfg.sr
            y, sr = soundfile.read(path)
            assert sr == self.cfg.sr
            tmp = np.zeros(length, dtype=np.float32)
            y = y[:length]
            tmp[:len(y)] = y
            if self.cfg.full_clip:
                out['audio'] = tmp
            else:
                # out['audio'] = np.stack(np.split(tmp, length//(self.cfg.win_seconds*self.cfg.sr), axis=0))
                out['audio'] = np.lib.stride_tricks.sliding_window_view(tmp, self.cfg.win_seconds*self.cfg.sr)
                out['audio'] = out['audio'][::self.cfg.sr*self.cfg.slide_offset]

        # if self.mode == 'valid':
        #     out['labels'] = np.ones(24, dtype=np.float32)*self.cfg.default_label
        #     cur_df = self.df.loc[[uid]]
        #     for i, row in cur_df.iterrows():
        #         out['labels'][row['species_id']] = 1 if row['is_tp'] else 0

        # normalize
        out['audio'] = Normalize(p=1)(out['audio'], sr)

        for k in out.keys():
            out[k] = torch.from_numpy(out[k])

        return out


def get_loader(cfg, mode='train', distributed=True):
    batch_size = cfg.batch_size if mode == 'train' else cfg.infer_batch_size
    dataset = RFCXDataset(cfg, mode=mode)
    if distributed:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=cfg.num_workers, pin_memory=True,
                            sampler=torch.utils.data.distributed.DistributedSampler(dataset, shuffle=mode == 'train'))
    else:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=cfg.num_workers, shuffle=mode == 'train', pin_memory=True)
    return loader
