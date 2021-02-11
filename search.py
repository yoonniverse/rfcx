import subprocess
import optuna
import joblib
import os
import shutil
import numpy as np
from time import time
import torch

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial):
    t0 = time()
    params = {
        'logdir': 'optuna_tmp_logdir',
        'seed': 0,
        'batch_size': 128,
        'infer_batch_size': 256,
        'weight_decay': 0.01,
        'epochs': 20,
        'encoder': 'efficientnet_b0',
        'num_workers': 32,
        'data_frac': 1.,
        'amp': 0,
        'gpu_numbers': '"0, 1, 2, 3"',
        'data_dir': '../../input/kaggle',
        'val_freq': 1,
        'aug': 0,
        'n_fft': 2048,
        'sr': 48000,
        'use_fp': 1,
        'verbose': 0,
        'lr': trial.suggest_loguniform('lr', 0.0003, 0.001),
        'win_seconds': trial.suggest_int('win_seconds', 3, 15),
        'min_overlap_ratio': trial.suggest_uniform('min_overlap_ratio', 0.1, 1.0),
        'default_label': trial.suggest_uniform('default_label', 0, 0.1),
        'dropout': trial.suggest_uniform('dropout', 0, 0.5),
        'focal': trial.suggest_uniform('focal', 0, 1),
        'pos_weight': trial.suggest_uniform('pos_weight', 1, 10),
        'uncertain_weight': trial.suggest_uniform('uncertain_weight', 0, 1),
        'mixup': trial.suggest_uniform('mixup', 0, 0.5),
        'hop_length': trial.suggest_int('hop_length', 256, 384),
        'n_mels': trial.suggest_int('n_mels', 64, 128),
        'fmin': trial.suggest_int('fmin', 0, 90),
        'fmax': trial.suggest_int('fmax', 14000, 24000),
    }

    command = f"python train.py"
    for k, v in params.items():
        command += f" --{k} {v}"
    print(command)
    response = subprocess.call(command, shell=True)
    if response != 0:
        print("Something went wrong")
        return 0
    val_metric = torch.load('optuna_tmp_logdir/best.pth')['val_metric']
    shutil.rmtree('optuna_tmp_logdir')
    print(f'# Trial: {trial.number} | Val Metric: {np.round(val_metric, 6)} | Runtime: {int(time()-t0)}')

    trial.set_user_attr('command', command)

    return val_metric


def callback(study, _):
    joblib.dump(study, 'optuna_study.jl')


if os.path.exists(f'optuna_study.jl'):
    study = joblib.load(f'optuna_study.jl')
else:
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(n_startup_trials=50, multivariate=True))
study.optimize(objective, n_trials=1000, n_jobs=1, callbacks=[callback])

