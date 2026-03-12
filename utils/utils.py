from pathlib import Path

#---->
import yaml
from addict import Dict
def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

import os
from pytorch_lightning import loggers as pl_loggers

def load_loggers(cfg):
    'log_path: /multi_scale/logs/ config: “BLCA/AMIL.yaml” '

    '''
    os.path.split(suvival/BLCA/AMIL/AMIL.yaml)-->('Suvival/BLCA/AMIL', 'AMIL.yaml')
    suvival/BLCA/AMIL/AMIL.yaml
    os.path.split(cfg.config)[0]: BLCA
    os.path.split(cfg.config)[1]: AMIL
    os.path.split(cfg.config)[2][:-5]: AMIL
    '''
    log_path = cfg.General.log_path
    os.makedirs(log_path, exist_ok=True)

    task = os.path.split(cfg.config)[0].split('/')[0]  # BLCA/AMIL/AMIL.yaml -> BLCA
    dataset_name = os.path.split(cfg.config)[0].split('/')[1]  # BLCA/AMIL/AMIL.yaml -> BLCA
    model_name = os.path.split(cfg.config)[0].split('/')[2]  # AMIL
    version_identifier = os.path.split(cfg.config)[1][:-5] # AMIL.yaml -> AMIL

    cfg.log_path = os.path.join(log_path, task, dataset_name, model_name, version_identifier, f'Shot{cfg.Data.n_shot}', f'fold{cfg.Data.fold}')

    print(f'---->Log dir: {cfg.log_path}')

    # ---->TensorBoard
    tb_logger = pl_loggers.TensorBoardLogger(
        os.path.join(log_path, task, dataset_name, model_name, version_identifier),
        name=f'Shot{cfg.Data.n_shot}', version=f'fold{cfg.Data.fold}',
        log_graph=True, default_hp_metric=False
    )

    # ---->CSV
    csv_logger = pl_loggers.CSVLogger(
        os.path.join(log_path, task, dataset_name, model_name, version_identifier),
        name=f'Shot{cfg.Data.n_shot}', version=f'fold{cfg.Data.fold}'
    )

    return [tb_logger, csv_logger]

def load_log_path(cfg):
    '''
    BLCA/AMIL/AMIL.yaml
    os.path.split(cfg.config)[0]: BLCA
    os.path.split(cfg.config)[1]: AMIL
    os.path.splitext(os.path.split(cfg.config)[2])[0]: AMIL
    '''
    log_path = cfg.General.log_path
    os.makedirs(log_path, exist_ok=True)

    task = os.path.split(cfg.config)[0].split('/')[0]  # BLCA/AMIL/AMIL.yaml -> BLCA
    dataset_name = os.path.split(cfg.config)[0].split('/')[1]  # BLCA/AMIL/AMIL.yaml -> BLCA
    model_name = os.path.split(cfg.config)[0].split('/')[2]  # AMIL
    version_identifier = os.path.split(cfg.config)[1][:-5] # AMIL.yaml -> AMIL

    log_path = os.path.join(log_path, task, dataset_name, model_name, version_identifier, f'Shot{cfg.Data.n_shot}')
    print(f'---->Log dir: {log_path}')

    return log_path

import logging

def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    for level in (
            "debug",
            "info",
            "warning",
            "error",
            "exception",
            "fatal",
            "critical",
    ):
        setattr(logger, level, getattr(logger, level))

    return logger

# ---->Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def load_callbacks(cfg):
    Mycallbacks = []
    # Make output path
    output_path = cfg.log_path
    os.makedirs(output_path, exist_ok=True)

    # Determine the monitor metric based on the task
    monitor_metric = 'val_acc' if cfg.task == 'cls' else 'val_loss'

    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        min_delta=0.00,
        patience=cfg.General.patience,
        verbose=True,
        mode='max' if monitor_metric == 'val_acc' else 'min'
    )
    Mycallbacks.append(early_stop_callback)

    if cfg.General.server == 'train':
        Mycallbacks.append(ModelCheckpoint(
            monitor=monitor_metric,
            dirpath=output_path,
            filename='{epoch:02d}-{' + monitor_metric + ':.4f}',
            verbose=True,
            save_last=True,
            save_top_k=1,
            mode='max' if monitor_metric == 'val_acc' else 'min',
            save_weights_only=True
        ))

    return Mycallbacks

#---->
import torch.nn as nn
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

#---->loss
import torch
def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y)+eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss

class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None): 
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)


# divide continuous time scale into k discrete bins in total,  T_cont \in {[0, a_1), [a_1, a_2), ...., [a_(k-1), inf)}
# Y = T_discrete is the discrete event time:
# Y = -1 if T_cont \in (-inf, 0), Y = 0 if T_cont \in [0, a_1),  Y = 1 if T_cont in [a_1, a_2), ..., Y = k-1 if T_cont in [a_(k-1), inf)
# discrete hazards: discrete probability of h(t) = P(Y=t | Y>=t, X),  t = -1,0,1,2,...,k
# S: survival function: P(Y > t | X)
# all patients are alive from (-inf, 0) by definition, so P(Y=-1) = 0
# h(-1) = 0 ---> do not need to model
# S(-1) = P(Y > -1 | X) = 1 ----> do not need to model
'''
Summary: neural network is hazard probability function, h(t) for t = 0,1,2,...,k-1
corresponding Y = 0,1, ..., k-1. h(t) represents the probability that patient dies in [0, a_1), [a_1, a_2), ..., [a_(k-1), inf]
'''

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1) 
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)

class CoxSurvLoss(object):
    def __init__(self):
        pass

    def __call__(self, hazards, S, Y, c, alpha=None):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(S)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = S[j] >= S[i]

        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * (1-c))
        return loss_cox


#---->metrics
from lifelines.statistics import logrank_test
import numpy as np
def cox_log_rank(hazards, labels, survtime_all):
    hazardsdata = hazards.reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata < median] = 1
    survtime_all = survtime_all.reshape(-1)
    idx = hazards_dichotomize == 0
    labels = labels
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return (pvalue_pred)

#---->test
import pandas as pd 
import torch
def _predictions_to_pycox(data, time_points=None):
    # predictions = {k: v['probabilities'] for k, v in data}
    predictions = {index: data[index][0] for index in range(len(data))}
    df = pd.DataFrame.from_dict(predictions)

    # Use predictions at same "time_points" for all models
    # Use MultiSurv's default output interval midpoints as default
    if time_points is None:
        time_points = np.arange(0, 4, 1)

    # Replace automatic index by time points
    df.insert(0, 'time', time_points)
    df = df.set_index('time')

    return df


#---->val loss
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F


def cross_entropy_torch(x, y):
    # 确保 x 和 y 都是 2D 张量
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 0:
        y = y.unsqueeze(0)

    # 确保 y 是长整型
    y = y.long()

    # 对整个批次应用 softmax
    x_softmax = F.softmax(x, dim=-1)

    # 获取每个样本对应的概率
    correct_probs = x_softmax[torch.arange(x.size(0)), y]

    # 计算对数并求平均
    log_probs = torch.log(correct_probs)
    loss = -torch.mean(log_probs)

    return loss