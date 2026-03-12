import random
import torch
import pandas as pd
import os
from pathlib import Path
import time  # 引入 time 模块来观察加载时间

import torch.utils.data as data
from torch.utils.data import dataloader

import os
import time
import shutil
import pandas as pd
import torch
import torch.utils.data as data
from types import SimpleNamespace


class ClsVlmData(data.Dataset):
    """
    一个在初始化时将所有特征数据预加载到内存中的PyTorch Dataset。

    Args:
        dataset_cfg (object): 包含数据集配置的对象。
                              需要有 .nfold, .fold, .data_high_dir, .label_dir,
                              .n_shot, .sampling_seed 等属性。
        state (str): 'train', 'val', 或 'test'，用于确定加载哪个数据集。
    """

    def __init__(self, dataset_cfg=None, state=None):
        self.__dict__.update(locals())  # 将所有输入参数设为类属性

        # 1. ----> 基本配置和元数据加载 (逻辑不变)
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_high_path = self.dataset_cfg.data_high_dir
        self.csv_dir = os.path.join(self.dataset_cfg.label_dir, f'fold{self.fold}.csv')
        self.concept_dit = self.dataset_cfg.concept_dir
        self.slide_data = pd.read_csv(self.csv_dir)

        # 这个文本数据被所有样本共享，在初始化时加载一次是最高效的
        self.text_data = torch.load(self.concept_dit)

        # 特殊数据集的标志位 (逻辑不变)
        self.is_nsclc = 'NSCLC' in self.feature_high_path
        self.is_rcc = 'RCC' in self.feature_high_path

        # 2. ----> 根据 'state' 确定要使用的数据子集 (逻辑不变)
        n_shot = self.dataset_cfg.n_shot
        sampling_seed = int(self.dataset_cfg.sampling_seed)

        if state == 'train':
            shot_samples_dir = os.path.join(self.dataset_cfg.label_dir, 'shot_samples')
            os.makedirs(shot_samples_dir, exist_ok=True)
            shot_samples_file = os.path.join(shot_samples_dir, f'fold{self.fold}_shot{n_shot}_seed{sampling_seed}.csv')

            if os.path.exists(shot_samples_file):
                print(f"[{state.upper()}] Loading existing shot samples from {shot_samples_file}")
                selected_df = pd.read_csv(shot_samples_file)
            else:
                print(f"[{state.upper()}] Creating new shot samples with seed {sampling_seed} and n_shot {n_shot}")
                train_df = pd.DataFrame({
                    'train_slide_id': self.slide_data['train_slide_id'].dropna(),
                    'train_label': self.slide_data['train_label'].dropna()
                }).reset_index(drop=True)
                selected_data = [group.sample(n=min(n_shot, len(group)), random_state=sampling_seed) for _, group in
                                 train_df.groupby('train_label')]
                selected_df = pd.concat(selected_data).reset_index(drop=True)
                selected_df.to_csv(shot_samples_file, index=False)
                print(f"[{state.upper()}] Shot samples saved to {shot_samples_file}")

            self.data = selected_df['train_slide_id']
            self.label = selected_df['train_label']
        elif state == 'val':
            self.data = self.slide_data['val_slide_id'].dropna().reset_index(drop=True)
            self.label = self.slide_data['val_label'].dropna().reset_index(drop=True)
        elif state == 'test':
            self.data = self.slide_data['test_slide_id'].dropna().reset_index(drop=True)
            self.label = self.slide_data['test_label'].dropna().reset_index(drop=True)
        else:
            raise ValueError(f"Invalid state '{state}'. Must be 'train', 'val', or 'test'.")

        # 3. ----> 整合 slide_id 和 label (逻辑不变)
        self.split_data = pd.concat([self.data, self.label], ignore_index=True, axis=1)
        self.split_data.columns = ['slide_id', 'label']
        self.split_data['slide_id'] = self.split_data['slide_id'].astype(str)
        self.slide_id = self.split_data['slide_id'].values
        self.label = self.split_data['label'].values.astype(int)

        # 4. ========================【核心修改：预加载数据到内存】========================
        self.preloaded_features = {}
        print(f"[{state.upper()}] Pre-loading {len(self.slide_id)} feature files into memory. This may take a while...")
        start_time = time.time()

        for idx, slide_id_str in enumerate(self.slide_id):
            current_label = self.label[idx]

            # 使用与原 __getitem__ 中完全相同的逻辑来构建每个文件的正确路径
            feature_path = self.feature_high_path
            clean_slide_id = slide_id_str.replace('.0', '') if 'UBC' in feature_path else slide_id_str

            if self.is_nsclc:
                nsclc_map = {0: 'LUSC', 1: 'LUAD'}
                feature_path = feature_path.replace('NSCLC', nsclc_map.get(current_label, ''))
            if self.is_rcc:
                rcc_map = {0: 'KIRC', 1: 'KICH', 2: 'KIRP'}
                feature_path = feature_path.replace('RCC', rcc_map.get(current_label, ''))

            full_path = os.path.join(feature_path, f'{clean_slide_id}.pt')

            if os.path.exists(full_path):
                # 加载特征并以 slide_id 为键，存储在字典中
                self.preloaded_features[slide_id_str] = torch.load(full_path)
            else:
                print(f"Warning: File not found and will be skipped. Path: {full_path}")
                self.preloaded_features[slide_id_str] = None  # 标记为None，在__getitem__中处理

        end_time = time.time()
        print(f"[{state.upper()}] Finished pre-loading. Time taken: {end_time - start_time:.2f} seconds.")
        # =================================================================================

    def __len__(self):
        return len(self.slide_id)

    def __getitem__(self, idx):
        slide_id = str(self.slide_id[idx])
        label = self.label[idx]

        features = self.preloaded_features.get(slide_id)

        if features is None:
            raise FileNotFoundError(f"Features for slide_id {slide_id} were not preloaded or not found.")

        return (features, self.text_data), torch.tensor(label, dtype=torch.long)
