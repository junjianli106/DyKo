import os
import argparse
from pathlib import Path
import numpy as np
import glob
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import setproctitle

from datasets import DataInterface
from models import ModelInterfaceProg, ModelInterfaceCls #, MyCallback
from glob import glob
from utils.utils import *




def parse_arguments():
    parser = argparse.ArgumentParser()

    # Basic configuration parameters
    parser.add_argument('--log_path',  default='/mnt/RAID_12T/all_logs/', type=str, help='log_path')
    parser.add_argument('--config', default='', type=str, help='Path to configuration file')
    parser.add_argument('--task', default='prog', type=str, choices=['cls', 'prog'],
                        help='Task type: classification or progression')
    parser.add_argument('--stage', default='train', type=str, help='Stage: train or test')

    # Training control parameters
    parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducibility')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of training epochs')
    parser.add_argument('--patience', default=40, type=int, help='Patience for early stopping')

    # Text info
    parser.add_argument('--text_prompt_path', default='./text_prompt', type=str, help='Path to text prompt file')
    parser.add_argument('--llm_model', default='Cluade-3.5-Sonnet', type=str, help='Gemini_2_flash')

    # Data-related parameters
    parser.add_argument('--fold', default=2, type=int, help='Cross-validation fold number')
    parser.add_argument('--n_shot', default=1, type=int, help='Number of samples per class (for few-shot learning)')

    # Hardware and performance parameters
    parser.add_argument('--gpus', default=[0], help='GPU indices to use')
    parser.add_argument('--grad_acc', default=4, type=int, help='Gradient accumulation steps')

    return parser.parse_args()

def initialize_trainer(cfg):
    return Trainer(
        num_sanity_val_steps=0,
        logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs=cfg.General.epochs,
        devices=[int(cfg.General.gpus)],
        accelerator="gpu",
        precision=cfg.General.precision,
        accumulate_grad_batches=cfg.General.grad_acc,
        check_val_every_n_epoch=1,
    )

def load_data_interface(cfg):
    return DataInterface(
        train_batch_size=cfg.Data.train_dataloader.batch_size,
        train_num_workers=cfg.Data.train_dataloader.num_workers,
        test_batch_size=cfg.Data.test_dataloader.batch_size,
        test_num_workers=cfg.Data.test_dataloader.num_workers,
        dataset_name=cfg.Data.dataset_name,
        dataset_cfg=cfg.Data,
    )

def load_model_interface(cfg):
    model_params = {
        'model': cfg.Model,
        'loss': cfg.Loss,
        'optimizer': cfg.Optimizer,
        'data': cfg.Data,
        'log': cfg.log_path
    }
    if cfg.task == 'prog':
        return ModelInterfaceProg(**model_params)
    elif cfg.task == 'cls':
        return ModelInterfaceCls(**model_params)
    else:
        raise NotImplementedError


def main(cfg):
    # Set random seed for reproducibility
    pl.seed_everything(cfg.Data.sampling_seed)

    # Initialize logging and callbacks
    cfg.load_loggers = load_loggers(cfg)
    cfg.callbacks = load_callbacks(cfg)

    # Initialize data module and trainer
    dm = load_data_interface(cfg)
    trainer = initialize_trainer(cfg)

    test_metrics_path = os.path.join(cfg.log_path, "test_metrics.csv")

    if cfg.General.server == 'train':
        # Training mode
        # Check if test_metrics.csv already exists
        if os.path.exists(test_metrics_path):
            print(f"Skipping training because {test_metrics_path} already exists in {cfg.log_path}.")
            return  # Exit the training block
        else:
            model = load_model_interface(cfg)
            trainer.fit(model=model, datamodule=dm)
    else:
        # Testing mode - Load checkpoints and evaluate
        # Determine which model class to use based on task
        model_classes = {
            'prog': ModelInterfaceProg,
            'cls': ModelInterfaceCls
        }

        model_class = model_classes[cfg.task]

        # Find all checkpoint files
        model_paths = [str(path) for path in glob(f'{cfg.log_path}/*.ckpt') if 'epoch' in str(path)]
        # Test each checkpoint
        for path in model_paths:
            print("\n" + "=" * 50)
            print(f"Testing checkpoint: {os.path.basename(path)}")
            print("=" * 50)

            # Check if test_metrics.csv exists in the log_path
            if os.path.exists(test_metrics_path):
                print(f"Skipping testing for {os.path.basename(path)} because {test_metrics_path} already exists.")
                continue  # Skip to the next checkpoint

            # Load model from checkpoint and test
            new_model = model_class.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            trainer.test(model=new_model, datamodule=dm)


if __name__ == '__main__':
    args = parse_arguments()
    # --config="Cls/BRACS-3/CoOpMIL/FOCUS.yaml"
    dataset_name = args.config.split('/')[1]
    model_name = args.config.split('/')[2]
    print(f"Dataset name: {dataset_name}")

    cfg = read_yaml(args.config)

    # Set basic configuration parameters
    cfg.task = args.task
    cfg.config = args.config
    cfg.General.server = args.stage
    cfg.General.log_path = args.log_path

    # Set training parameters
    cfg.General.seed = args.seed
    cfg.General.epochs = args.epochs
    cfg.General.grad_acc = args.grad_acc
    cfg.General.patience = args.patience
    cfg.General.gpus = args.gpus

    # Set data parameters
    cfg.Data.fold = args.fold
    cfg.Data.sampling_seed = args.seed
    cfg.Data.n_shot = args.n_shot

    # Set text configuration parameters
    if model_name != 'ViLaMIL':
        cfg.Model.text_prompt_path = os.path.join(str(args.text_prompt_path),  str(args.llm_model),  f'{dataset_name}_text_prompt.csv')
    else:
        cfg.Model.text_prompt_path = os.path.join(str(args.text_prompt_path),  str(args.llm_model),  f'{dataset_name}_two_scale_text_prompt.csv')

    long_title = (f'Stage: {cfg.General.server} {cfg.config} fold{cfg.Data.fold} shot{args.n_shot}')
    setproctitle.setproctitle(long_title)

    # Configure data directories based on host environment
    host_data_mapping = {
        'username-SYS-7048GR-TR': '/mnt/HDD_12T',
        '/public/home/hpc234701067': '/public/home/hpc234701067/data',
        '/home/username': '/home/username/data',
        '/home/username': '/home/username/data'
    }

    # Get hostname or home directory
    hostname = os.uname().nodename
    home_dir = os.path.expanduser("~")

    # Find the appropriate data directory
    found_match = False
    if hostname in host_data_mapping:
        target_dir = host_data_mapping[hostname]
        found_match = True
    elif home_dir in host_data_mapping:
        target_dir = host_data_mapping[home_dir]
        found_match = True

    if found_match:
        cfg.Data.data_high_dir = cfg.Data.data_high_dir.replace('/mnt/HDD_12T', target_dir).replace('/mnt/HDD_16T', target_dir)
        cfg.Data.data_low_dir = cfg.Data.data_low_dir.replace('/mnt/HDD_12T', target_dir).replace('/mnt/HDD_16T', target_dir)
    else:
        raise NotImplementedError("Unknown host environment. Please add your data paths to the configuration.")

    main(cfg)