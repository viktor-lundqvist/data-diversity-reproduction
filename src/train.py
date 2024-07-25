import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import argparse
import yaml
import hashlib
import time
import os

import utils as u
from model import GPT2
from collect_data import LinRegData

if __name__ == "__main__":
    wandb.login()

    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--resume_train', help='Boolean.', action='store_true')
    parser.add_argument('--config', type=u.file_path, required=True)
    args = parser.parse_args()
    resume_train = args.resume_train
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if resume_train:
        train_data = LinRegData(config, task_pool=torch.load(config['task_pool_path']))
    else:
        train_data = LinRegData(config)

    train_loader = train_data.batch_iterator()
    checkpoint_callback = ModelCheckpoint(every_n_train_steps=config['checkpoint_interval'], save_top_k=-1)
    wandb_logger = WandbLogger(
        log_model="all",
        project="Data Diversity Reproduction"
    )

    trainer = pl.Trainer(
        max_epochs=1,
        max_steps=config['train_steps'],
        accelerator="gpu",
        devices="auto",
        callbacks=[checkpoint_callback],
        logger=wandb_logger
    )

    model = GPT2(n_dims_in=config['dim'] + 1, n_positions=config['n_points'], config=config, n_dims_out=1)

    if resume_train:
        trainer.fit(model=model, train_dataloaders=train_loader, ckpt_path=config['ckpt_path'])
    else:
        run = wandb.init(
            # Set the project where this run will be logged
            project="Data Diversity Reproduction",
            # Track hyperparameters and run metadata
            config=config,
        )

        # Create the directory before wandb does, and fill with config and taskpool. Also save taskpool to wandb.
        run_dir = f"./Data Diversity Reproduction/{run.id}"
        os.mkdir(run_dir)
        task_pool_path = f"{run_dir}/task_pool.pt"
        torch.save(train_data.task_pool, task_pool_path)
        wandb.save(task_pool_path)
        with open(f"{run_dir}/config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        trainer.fit(model=model, train_dataloaders=train_loader)
