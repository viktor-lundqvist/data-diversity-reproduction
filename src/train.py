import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model import GPT2
from collect_data import BatchIterator
from config import Config

if __name__ == "__main__":
    config = Config()

    model = GPT2(n_dims_in=config.dim + 1, n_positions=config.n_points, n_dims_out=1)

    train_loader = BatchIterator(config)
    checkpoint_callback = ModelCheckpoint(every_n_train_steps=config.checkpoint_interval, save_top_k=-1)    

    trainer = pl.Trainer(max_epochs=1, max_steps=config.train_steps, accelerator="gpu", devices="auto", callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=train_loader)
