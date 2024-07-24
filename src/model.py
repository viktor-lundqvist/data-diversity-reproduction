import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
import pytorch_lightning as pl

import utils as u
from collect_data import LinRegData

class GPT2(pl.LightningModule):
    def __init__(self, n_dims_in, n_positions, config, n_embd=128, n_layer=8, n_head=2, n_dims_out=1):
        super(GPT2, self).__init__()
        configuration = GPT2Config(
            n_positions=2048,  # set to sthg large advised
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']

        self.n_positions = n_positions # Don't know what this is doing
        self.n_dims_in = n_dims_in # Should perhaps just go through config
        self.n_dims_out = n_dims_out
        self._read_in = nn.Linear(n_dims_in, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, n_dims_out)

    def forward(self, batch):
        ys = batch["ys"]
        seq = batch["seq"]

        embeds = self._read_in(seq)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        yhats = u.seq_to_targets(prediction)

        return yhats

    def training_step(self, batch, batch_idx):
        ys = batch["ys"]
        seq = batch["seq"]

        embeds = self._read_in(seq)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        yhats = u.seq_to_targets(prediction)

        loss = (yhats - ys) ** 2
        loss = loss.mean()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate,
                                    weight_decay=self.weight_decay)
        return optimizer

if __name__ == "__main__":
    pass
