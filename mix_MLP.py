import os
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import torchmetrics


class MLP(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes, lr=0.0001, weight_decay=0.02):
        super().__init__()
        self.metric = torchmetrics.functional.auroc
        self.lr = lr
        self.automatic_optimization = False
        self.weight_decay = weight_decay
        self.MLPstack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            #nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.MLPstack(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = nn.functional.binary_cross_entropy(pred.view(pred.shape[0]), y.float())
        mlp_opt = self.optimizers()
        mlp_opt.zero_grad()
        self.manual_backward(loss)
        mlp_opt.step()
        values = {'training_loss': loss}
        self.log_dict(values, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = nn.functional.binary_cross_entropy(pred.view(pred.shape[0]), y.float())
        val_auc = self.metric(pred.view(pred.shape[0]), y, task='binary')
        values = {'val_loss': loss, 'val_auc': val_auc}
        self.log_dict(values, prog_bar=True, on_epoch=True)
        return val_auc

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = nn.functional.binary_cross_entropy(pred.view(pred.shape[0]), y.float())
        test_auc = self.metric(pred.view(pred.shape[0]), y, task='binary')
        values = {'test_loss': loss, 'test_auc': test_auc}
        self.log_dict(values, prog_bar=True, on_epoch=True)
        return test_auc
    
    def on_train_epoch_end(self):
        mlp_sch = self.lr_schedulers() 
        mlp_sch.step(self.trainer.callback_metrics["val_loss"])

    def configure_optimizers(self):
        mlp_opt = torch.optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)
        mlp_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(mlp_opt, factor=0.3, patience=70)
        return [mlp_opt], [mlp_scheduler] 
