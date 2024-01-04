import os
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import optim, nn, utils, Tensor
from vime_utils import mask_generator, pretext_generator
import pytorch_lightning as pl
import torchmetrics


class EncoderSemi(pl.LightningModule):
    def __init__(self, trained_encoder, trained_mlp, lr=0.0001, p_m=0.3, beta=1.0, K=5, batch_size=15):
        super().__init__()
        self.metric = torchmetrics.functional.auroc
        self.lr = lr
        self.p_m = p_m
        self.beta = beta
        self.K = K
        self.batch_size = batch_size
        self.trained_encoder = trained_encoder
        self.trained_mlp = trained_mlp
        
    def forward(self, x):
        z = self.trained_encoder(x)
        pred = self.trained_mlp(z)
        return pred
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        u_loss = 0
        y_hat = self.forward(x)
        for i in range(self.K):
            mask = mask_generator(self.p_m, x)
            new_mask, corrupt_input = pretext_generator(mask, x)
            y_k = self.forward(corrupt_input)
            u_loss += nn.functional.mse_loss(y_k.view(y_k.shape[0]), y_hat)
        
        s_loss = nn.functional.binary_cross_entropy(y_hat.view(y_hat.shape[0]), y.float())
        u_loss /= self.K*self.batch_size

        loss = s_loss + u_loss*self.beta
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        u_loss = 0
        val_auc = 0
        y_hat = self.forward(x)
        val_auc += self.metric(y_hat.view(y_hat.shape[0]), y, task='binary')
        for i in range(self.K):
            mask = mask_generator(self.p_m, x)
            new_mask, corrupt_input = pretext_generator(mask, x)
            y_k = self.forward(corrupt_input)
            u_loss += nn.functional.mse_loss(y_k.view(y_k.shape[0]), y_hat)
            val_auc += self.metric(y_k.view(y_k.shape[0]), y, task='binary')
        
        s_loss = nn.functional.binary_cross_entropy(y_hat.view(y_hat.shape[0]), y.float())
        u_loss /= self.K
        val_auc /= (self.K + 1)

        loss = s_loss + u_loss*self.beta
        val_values = {'val_auc': val_auc, 'val_loss': loss, 's_loss': s_loss, 'u_loss': u_loss}
        self.log_dict(val_values, prog_bar=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        u_loss = 0
        val_auc = 0
        y_hat = self.forward(x)
        val_auc += self.metric(y_hat.view(y_hat.shape[0]), y, task='binary')
        for i in range(self.K):
            mask = mask_generator(self.p_m, x)
            new_mask, corrupt_input = pretext_generator(mask, x)
            y_k = self.forward(corrupt_input)
            u_loss += nn.functional.mse_loss(y_k.view(y_k.shape[0]), y_hat)
            val_auc += self.metric(y_k.view(y_k.shape[0]), y, task='binary')
        
        s_loss = nn.functional.binary_cross_entropy(y_hat.view(y_hat.shape[0]), y.float())
        u_loss /= self.K
        val_auc /= (self.K + 1)

        loss = s_loss + u_loss*self.beta
        val_values = {'test_auc': val_auc, 'test_loss': loss, 'test_s_loss': s_loss, 'test_u_loss': u_loss}
        self.log_dict(val_values, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)