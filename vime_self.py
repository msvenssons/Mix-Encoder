import os
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import optim, nn, utils, Tensor
from vime_utils import mask_generator, pretext_generator, mixer
import pytorch_lightning as pl
import torchmetrics


class Encoder(pl.LightningModule):
    def __init__(self, input_size, hidden_size, lr=0.0001, p_m=0.3, alpha=1.0):
        super().__init__()
        self.lr = lr
        self.p_m = p_m
        self.alpha = alpha
        self.encoder_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.mask_stack = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
        self.feature_stack = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.ReLU()
        )

    def forward(self, x):
        mask = mask_generator(self.p_m, x, N0=False)
        new_mask, corrupt_input = pretext_generator(mask, x)

        z = self.encoder_stack(corrupt_input)
        mask_pred = self.mask_stack(z)
        feature_pred = self.feature_stack(z)
        return mask_pred, feature_pred, new_mask
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        mask_pred, feature_pred, new_mask = self.forward(x)
        mask_loss = nn.functional.binary_cross_entropy(mask_pred, new_mask)
        feature_loss = nn.functional.mse_loss(feature_pred, x)
        loss = mask_loss + feature_loss*self.alpha
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mask_pred, feature_pred, new_mask = self.forward(x)
        mask_loss = nn.functional.binary_cross_entropy(mask_pred, new_mask)
        feature_loss = nn.functional.mse_loss(feature_pred, x)
        val_loss = mask_loss + feature_loss*self.alpha
        val_values = {'val loss': val_loss, 'val mask loss': mask_loss, 'val feature loss': feature_loss}
        self.log_dict(val_values, prog_bar=True, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        mask_pred, feature_pred, new_mask = self.forward(x)
        mask_loss = nn.functional.binary_cross_entropy(mask_pred, new_mask)
        feature_loss = nn.functional.mse_loss(feature_pred, x)
        test_loss = mask_loss + feature_loss*self.alpha
        test_values = {'test_loss': test_loss, 'test_mask_loss': mask_loss, 'test_feature_loss': feature_loss}
        self.log_dict(test_values, prog_bar=True, on_epoch=True)
        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)
    


class EncoderMLP(pl.LightningModule):
    def __init__(self, trained_encoder, input_size, hidden_size, num_classes, lr=0.0001, weight_decay=0.02, en_weight_decay=0.01):
        super().__init__()
        self.metric = torchmetrics.functional.auroc
        self.lr = lr
        self.weight_decay = weight_decay
        self.en_weight_decay = en_weight_decay
        self.trained_encoder = trained_encoder
        self.automatic_optimization = False
        self.MLPstack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.trained_encoder(x)
        return self.MLPstack(z)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = nn.functional.binary_cross_entropy(pred.view(pred.shape[0]), y.float())
        mlp_opt, e_opt = self.optimizers()
        mlp_opt.zero_grad()
        e_opt.zero_grad()
        self.manual_backward(loss)
        mlp_opt.step()
        if self.current_epoch >= 300 and self.current_epoch <= 1000:
            e_opt.step()
        values = {'training_loss': loss}
        self.log_dict(values, prog_bar=True, on_epoch=True)
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
        mlp_sch, e_sch = self.lr_schedulers() 
        mlp_sch.step(self.trainer.callback_metrics["val_loss"])
        if self.current_epoch >= 300 and self.current_epoch <= 1000:
            e_sch.step(self.trainer.callback_metrics["val_loss"])

    def configure_optimizers(self):
        mlp_opt = torch.optim.Adam(self.MLPstack.parameters(), self.lr, weight_decay=self.weight_decay)
        e_opt = torch.optim.Adam(self.trained_encoder.parameters(), self.lr, weight_decay=self.en_weight_decay)
        mlp_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(mlp_opt, factor=0.3, patience=70)
        e_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(e_opt, factor=0.3, patience=70)
        return [mlp_opt, e_opt], [mlp_scheduler, e_scheduler] 


class MixEncoder(pl.LightningModule):
    def __init__(self, input_size=27, hidden_size=27, lr=0.0001):
        super().__init__()
        self.lr = lr
        self.encoder_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.mix_stack = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.restore_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, x):
        mixed, lamb = mixer(x)
        z = self.encoder_stack(mixed)
        mix_pred = self.mix_stack(z)
        restored = self.restore_stack(z)
        return mix_pred, lamb, restored
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        mix_pred, lamb, restored = self.forward(x)
        lamb_matrix = torch.full(mix_pred.shape, lamb)
        mix_loss = nn.functional.mse_loss(mix_pred, lamb_matrix)
        restore_loss = nn.functional.mse_loss(restored, x)
        loss = mix_loss + restore_loss
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mix_pred, lamb, restored = self.forward(x)
        lamb_matrix = torch.full(mix_pred.shape, lamb)
        mix_val_loss = nn.functional.mse_loss(mix_pred, lamb_matrix) 
        restore_val_loss = nn.functional.mse_loss(restored, x)
        val_loss = mix_val_loss + restore_val_loss
        val_values = {'val loss': val_loss, 'mix loss': mix_val_loss, 'restore loss': restore_val_loss}
        self.log_dict(val_values, prog_bar=True, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        mix_pred, lamb, restored = self.forward(x)
        lamb_matrix = torch.full(mix_pred.shape, lamb)
        mix_test_loss = nn.functional.mse_loss(mix_pred, lamb_matrix)
        restore_test_loss = nn.functional.mse_loss(restored, x)
        test_loss = mix_test_loss + restore_test_loss
        test_values = {'test_loss': test_loss}
        self.log_dict(test_values, prog_bar=True, on_epoch=True)
        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)
    

class CombinedEncoderMLP(pl.LightningModule):
    def __init__(self, trained_encoder, trained_encoder2, input_size=32, hidden_size=32, num_classes=1, lr=0.0001, weight_decay=0.02, en_weight_decay=0.01):
        super().__init__()
        self.metric = torchmetrics.functional.auroc
        self.lr = lr
        self.weight_decay = weight_decay
        self.en_weight_decay = en_weight_decay
        self.trained_encoder = trained_encoder
        self.trained_encoder2 = trained_encoder2
        self.automatic_optimization = False
        self.MLPstack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        z1 = self.trained_encoder(x)
        z2 = self.trained_encoder2(x)
        z = torch.cat((z1, z2), 1)
        return self.MLPstack(z)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = nn.functional.binary_cross_entropy(pred.view(pred.shape[0]), y.float())
        mlp_opt, e_opt, e2_opt = self.optimizers()
        mlp_opt.zero_grad()
        e_opt.zero_grad()
        e2_opt.zero_grad()
        self.manual_backward(loss)
        mlp_opt.step()
        if self.current_epoch >= 300 and self.current_epoch <= 1000:
            e_opt.step()
            e2_opt.step()
        values = {'training_loss': loss}
        self.log_dict(values, prog_bar=True, on_epoch=True)
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
        mlp_sch, e_sch, e2_sch = self.lr_schedulers() 
        mlp_sch.step(self.trainer.callback_metrics["val_loss"])
        if self.current_epoch >= 300 and self.current_epoch <= 1000:
            e_sch.step(self.trainer.callback_metrics["val_loss"])
            e2_sch.step(self.trainer.callback_metrics["val_loss"])

    def configure_optimizers(self):
        mlp_opt = torch.optim.Adam(self.MLPstack.parameters(), self.lr, weight_decay=self.weight_decay)
        e_opt = torch.optim.Adam(self.trained_encoder.parameters(), self.lr, weight_decay=self.en_weight_decay)
        e2_opt = torch.optim.Adam(self.trained_encoder2.parameters(), self.lr, weight_decay=self.en_weight_decay)
        mlp_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(mlp_opt, factor=0.3, patience=70)
        e_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(e_opt, factor=0.3, patience=70)
        e2_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(e2_opt, factor=0.3, patience=70)
        return [mlp_opt, e_opt, e2_opt], [mlp_scheduler, e_scheduler, e2_scheduler] 