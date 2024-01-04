import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from sklearn.model_selection import KFold, train_test_split
import pytorch_lightning as pl
from vime_MLP import MLP
from vime_self import Encoder, EncoderMLP, MixEncoder, CombinedEncoderMLP
from vime_semi import EncoderSemi
from vime_utils import weights_init, kfolder, ksplit
import numpy as np
from torch import nn
from pytorch_lightning.loggers import CSVLogger


def MLPtrain(dataset, kfold=False, batch_size=15, epochs=20, lr=0.001, folds=5, seed=10, valsize=0.25, trainsize=0.75, weight_decay=0.02):
    # k-fold cross validation
    if kfold == True:
        np.random.seed(seed)
        kfold = KFold(n_splits=folds, shuffle=True)
        kfold.get_n_splits(dataset)
        average_val_auc = 0
        average_val_loss = 0
        for i, (train_index, val_index) in enumerate(kfold.split(dataset)):
            train_subsampler, val_subsampler = SubsetRandomSampler(train_index), SubsetRandomSampler(val_index)
            print(f'------------- FOLD {i+1} -------------')
            train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
            val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

            mlp = MLP(16, 16, 1, lr=lr, weight_decay=weight_decay) #27, 27 for N0, 16, 16 for diabetes
            trainer = pl.Trainer(limit_train_batches=batch_size, max_epochs=epochs, default_root_dir="MLP_fold_logs/")

            trainer.fit(mlp, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
            val_metrics = trainer.test(mlp, dataloaders=val_dataloader)
            average_val_auc += val_metrics[0]['test_auc']
            average_val_loss += val_metrics[0]['test_loss']
        average_val_auc /= folds
        average_val_loss /= folds
        print("------------------------- AVERAGES -------------------------")
        print(f"average_val_loss: {average_val_loss}, average_val_auc: {average_val_auc}")
        return average_val_auc, average_val_loss
    # variable training size
    else:
        np.random.seed(seed)
        train_index, val_index = train_test_split(list(range(len(dataset))), train_size=trainsize, test_size=valsize, random_state=seed)
        train_subsampler, val_subsampler = SubsetRandomSampler(train_index), SubsetRandomSampler(val_index)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
        mlp = MLP(16, 16, 1, lr=lr, weight_decay=weight_decay)
        trainer = pl.Trainer(limit_train_batches=batch_size, max_epochs=epochs, default_root_dir="MLP_dia_logs/")
        mlp.apply(weights_init)
        trainer.fit(mlp, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        val_metrics = trainer.test(mlp, dataloaders=val_dataloader)
        val_auc = val_metrics[0]['test_auc']
        val_loss = val_metrics[0]['test_loss']
        print("------------------------- RESULTS -------------------------")
        print(f"val_loss: {val_loss}, val_auc: {val_auc}")
        return val_auc, val_loss



def encodertrain(dataset, kfold=False, alpha=2.0, batch_size=15, epochs=20, lr=0.001, folds=5, seed=10, p_m=0.3, valsize=0.25, trainsize=0.75):
    if kfold == True:
        np.random.seed(seed)
        kfold2 = KFold(n_splits=folds, shuffle=True)
        kfold2.get_n_splits(dataset)
        for i, (train_index, val_index) in enumerate(kfold2.split(dataset)):
            train_subsampler = SubsetRandomSampler(train_index)
            val_subsampler = SubsetRandomSampler(val_index)
            print(f'------------- FOLD {i+1} -------------')
            self_train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
            self_val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

            encoder = Encoder(27, 27, lr=lr, alpha=alpha, p_m=p_m)
            en_trainer = pl.Trainer(limit_train_batches=batch_size, max_epochs=epochs, default_root_dir="VIMEEncoder_fold_logs/")

            en_trainer.fit(encoder, train_dataloaders=self_train_dataloader, val_dataloaders=self_val_dataloader)
            en_trainer.test(encoder, dataloaders=self_val_dataloader)
    elif kfold == False:
        np.random.seed(seed)
        train_index, val_index = train_test_split(list(range(len(dataset))), train_size=trainsize, test_size=valsize, random_state=seed)
        train_subsampler, val_subsampler = SubsetRandomSampler(train_index), SubsetRandomSampler(val_index)
        self_train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        self_val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
        encoder = Encoder(16, 16, lr=lr, alpha=alpha, p_m=p_m)
        en_trainer = pl.Trainer(limit_train_batches=batch_size, max_epochs=epochs, accelerator='cpu', default_root_dir="VIMEEncoder_logs/")

        en_trainer.fit(encoder, train_dataloaders=self_train_dataloader, val_dataloaders=self_val_dataloader)
        val_metrics = en_trainer.test(encoder, dataloaders=self_val_dataloader)
        val_loss = val_metrics[0]['test_loss']
        print("------------------------- RESULTS -------------------------")
        print(f"val_loss: {val_loss}")
        return val_loss


def encoderMLPtrain(dataset, checkpoint, kfold=False, batch_size=15, epochs=20, lr=0.001, folds=5, seed=10, valsize=0.25, trainsize=0.75, weight_decay=0.02, en_weight_decay=0.01, encoder_type="VIME"):
    if kfold == True:
        kfold = kfolder(dataset, folds, seed)
        average_val_auc = 0
        average_val_loss = 0
        for i, (train_index, val_index) in enumerate(kfold.split(dataset)):
            train_subsampler, val_subsampler = SubsetRandomSampler(train_index), SubsetRandomSampler(val_index)
            print(f'------------- FOLD {i+1} -------------')
            full_train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
            full_val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
            # load the encoder checkpoint
            if encoder_type == "VIME":
                trained_model = Encoder.load_from_checkpoint(checkpoint, input_size=16, hidden_size=16) # 35 --> 27
                with torch.no_grad():
                    trained_encoder = trained_model.encoder_stack.eval()
                encoder_mlp = EncoderMLP(trained_encoder, 16, 16, 1, lr=lr, weight_decay=weight_decay, en_weight_decay=en_weight_decay) # 35 --> 27
                full_trainer = pl.Trainer(limit_train_batches=batch_size, max_epochs=epochs, default_root_dir="VIMEEncoderMLP_fold_logs/")
            elif encoder_type == "mix":
                trained_model = MixEncoder.load_from_checkpoint(checkpoint, input_size=16, hidden_size=16)
                with torch.no_grad():
                    trained_encoder = trained_model.encoder_stack.eval()
                encoder_mlp = EncoderMLP(trained_encoder, 16, 16, 1, lr=lr, weight_decay=weight_decay, en_weight_decay=en_weight_decay) # 35 --> 27
                full_trainer = pl.Trainer(limit_train_batches=batch_size, max_epochs=epochs, default_root_dir="MixEncoderMLP_fold_logs/")
            encoder_mlp.apply(weights_init)
            full_trainer.fit(encoder_mlp, train_dataloaders=full_train_dataloader, val_dataloaders=full_val_dataloader)
            full_metrics = full_trainer.test(encoder_mlp, dataloaders=full_val_dataloader)
            average_val_auc += full_metrics[0]['test_auc']
            average_val_loss += full_metrics[0]['test_loss']
        average_val_auc /= folds
        average_val_loss /= folds
        print("------------------------- AVERAGES -------------------------")
        print(f"average_val_loss: {average_val_loss}, average_val_auc: {average_val_auc}")
        return average_val_auc, average_val_loss
    else:
        train_index, val_index = ksplit(dataset, trainsize, valsize, seed)
        train_subsampler, val_subsampler = SubsetRandomSampler(train_index), SubsetRandomSampler(val_index)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
        # load the encoder checkpoint
        if encoder_type == "VIME":
            trained_model = Encoder.load_from_checkpoint(checkpoint, input_size=16, hidden_size=16) # 35 --> 27
            trained_encoder = trained_model.encoder_stack
            encoder_mlp = EncoderMLP(trained_encoder, 16, 16, 1, lr=lr, weight_decay=weight_decay, en_weight_decay=en_weight_decay)
            trainer = pl.Trainer(limit_train_batches=batch_size, max_epochs=epochs, default_root_dir="VIMEEncoderMLP_dia_logs/")
        elif encoder_type == "mix":
            trained_model = MixEncoder.load_from_checkpoint(checkpoint, input_size=16, hidden_size=16)
            trained_encoder = trained_model.encoder_stack
            encoder_mlp = EncoderMLP(trained_encoder, 16, 16, 1, lr=lr, weight_decay=weight_decay,en_weight_decay=en_weight_decay)
            trainer = pl.Trainer(limit_train_batches=batch_size, max_epochs=epochs, default_root_dir="MixEncoderMLP_dia_logs/")
        encoder_mlp.MLPstack.apply(weights_init)
        trainer.fit(encoder_mlp, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        val_metrics = trainer.test(encoder_mlp, dataloaders=val_dataloader)
        val_auc = val_metrics[0]['test_auc']
        val_loss = val_metrics[0]['test_loss']
        print("------------------------- RESULTS -------------------------")
        print(f"val_loss: {val_loss}, val_auc: {val_auc}")
        return val_auc, val_loss


def combinedencodertrain(dataset, checkpoint, checkpoint2, batch_size=15, epochs=20, lr=0.001, seed=10, valsize=0.25, trainsize=0.75, weight_decay=0.02, en_weight_decay=0.01):
    train_index, val_index = ksplit(dataset, trainsize, valsize, seed)
    train_subsampler, val_subsampler = SubsetRandomSampler(train_index), SubsetRandomSampler(val_index)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
    # load the encoder checkpoint
    trained_model = Encoder.load_from_checkpoint(checkpoint, input_size=16, hidden_size=16)
    trained_model2 = MixEncoder.load_from_checkpoint(checkpoint2, input_size=16, hidden_size=16)
    with torch.no_grad():
        trained_encoder = trained_model.encoder_stack.eval()
        trained_encoder2 = trained_model2.encoder_stack.eval()
    encoder_mlp = CombinedEncoderMLP(trained_encoder, trained_encoder2, lr=lr, weight_decay=weight_decay, en_weight_decay=en_weight_decay)
    trainer = pl.Trainer(limit_train_batches=batch_size, max_epochs=epochs, default_root_dir="CombinedEncoder_dia_logs/")
    encoder_mlp.MLPstack.apply(weights_init)
    trainer.fit(encoder_mlp, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    val_metrics = trainer.test(encoder_mlp, dataloaders=val_dataloader)
    val_auc = val_metrics[0]['test_auc']
    val_loss = val_metrics[0]['test_loss']
    print("------------------------- RESULTS -------------------------")
    print(f"val_loss: {val_loss}, val_auc: {val_auc}")
    return val_auc, val_loss


def mixencodertrain(dataset, kfold=False, batch_size=15, epochs=20, lr=0.001, folds=5, seed=10, valsize=0.25, trainsize=0.75):
    if kfold == True:
        np.random.seed(seed)
        kfold2 = KFold(n_splits=folds, shuffle=True)
        kfold2.get_n_splits(dataset)
        for i, (train_index, val_index) in enumerate(kfold2.split(dataset)):
            train_subsampler = SubsetRandomSampler(train_index)
            val_subsampler = SubsetRandomSampler(val_index)
            print(f'------------- FOLD {i+1} -------------')
            self_train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
            self_val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

            encoder = MixEncoder(27, 27, lr=lr)
            en_trainer = pl.Trainer(limit_train_batches=batch_size, max_epochs=epochs, default_root_dir="MixEncoder_fold_logs/")

            en_trainer.fit(encoder, train_dataloaders=self_train_dataloader, val_dataloaders=self_val_dataloader)
            en_trainer.test(encoder, dataloaders=self_val_dataloader)
    elif kfold == False:
        np.random.seed(seed)
        train_index, val_index = train_test_split(list(range(len(dataset))), train_size=trainsize, test_size=valsize, random_state=seed)
        train_subsampler, val_subsampler = SubsetRandomSampler(train_index), SubsetRandomSampler(val_index)
        self_train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        self_val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
        encoder = MixEncoder(16, 16, lr=lr)
        en_trainer = pl.Trainer(limit_train_batches=batch_size, max_epochs=epochs, accelerator='cpu', default_root_dir="MixEncoder_logs/")

        en_trainer.fit(encoder, train_dataloaders=self_train_dataloader, val_dataloaders=self_val_dataloader)
        val_metrics = en_trainer.test(encoder, dataloaders=self_val_dataloader)
        val_loss = val_metrics[0]['test_loss']
        print("------------------------- RESULTS -------------------------")
        print(f"val_loss: {val_loss}")
        return val_loss


def MLPperf(N0data, epochs=2500, folds=5, lr=0.00005, seed=2129, batch_size=25, trainsize=0.375, weight_decay=0.063125, runs=1, kfold=False):
    average_auc = 0
    average_loss = 0
    for i in range(runs):
        print(f"------------------------ RUN {i+1} ------------------------")
        av_auc, av_loss = MLPtrain(N0data, epochs=epochs, folds=folds, lr=lr, seed=seed, batch_size=batch_size, trainsize=trainsize, weight_decay=weight_decay, kfold=kfold)
        average_auc += av_auc
        average_loss += av_loss
    average_auc /= runs
    average_loss /= runs
    print(f"Average-average loss: {average_loss},  Average-average AUC: {average_auc}, Number of runs: {runs}")
    return average_auc, average_loss


def encoderperf(N0data, checkpoint, epochs=3000, folds=5, lr=0.00005, seed=2129, batch_size=200, trainsize=0.25, weight_decay=0.063125, en_weight_decay=0.1, runs=1, kfold=False, encoder_type="VIME"):
    average_en_auc = 0
    average_en_loss = 0
    runs = 1
    for i in range(runs):
        print(f"------------------------ RUN {i+1} ------------------------")
        av_en_auc, av_en_loss = encoderMLPtrain(N0data, checkpoint, epochs=epochs, folds=folds, lr=lr, seed=seed, batch_size=batch_size, trainsize=trainsize, weight_decay=weight_decay, en_weight_decay=en_weight_decay, kfold=kfold, encoder_type=encoder_type)
        average_en_auc += av_en_auc
        average_en_loss += av_en_loss
    average_en_auc /= runs
    average_en_loss /= runs
    print(f"Average-average loss: {average_en_loss},  Average-average AUC: {average_en_auc}")
    return average_en_auc, average_en_loss



