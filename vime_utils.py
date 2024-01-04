import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from torch.utils.data import Dataset
from sklearn.model_selection import KFold, train_test_split
from random import choice
from torch import nn


# create custom N0 dataloader
class preloader(Dataset):
    def __init__(self, dataset, normalize=True):
        self.dataset = dataset
        self.df=pd.read_csv(dataset, sep='\t')
        self.df_labels=self.df[["N0"]]
        self.df=self.df.drop(columns=["ID", "N0"])
        # normalize all values (z-score)
        if normalize == True:
            for column in self.df.columns:
                self.df[column] = (self.df[column] - self.df[column].mean()) / self.df[column].std()
        self.dataset=torch.tensor(self.df.to_numpy()).float()
        self.labels=torch.tensor(self.df_labels.to_numpy().reshape(-1)).long()
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]

# create custom diabetes dataloader
class dia_preloader(Dataset):
    def __init__(self, dataset, normalize=True):
        self.dataset = dataset
        self.df=pd.read_csv(dataset, sep=';')
        for i in range(len(self.df["gender"])):
            self.df.at[i, "gender"] = 0.0 if self.df.at[i, "gender"] == "Male" else 1.0
        self.df = self.df.astype(float)
        self.df_labels=self.df[["class"]]
        self.df=self.df.drop(columns=["class"])
        # normalize all values (z-score)
        if normalize == True:
            for column in self.df.columns:
                self.df[column] = (self.df[column] - self.df[column].mean()) / self.df[column].std()
        self.dataset=torch.tensor(self.df.to_numpy()).float()
        self.labels=torch.tensor(self.df_labels.to_numpy().reshape(-1)).long()
        
    def __len__(self):
        return len(self.dataset) 
    
    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]


# weight initialization
def weights_init(m):
    torch.manual_seed(100)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)


def mask_generator(p_m, x, N0=True):
    mask = np.random.binomial(1., p_m, x.shape)
    # treat several columns as one feature (N0 dataset specific)
    if N0 == True:
        for row in mask:
            for C in row[7:12]:
                if C == 1:
                    row[7:12] = [1, 1, 1, 1, 1]

            for T in row[15:18]:
                if T == 1:
                    row[15:18] = [1, 1, 1]
    return mask


def pretext_generator(m, x):
    x = x
    no, dim = x.shape  
    # randomly (and column-wise) shuffle data
    x_bar = np.zeros([no, dim])
    for i in range(dim):
        idx = np.random.permutation(no)
        x_bar[:, i] = x[idx, i]

    # corrupt samples
    x_tilde = x * (1-m) + x_bar * m  
    # define new mask matrix
    m_new = 1 * (x != x_tilde)

    return m_new.float(), x_tilde.float()


def mixer(x):
    lamb = round(np.random.beta(5, 1), 2)
    if lamb > 0.9:
        lamb = 1.0
    if lamb <= 0.7:
        lamb = 0.7
    mixed_x = torch.clone(x)
    for i in range(x.shape[0]):
        index = choice(list(set([k for k in range(0, x.shape[0])]) - set([i]))) # avoid mixing with itself
        random_x = x[index]
        mixed_x[i] = (x[i]*lamb) + (random_x*(1-lamb))
    return mixed_x, lamb


def kfolder(dataset, folds, seed):
    np.random.seed(seed)
    kfold = KFold(n_splits=folds, shuffle=True)
    kfold.get_n_splits(dataset)
    return kfold


def ksplit(dataset, trainsize, valsize, seed):
    np.random.seed(seed)
    train_index, val_index = train_test_split(list(range(len(dataset))), train_size=trainsize, test_size=valsize, random_state=seed)
    return train_index, val_index


def aucplot(aucs, sizes, title="Plot title", save=False):
    if save == True:
        with open('dia_results.csv', 'a', newline='\n') as file:
            writer = csv.writer(file)
            writer.writerow([aucs])
            file.close()
    models = ["MLP", "VIME", "Mix", "Combined"]
    markers = ['o', 'x', 'v', 'o']
    sizes = [x*520 for x in sizes] # 800 for N0, 520 for dia
    for m in range(len(aucs)):
        averages = [np.mean(x) for x in aucs[m]]
        e = [np.std(x)/np.sqrt(len(x)) for x in aucs[m]]
        plt.errorbar(sizes, averages, e, linestyle='dashed', marker=markers[m], label=models[m])
        plt.legend(title="Model")
        plt.title(f"{title}")
        plt.xlabel("Training size")
        plt.ylabel("AUC")
        for k in range(len(averages)):
            print(f"----- {models[m]} AVERAGES -----")
            print(f"Average ({sizes[k]}): {averages[k]} +- {e[k]} \n")
    plt.show()