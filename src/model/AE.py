import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import  Dataset, DataLoader
from torchinfo import summary
import random
import numpy as np
from torch.optim import *
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from util.utils import EarlyStopping

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_seed(72)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

class MLP(nn.Module):
    def __init__(self, mlp_units, dropout, activation=nn.ReLU, output_activation=nn.Sigmoid):
        super().__init__()
        self.mlp_layers = []
        for i in range(len(mlp_units)-1):
            act = activation if i < len(mlp_units)-2 else output_activation
            if i!=len(mlp_units)-2:
                self.mlp_layers += [nn.Linear(mlp_units[i], mlp_units[i+1]), nn.Dropout(dropout), act()]
            else:
                if act == None:
                    self.mlp_layers += [nn.Linear(mlp_units[i], mlp_units[i + 1])]
                else:
                    self.mlp_layers += [nn.Linear(mlp_units[i], mlp_units[i+1]), act()]
        self.mlp_layers = nn.Sequential(*self.mlp_layers)
    def forward(self, inputs):
        return self.mlp_layers(inputs)


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        # Encoder
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        # Decoder
        self.tran_cnn_layer1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0),
            nn.ReLU())

        self.tran_cnn_layer2 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        output = self.cnn_layer1(x)
        output = self.cnn_layer2(output)
        output = self.tran_cnn_layer1(output)
        output = self.tran_cnn_layer2(output)

        return output


class AutoEncoder():
    def __init__(
            self,
            units,
            dropout,
            lr,
            input_type: str = 'vector',
            epochs = 100,
            device = 'cpu',
            use_early_stopping = False
    ):
        super().__init__()

        # convolution layer 수정 중

        #         self.encoder = CNN(units) if input_type=='img' else MLP(units, dropout, output_activation=None)
        #         self.decoder = CNN(units) if input_type=='img' else MLP(list(reversed(units))[1:], dropout)
        self.AE = ConvAutoEncoder() if input_type == 'img' else MLP(units, dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.AE = self.AE.to(self.device)
        self.epochs = epochs
        self.history = dict()
        self.use_early_stopping = use_early_stopping
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.AE.parameters(), lr=lr)
        self.best = {"loss": sys.float_info.max}

    def train(self, trainloader, validloader):

        summary(self.AE, next(iter(trainloader))[0].shape)
        if self.use_early_stopping == True:
            early_stopping = EarlyStopping(patience=10, verbose=1)

        self.history = dict()
        self.best = {"loss": sys.float_info.max}

        for epoch in range(1, self.epochs + 1):
            epoch_loss = self._train(trainloader)
            val_loss = self.validation(validloader, mean_loss=True)

            self.history.setdefault('loss', []).append(epoch_loss)
            self.history.setdefault('val_loss', []).append(val_loss)

            print(f"[Train] Epoch : {epoch:^3}" \
                  f"  Train Loss: {epoch_loss:.4}" \
                  f"  Validation Loss: {val_loss:.4}")
            if epoch_loss < self.best["loss"]:
                self.best["state"] = self.AE.state_dict()
                self.best["loss"] = epoch_loss
                self.best["epoch"] = epoch + 1

            if self.use_early_stopping == True:
                if early_stopping.validate(val_loss):
                    break

        self.AE.load_state_dict(self.best["state"])

    def _train(self, train_data, use_fp16=True, max_norm=None):
        epoch_loss = 0

        self.AE.train()

        for idx, [x, y] in enumerate(train_data):

            self.optimizer.zero_grad(set_to_none=True)
            scaler = torch.cuda.amp.GradScaler()

            x = x.to(self.device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                output = self.AE.forward(x)
                train_loss = self.loss_fn(output, x)
            if use_fp16:
                scaler.scale(train_loss).backward()
                if max_norm is not None:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.AE.parameters(), max_norm)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                train_loss.backward()
                self.optimizer.step()
            epoch_loss += train_loss.item()

        return epoch_loss / len(train_data)

    def validation(self, x, mean_loss=False):
        self.AE.eval()
        x = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            output = self.AE.forward(x)
            val_loss = torch.nn.MSELoss()(output, x)
            ad_scores = torch.sum((output - x) ** 2, 1).detach().cpu().numpy()
            ad_scores = ad_scores * -1

        return val_loss if mean_loss == True else ad_scores

    def test(self, x, mean_loss=False):
        self.AE.eval()
        x = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            output = self.AE.forward(x)
            val_loss = torch.nn.MSELoss()(output, x)
            ad_scores = torch.sum((output - x) ** 2, 1).detach().cpu().numpy()
            ad_scores = ad_scores * -1

        return val_loss if mean_loss == True else ad_scores

    def save(self, filename):

        state = {
            'AE': self.AE.state_dict(),
            'best_loss': self.best["loss"],
            'best_epoch': self.best["epoch"],
        }

        torch.save(state, filename)


if __name__ == '__main__':
    pass