import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import  Dataset, DataLoader
from torchinfo import summary
import random
import numpy

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


# set_seed(72)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


# convolution layer 수정 중
class CNN(nn.Module):
    def __init__(self, conv_channel_lst):
        # call the pytorch constructor of the parent class
        super().__init__()

        self.nb_conv = len(conv_channel_lst)

        # Conv layer
        self.conv_lst = nn.Sequential()
        for i in range(len(conv_channel_lst)):
            if i == 0:
                self.conv_lst.add_module(f"conv{i}", nn.Conv2d(1, conv_channel_lst[i], 3))
            else:
                self.conv_lst.add_module(f"conv{i}", nn.Conv2d(conv_channel_lst[i - 1], conv_channel_lst[i], 3))
            self.conv_lst.add_module(f"maxpool{i}", nn.MaxPool2d(2, 2))
            self.conv_lst.add_module(f"relu{i}", nn.ReLU())

    def forward(self, x):
        # conv layers
        x = self.conv_lst(x)
        return x

    def calc_fcn_layer(self):
        input_dim = 32
        k = 3  # kernel size
        s = 1  # strides
        p = 0  # padding size

        for i in range(self.nb_conv):
            input_dim = (input_dim - k + 2 * p) / s + 1
            input_dim = int(input_dim / 2)  # max pooling with kernel size 2 and strides 2

        return input_dim

class AutoEncoder(nn.Module):
    def __init__(self, units, dropout, input_type):
        super().__init__()

        # convolution layer 수정 중

        self.encoder = CNN(units) if input_type=='img' else MLP(units, dropout, output_activation=None)
        self.decoder = CNN(units) if input_type=='img' else MLP(list(reversed(units)), dropout)


    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


def train(AE, train_data, optimizer, loss_fn, use_fp16=True, max_norm=None):
    epoch_loss = 0

    AE.train()

    for idx, [x, y] in enumerate(train_data):

        optimizer.zero_grad(set_to_none=True)
        scaler = torch.cuda.amp.GradScaler()

        x = x.to(device)

        with torch.cuda.amp.autocast(enabled=use_fp16):
            output = AE.forward(x)
            train_loss = loss_fn(output, x)
        if use_fp16:
            scaler.scale(train_loss).backward()
            if max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(AE.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss.backward()
            optimizer.step()
        epoch_loss += train_loss.item()

    return epoch_loss / len(train_data)

def validation(AE, val_data):
    AE.eval()
    with torch.no_grad():
        output = AE.forward(val_data[0])
        val_loss = torch.nn.MSELoss()(output, val_data[0])
    return val_loss

def test(AE, test_data, loss_fn):
    AE.eval()
    with torch.no_grad():
        output = AE.forward(test_data[0])
        test_loss = torch.sum((output - test_data[0]) ** 2, 0)
    return test_loss


if __name__ == '__main__':
    pass