# -*- coding: utf-8 -*-
# @File   : 3
# @Time   : 2022/12/16 11:50 
# @Author : huangqifan
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from tqdm.notebook import tqdm
import os
import copy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as ptl
from torchvision import transforms
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.utils.data as Data




class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)


class ModelGRU(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=300, output_size=4):
        super(ModelGRU, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_layer_size,
            dropout=0.8,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_layer_size, 4)

    def forward(self, input_seq):
        r_out, h_c = self.lstm(input_seq, None)
        x = self.linear(r_out[:, -1, :])
        return x

def abs_sum(y_pre,y_tru):
    y_pre=np.array(y_pre)
    y_tru=np.array(y_tru)
    loss=sum(abs(y_pre-y_tru))
    return loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.3, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


class MyDataset(Dataset):
    def __init__(self, X, Y, is_train=False):
        # 定义好 image 的路径
        self.X = torch.FloatTensor(X)
        self.Y = torch.LongTensor(Y)
        self.is_train = is_train

    def __getitem__(self, index):
        x = self.X[index]
        if self.is_train:
            # add random noise
            x += torch.rand_like(x) * 0.03

            # shift
            offset = int(np.random.uniform(-10, 10))
            if offset >= 0:
                x = torch.cat((x[offset:], torch.rand(offset) * 0.001))
            else:
                x = torch.cat((torch.rand(-offset) * 0.001, x[:offset]))

        x = x.reshape(*x.shape, 1)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)

def fun(test_feature,X_resampled,y_resampled,batch_size,device,test_loader):
    epochs = 40
    time_step = 205
    input_size = 1
    lr = 0.001     # 0.001跨度大，到几百以后便就一直2 3万

    plot_loss = []
    global output_test
    output_test = np.zeros((test_feature.shape[0], 4))

    kf = KFold(n_splits=5,shuffle=True, random_state=2021) #
    for k, (train, val) in enumerate(kf.split(X_resampled, y_resampled)):
        best_score = float('inf')
        model = ModelGRU()
        model = model.to(device)
        best_model = ModelGRU()
        best_model = best_model.to(device)
        loss_function = LabelSmoothingCrossEntropy()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        # print(train)  # 索引，没毛病
        # print(val)
        X_train = X_resampled[train]
        X_val = X_resampled[val]
        y_train = y_resampled[train]
        y_val = y_resampled[val]

        train_torch_dataset = MyDataset(X_train, y_train, is_train=True)
        val_torch_dataset = MyDataset(X_val, y_val, is_train=False)

        train_loader = Data.DataLoader(
            dataset=train_torch_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
        )
        val_loader = Data.DataLoader(
            dataset=val_torch_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
        )

        for epoch in range(epochs):
            model.train()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                out = model(batch_x)
                loss = loss_function(out, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(loss)

                print('\r batch: {}/{}'.format(i, len(train_loader)), end='')
            # print(loss)
            # print(loss.item())
            score = 0
            model.eval()
            for i, (batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                out = model(batch_x)
                pred = out.argmax(dim=1)
                score += abs_sum(pred.detach().cpu().numpy(), batch_y.detach().cpu().numpy())
            if score < best_score:
                best_score = score
                best_model.load_state_dict(model.state_dict())
            print('epoch: {}, loss: {}, score: {}'.format(epoch, loss.item(), score))
        #         plot_loss.append(loss.item())

        result = None
        best_model.eval()
        for x in test_loader:
            x = x[0]
            x = x.to(device)
            out = best_model(x)
            out = torch.softmax(out, dim=1)
            pred = out.detach().cpu().numpy()
            if result is None:
                result = pred
            else:
                result = np.concatenate((result, pred), axis=0)
            # print(result)
        output_test += result
        # print(output_test)
    # print(output_test)
    return output_test
    # 保存
#     torch.save(best_model.state_dict(), 'GRU-KFold-lm-rand-shift-03-{}.pkl'.format(k))
# plt.plot(plot_loss)


class LabelSmoothingCrossEntropy1(nn.Module):
    def __init__(self, eps=0.3, reduction='mean'):
        super(LabelSmoothingCrossEntropy1, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


class MyDataset1(Dataset):
    def __init__(self, X, Y, is_train=False):
        # 定义好 image 的路径
        self.X = torch.FloatTensor(X)
        self.Y = torch.LongTensor(Y)
        self.is_train = is_train

    def __getitem__(self, index):
        x = self.X[index]
        if self.is_train:
            # add random noise
            x += torch.rand_like(x) * 0.03

            # shift
        #             offset = int(np.random.uniform(-10,10))
        #             if offset >= 0:
        #                 x = torch.cat((x[offset:], torch.rand(offset)*0.001))
        #             else:
        #                 x = torch.cat((torch.rand(-offset)*0.001, x[:offset]))

        x = x.reshape(*x.shape, 1)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)

def fun1(test_feature,X_resampled,y_resampled,batch_size,device,test_loader):
    epochs = 60
    lr = 0.001

    plot_loss = []
    global output_test2
    output_test2 = np.zeros((test_feature.shape[0], 4))

    kf = KFold(n_splits=5, shuffle=True, random_state=2021)
    for k, (train, val) in enumerate(kf.split(X_resampled, y_resampled)):
        best_score = float('inf')
        model = TCN(input_size=1, output_size=4, num_channels=[60] * 6, kernel_size=7, dropout=0.1)
        best_model = TCN(input_size=1, output_size=4, num_channels=[60] * 6, kernel_size=7, dropout=0.1)
        model = model.to(device)
        best_model = best_model.to(device)
        loss_function = LabelSmoothingCrossEntropy1()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        X_train = X_resampled[train]
        X_val = X_resampled[val]
        y_train = y_resampled[train]
        y_val = y_resampled[val]

        train_torch_dataset = MyDataset1(X_train, y_train, is_train=True)
        val_torch_dataset = MyDataset1(X_val, y_val, is_train=False)

        train_loader = Data.DataLoader(
            dataset=train_torch_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
        )
        val_loader = Data.DataLoader(
            dataset=val_torch_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
        )

        for epoch in range(epochs):
            model.train()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.permute(0, 2, 1)
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                out = model(batch_x)
                loss = loss_function(out, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('\r batch: {}/{}'.format(i, len(train_loader)), end='')
            # print(loss)
            score = 0
            model.eval()
            for i, (batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.permute(0, 2, 1)
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                out = model(batch_x)
                pred = out.argmax(dim=1)
                score += abs_sum(pred.detach().cpu().numpy(), batch_y.detach().cpu().numpy())
            if score < best_score:
                best_score = score
                best_model.load_state_dict(model.state_dict())
            print('epoch: {}, loss: {}, score: {}'.format(epoch, loss.item(), score))

        result = None
        best_model.eval()
        for x in test_loader:
            x = x[0]
            x = x.permute(0, 2, 1)
            x = x.to(device)
            out = best_model(x)
            out = torch.softmax(out, dim=1)
            pred = out.detach().cpu().numpy()
            if result is None:
                result = pred
            else:
                result = np.concatenate((result, pred), axis=0)
        output_test2 += result
    #         plot_loss.append(loss.item())
    #     torch.save(best_model.state_dict(), 'TCN-60-8-7-KFold-label-smooth-rand-03-{}.pkl'.format(k))
    # plt.plot(plot_loss)
    return output_test2

def Weighted_method(*arrs):
    n = len(arrs)
    Weighted_result = np.zeros_like(arrs[0])
    for i in range(n):
        Weighted_result += 1.0/n * arrs[i]
    return Weighted_result
if __name__ == '__main__':
    device = torch.device('cuda:0')
    batch_size = 256

    train_data = pd.read_csv('./train.csv')
    test_data = pd.read_csv('./testA.csv')

    train_feature = train_data['heartbeat_signals'].str.split(",", expand=True).astype(float)
    test_feature = test_data['heartbeat_signals'].str.split(",", expand=True).astype(float)

    train_label = train_data['label'].values
    train_feature = train_feature.values
    test_feature = test_feature.values

    ros = RandomOverSampler(random_state=2021)
    X_resampled, y_resampled = ros.fit_resample(train_feature, train_label)

    X_test = torch.FloatTensor(test_feature)
    test_torch_dataset = Data.TensorDataset(X_test.reshape([X_test.shape[0], X_test.shape[1], 1]))
    test_loader = Data.DataLoader(
        dataset=test_torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )
    fun(test_feature, X_resampled, y_resampled, batch_size, device, test_loader)
    fun1(test_feature, X_resampled, y_resampled, batch_size, device, test_loader)

    # output_test = fun(test_feature,X_resampled,y_resampled,batch_size,device,test_loader)
    # output_test2=fun1(test_feature,X_resampled,y_resampled,batch_size,device,test_loader)
    final_result = Weighted_method(output_test,output_test2) #

    tmp = final_result.argmax(axis=1)
    one_hot = OneHotEncoder(sparse=False).fit(tmp.reshape([-1,1]))
    pred_tmp = one_hot.transform(tmp.reshape([-1,1]))

    output = pd.DataFrame({'id':test_data['id'], 'label_0':pred_tmp[:,0], 'label_1':pred_tmp[:,1], 'label_2':pred_tmp[:,2], 'label_3':pred_tmp[:,3]})
    output.to_csv('submit_final.csv',index=False)