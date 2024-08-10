import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import h5py
import time
import numpy as np
import torch.utils.data as da
import scipy.io as scio
import matplotlib.pyplot as plt
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)  # 对所有batch按照dim维度进行归一化

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=3, dropout=0.4, dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),  # 激活函数
        nn.Dropout(dropout),  # 专门用于训练，推理阶段需要关掉model.eval()
        # 在训练过程的前向传播中，让每个神经元以一定概率dropout处于不激活的状态。以达到减少过拟合的效果
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )
    # 定义多层感知机。MLP层设计，两个FC全连接层+一个激活函数GELU，输入输出维度不变，只是中间全连接层神经元数目有改变


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)  # 自定义reshape层，以便在Sequential中使用
    # 实现在Sequantial中使用Reshape层


def MLPMixer(*, image_size, patch_size, dim, depth, num_classes, expansion_factor=3, dropout=0.4):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size)  # ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear  # partial：把一个函数的某些参数给固定住，返回一个新的函数

    return nn.Sequential(
        Rearrange('b c (h p1) -> b h (p1 c)', p1=patch_size),  # 对张量进行重排
        nn.Linear((patch_size) * 1, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        nn.Conv1d() # 50卷积成2
        # Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes),
        Reshape(1, 40, 40)
    )


def TrainWork(batch_size, test_path, save_path, device, train_data, val_data, loss_list, val_list):
    NN = 40
    model = MLPMixer(
        image_size=1000,
        patch_size=20,
        dim=1024,
        depth=7,
        num_classes=3200
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                 amsgrad=False)

    target1 = h5py.File(os.path.join(test_path, "Mesh2977.mat"))['Mesh2977'][:][0:NN, 0:NN, 0:2, 0:2900]
    # target2 = h5py.File(os.path.join(test_path, "DataMesh_ZY2.mat"))['Out_Train'][:][3014:3167, 0:NN, 0:NN]
    target = np.vstack(target1)
    target = np.expand_dims(target, axis=1)
    target = torch.from_numpy(target)

    val_target = h5py.File(os.path.join(test_path, "Mesh2977.mat"))['Mesh2977'][:][0:NN, 0:NN, 1:2, 2900:2977]
    val_target = np.expand_dims(val_target, axis=1)
    val_target = torch.from_numpy(val_target)

    data_tensor = da.TensorDataset(train_data, target)
    data_tensor = da.DataLoader(data_tensor, batch_size=batch_size, shuffle=True)

    val_tensor = da.TensorDataset(val_data, val_target)
    val_tensor = da.DataLoader(val_tensor, batch_size=batch_size, shuffle=True)

    lossMSE = nn.MSELoss(reduce=True, size_average=True)

    class my_loss(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            NormLoss = 0
            x = torch.reshape(x, (x.shape[0], NN * NN))
            y = torch.reshape(y, (y.shape[0], NN * NN))
            zero = torch.zeros(x.shape[0], NN * NN).to(device)
            one = torch.ones(x.shape[0], NN * NN).to(device)
            torch.where(x >= 0.5, x, one)
            torch.where(x < 0.5, x, zero)

            for i in range(x.shape[0]):
                NormLoss += torch.norm(x[i] - y[i]) / torch.norm(y[i])
            return torch.div(NormLoss, x.shape[0])

    lossNorm = my_loss()

    for epoch in range(350):
        for batch_idx, (data, target) in enumerate(data_tensor):
            data, target = data.to(device), target.to(device)
            result = model(data.float())
            # loss = lossMSE(result, target.float())
            loss = lossNorm(result, target.float())
            loss_list.append(loss)
            loss.requires_grad_(True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.12f}'.format(
                epoch, batch_idx * len(data), len(data_tensor.dataset),
                       100. * batch_idx / len(data_tensor), loss.item()))

        if epoch % 9 == 0:
            model.eval()
            total = 0
            with torch.no_grad():
                for val_batch_idx, (val_data, val_target) in enumerate(val_tensor):
                    val_data, val_target = val_data.to(device), val_target.to(device)
                    val_result = model(val_data.float())
                    # v_loss = lossBCE(val_result, val_target.float())
                    v_loss = lossNorm(val_result, val_target.float())
                    val_list.append(v_loss)
                    print("val_loss:", v_loss.item())

    torch.save(model, os.path.join(save_path, "mlpMixer_Net.pth"))
    return (v_loss.item())


def TestWork(batch_size_Test, test_path, save_path, device, train_data, val_data):
    NN = 40
    model = torch.load(os.path.join(save_path, "mlpMixer_Net.pth")).to(device)

    val_target = h5py.File(os.path.join(test_path, "Mesh2977.mat"))['Mesh2977'][:][0:NN, 0:NN, 1:2, 2900:2977]
    val_target = np.expand_dims(val_target, axis=1)
    val_target = torch.from_numpy(val_target)

    val_tensor = da.TensorDataset(val_data, val_target)
    val_tensor = da.DataLoader(val_tensor, batch_size=batch_size_Test, shuffle=False)

    lossMSE = nn.MSELoss(reduce=True, size_average=True)
    lossBCE = nn.BCELoss()

    class my_loss(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            NormLoss = 0
            x = torch.reshape(x, (x.shape[0], NN * NN))
            y = torch.reshape(y, (y.shape[0], NN * NN))
            for i in range(x.shape[0]):
                NormLoss += torch.norm(x[i] - y[i]) / torch.norm(y[i])
            return torch.div(NormLoss, x.shape[0])

    lossNorm = my_loss()

    for batch_idx, (data, label) in enumerate(val_tensor):
        data, label = data.to(device), label.to(device)
        result = model(data.float())

        v_loss = lossNorm(result, label.float())
        print("val_loss:", v_loss.item())

    S11 = result.cpu().detach().numpy().reshape((batch_size_Test, NN, NN))

    return S11


batch_size = 32
batch_size_Test = 50
test_path = "D:\XZQ\mixertest"
save_path = "D:\XZQ\mixertest"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

loss_list = []
val_list = []

train_data = np.vstack(h5py.File('S11sqr2977.mat')['S11sqr2977'][:][0:2900, 1:1000])
# 正好被batch_size整除h5py.File('newS11_train.mat')['newS11_train'][:][3014:3167, :])
train_data = train_data * (1 / 10) + 1
train_data = np.expand_dims(train_data, axis=1)
train_data = torch.from_numpy(train_data)

val_data = h5py.File('S11sqr2977.mat')['S11sqr2977'][:][2901:2977, 1:1000]
val_data = val_data * (1 / 10) + 1
val_data = np.expand_dims(val_data, axis=1)
val_data = torch.from_numpy(val_data)

Problem = []

time1 = time.time()
misfit = TrainWork(batch_size, test_path, save_path, device, train_data, val_data, loss_list, val_list)
time2 = time.time()
print("time:", time2 - time1)

S = TestWork(batch_size_Test, test_path, save_path, device, train_data, val_data)
scio.savemat('D:\XZQ\mixertest\Mesh_Out.mat', {'MeshOut': S})
