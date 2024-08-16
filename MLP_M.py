import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from einops.layers.torch import Rearrange
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from functools import partial
import numpy as np
import imageio.v2 as imageio
from torchmetrics.classification import JaccardIndex
from torch.utils.tensorboard import SummaryWriter


class ExpanDim(nn.Module):
    '''扩张复制最后一个维度，如(batch, channels, weight) -> (batch, channels, weight, height(weight dims))'''
    def __init__(self, expand_size):
        super(ExpanDim, self).__init__()
        self.expand_size = expand_size
    def forward(self,x):
        x = x.unsqueeze(-1)
        x = x.expand(-1,-1,-1, self.expand_size)
        return x
    
class ConcatenateLayer(nn.Module):
    '''根据指定维度dim叠加两个张量，等效tensorflow concatenate'''
    def __init__(self, dim=1):
        super(ConcatenateLayer, self).__init__()
        self.dim = dim
        # self.cat = torch.cat()
    def forward(self,x1,x2):
        return torch.cat((x1,x2),self.dim)
    
class UpsampConv(nn.Module):
    def __init__(self, ini_channels, filters_num: list = None,activation=nn.GELU):
        super(UpsampConv, self).__init__()
        self.filters_num = filters_num
        self.ini_channels = ini_channels
        self.act = activation
        self.ndim = len(self.filters_num)
        self.upsamples = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bicubic') for _ in range(self.ndim)])
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=ini_channels if i==0 else ini_channels+filters_num[i-1],out_channels=filters_num[i],kernel_size=2,stride=2) for i in range(self.ndim)])
        self.conca = ConcatenateLayer()
    def forward(self,x):
        for ups, con in zip(self.upsamples, self.convs):
            x = self.conca(x, self.act()(con(ups(x))))
        return x

class RandomConcatLayer(nn.Module):
    '''对输入张量根据指定维度dim生成随机张量并拼接，这个随机在每个训练epoch应该是不重复的'''
    def __init__(self, dim=1):
        super(RandomConcatLayer, self).__init__()
        self.dim = dim
        # self.device = device
        # self.list = list()

        # self.random_matrix = torch.rand()
        # self.cat = torch.cat()

    def forward(self, x):
        # 获取输入的形状
        shape = list(x.shape)
        
        # 生成随机矩阵
        shape[self.dim] = 1
        random_matrix = torch.rand(*shape, device=x.device)
        
        # 将随机矩阵与输入图像在通道维度上拼接
        x = torch.cat((x, random_matrix), dim=self.dim)
        return x

class PreNormResidual(nn.Module):
    '''对所有batch按照dim维度进行归一化，之后fn网络处理，最后使用残差连接'''
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)  # 对所有batch按照dim维度进行归一化

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=3, dropout=0.4, dense=nn.Linear):
    '''定义多层感知机。MLP层设计，两个FC全连接层+一个激活函数GELU，输入输出维度不变，只是中间全连接层神经元数目根据expansion_factor先扩张后缩减'''
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),  # 激活函数
        nn.Dropout(dropout),  # 专门用于训练，推理阶段需要关掉model.eval()
        # 在训练过程的前向传播中，让每个神经元以一定概率dropout处于不激活的状态。以达到减少过拟合的效果
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )
    


class MLPm(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth,filters_num, upchannels, expansion_factor=3, dropout=0.4,):
        super().__init__()
        assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_size // patch_size)** 2; ini_chan = 3*patch_size**2
        
        self.random_concat_layer = RandomConcatLayer(dim=1)
        self.rearrange = Rearrange("b c (h p1) (w p2)  -> b (p2 p1 c) (h w) ", p1=patch_size, p2=patch_size)
        self.linear = nn.Linear(num_patches, dim)
        self.chan_first = partial(nn.Conv1d, kernel_size=1)
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                PreNormResidual(dim, FeedForward(ini_chan, expansion_factor, dropout, self.chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, nn.Linear))
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.linear_out = nn.Linear(dim, upchannels)
        self.expand_dim = ExpanDim(upchannels)
        self.upsamp_conv = UpsampConv(ini_channels=ini_chan, filters_num=filters_num)
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')
        self.conv_out = nn.Conv2d(in_channels=sum(filters_num) + ini_chan, out_channels=2, kernel_size=1, bias=False, padding='valid')
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    def forward(self, x):
        x = self.random_concat_layer(x)
        x = self.rearrange(x)
        x = self.linear(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.linear_out(x)
        x = self.expand_dim(x)
        x = self.upsamp_conv(x)
        x = self.upsample(x)
        x = self.conv_out(x)
        return x
        # return nn.Sequential(
        # RandomConcatLayer(),
        # Rearrange("b c (h p1) (w p2)  -> b (p2 p1 c) (h w) ", p1=patch_size, p2=patch_size),
        # nn.Linear(num_patches, dim),
        # *[nn.Sequential(PreNormResidual(dim, FeedForward(ini_chan, expansion_factor, dropout, chan_first)),
        # PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))) for _ in range(depth)],
        # nn.LayerNorm(dim),
        # nn.Linear(dim,upchannels),
        # ExpanDim(upchannels),
        # UpsampConv(ini_channels=ini_chan,filters_num=filters_num),
        # nn.Upsample(scale_factor=2,mode='bicubic'),
        # nn.Conv2d(in_channels=sum(filters_num,start=ini_chan),out_channels=2,kernel_size=1,bias=False, padding='valid'))
        # 损失函数为交叉熵时不需要nn.Softmax(dim=1))，因为已经包含了softmax处理


def getimg(dir,num,j):
    '''获取dir下的所有图片并使用imageio读取，考虑使用torch io API替代加速'''
    input_dir = dir
    img_list = []
    for i in range(1+j,num+1+j):
        filename = f'{i}.tif'
        img_path = os.path.join(input_dir,filename)
        img = imageio.imread(img_path)
        img = img.astype(np.float32)
        img = img / 255.0
        img_list.append(img)
    return img_list

def train(train_tensor, vali_tensor, model, loss_f, optimizer, epoches, save_path, logdir, device):
    size = len(train_tensor.dataset)
    pre_total_loss = 1e4 
    pre_epoch = 0
    model.train()
    writer_train = SummaryWriter(log_dir=os.path.join(logdir,'train'))
    writer_vali = SummaryWriter(log_dir=os.path.join(logdir,'vali'))
    train_iou = JaccardIndex(num_classes=2, task="binary").to(device)
    val_iou = JaccardIndex(num_classes=2, task="binary").to(device)

    for epoch in range(epoches):
        for batch_index, (data, target) in enumerate(train_tensor):
            data, target = data.to(device), target.to(device)
            train_iou.reset()
            pred = model(data)
            loss = loss_f(pred, target)
            # loss.requires_grad_(True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predictions = torch.argmax(pred, dim=1)
            train_iou.update(predictions, target)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.12f}'.format(
                epoch, batch_index * len(data), len(train_tensor.dataset),
                       100. * batch_index / len(train_tensor), loss.item()))
            if batch_index % 100 == 0:
                loss, current = loss.item(), (batch_index + 1) * len(data)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        train_iou_value = train_iou.compute()
        print(f"Train Epoch: {epoch} - Jaccard Index (IoU): {train_iou_value:.4f}")
        writer_train.add_scalar('Training IoU', train_iou_value, epoch)
        # writer.add_scalar('CrossEntropyLoss_lastbatch', loss.item(), epoch)

        # fig = train_iou.plot()
        # writer.add_figure('Training IoU Plot', fig, epoch)
        # plt.close(fig)  # Close the figure to free memory

        if epoch %10 == 0:
            model.eval()
            val_iou.reset()
            total = 0
            all_preds = []
            with torch.no_grad():
                for _, (val_data, val_target) in enumerate(vali_tensor):
                    val_data, val_target = val_data.to(device), val_target.to(device)
                    pred = model(val_data)
                    
                    v_loss = loss_f(pred, val_target)
                    predictions = torch.argmax(pred, dim=1) 
                    val_iou.update(predictions, val_target)
                    print('val_loss: ', v_loss.item())
                    total += v_loss.item()*val_data.size(0)
                
                    all_preds.append(predictions.cpu())

                val_iou_value = val_iou.compute()
                print(f"Validation - Epoch: {epoch} - Loss: {v_loss:.4f}, IoU: {val_iou_value:.4f}")
                writer_vali.add_scalar('validation IoU', val_iou_value, epoch)

                # pr curves:
                all_preds = torch.cat(all_preds)
                writer_vali.add_pr_curve(f'PR_curve_epoch_{epoch}', tv_label, all_preds, global_step=epoch)
                if total <= pre_total_loss:
                    print(f"now loss :{v_loss:>7f} better than previous best epoch{pre_epoch:>3d}:{pre_total_loss:>7f} ")
                    model_name = f'MLPM_epoch{epoch:03d}-valoss{v_loss:.3f}.pth'
                    torch.save(model, os.path.join(save_path, model_name))
                    pre_total_loss = total
                    pre_epoch = epoch
            
    torch.save(model, os.path.join(save_path, 'MLPM_final_epoch.pth'))
    return

if __name__ == '__main__':
        
    NUM_train = 3900
    NUM_tv = 200
    Batch_Size = 16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    output_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/exped_output/'
    label_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/label/label/'
    img_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/data1/data/'
    img1_list = getimg(img_path,num=NUM_train, j=0)
    img_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/data2/data/'
    img2_list = getimg(img_path,num=NUM_train, j=900)
    label_list = getimg(label_path,num=NUM_train, j=0)
    img_list = np.stack((img1_list,img2_list),axis = 1)
    img_path2 = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/test/predicted_input_data/'
    label_path2 = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/test/predicted_label/'
    # output_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/test/outputv2/'
    img1_list = getimg(img_path2,num=100, j =0)
    img2_list = getimg(img_path2,num=100, j =100)
    test_img_list = np.stack((img1_list,img2_list),axis = 1)
    test_label_list = getimg(label_path2, num=100, j=0)
    img_list = np.vstack((img_list, test_img_list))
    label_list = np.vstack((label_list, test_label_list))

    Input = torch.from_numpy(img_list)
    Output = torch.from_numpy(label_list.astype(np.int64))   
    train_tensor = TensorDataset(Input, Output)
    train_tensor = DataLoader(train_tensor, batch_size=Batch_Size, shuffle=True)     

    test_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/test/test2/data/'
    test_label_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/test/test2/label/'
    test_output_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/test/outputv2/'
    img1_list = getimg(test_path,num=NUM_tv, j =0)
    img2_list = getimg(test_path,num=NUM_tv, j =200)
    img_list = np.stack((img1_list,img2_list),axis = 1)
    label_list = getimg(test_label_path,num=NUM_tv,j=0)

    tv_image = torch.from_numpy(img_list)
    tv_label = torch.from_numpy(np.array(label_list).astype(np.int64)) 
    vali_tensor = TensorDataset(tv_image, tv_label)
    vali_tensor = DataLoader(vali_tensor, batch_size=Batch_Size)
    '''_'''


    model = MLPm(image_size=128,
          patch_size=16,
          dim=1024,
          depth=10,
          filters_num=[16,8],
          upchannels=64).to(device)
            
    # we_loss = [0.1, 0.9]
    loss_fn = nn.CrossEntropyLoss() #(weight=we_loss)
    optimizer = torch.optim.Adam(model.parameters())
    total_params = 0
    save_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/save_model/torch'
    logdir = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/output/logs_torch'
    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        print(f'Layer: {param.shape}, Parameters: {num_params}')

    train(train_tensor=train_tensor, vali_tensor=vali_tensor,
          model=model, loss_f=loss_fn, optimizer=optimizer, 
          epoches=500,
          save_path=save_path, logdir=logdir,
          device=device)
    
        
        



    