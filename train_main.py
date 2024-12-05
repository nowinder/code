import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from einops.layers.torch import Rearrange
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
# from functools import partial
import numpy as np
import imageio.v2 as imageio
from torchmetrics.classification import JaccardIndex
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import torch.optim.lr_scheduler as lr_scheduler
from s2_mlp_v2 import S2MLPv2
import h5py
from lovasz_loss import lovasz_softmax
# from MLP_M import MLPm


# def getimg(dir,num,j):
#     '''获取dir下的所有图片并使用imageio读取，考虑使用torch io API替代加速，24/11由于使用mat获取数据废弃'''
#     input_dir = dir
#     img_list = []
#     for i in range(1+j,num+1+j):
#         filename = f'{i}.tif'
#         img_path = os.path.join(input_dir,filename)
#         img = imageio.imread(img_path)
#         img = img.astype(np.float32)
#         img = img / 255.0
#         img_list.append(img)
#     return img_list

def cosine_warmup_lambda(current_step, num_warmup_steps):
    if current_step < num_warmup_steps:
        return 1 * (1 - np.cos(np.pi/2 * (current_step / num_warmup_steps)))
    return 1.0


def train(train_tensor, vali_tensor, model, loss_f,wd, epoches, save_path, logdir, device, model_prefix, path, nowwhatturn='', initial_lr=1e-3, contin= False):
    # size = len(train_tensor.dataset)
    
    pre_epoch = 0
    estop_count = 0
    epoch_loss=0.0
    num_wmp_steps = 100#356
    warmrestart = False
    # decay_factor = 0.9  # 每次重启时减少的因子
    # initial_lr = 0.001
    # optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr,eps=1e-6)
    # optimizer = torch.optim.SGD(model.parameters(),)
    # 这里的正则化考虑不对BN，LN加
    params_with_decay = []
    params_without_decay = []

    # 遍历模型的所有模块和参数
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_param_name = f"{module_name}.{param_name}" if module_name else param_name
            # 如果是 bias 或 LayerNorm、BatchNorm 层的参数
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)) or 'bias' in param_name:
                params_without_decay.append({'params': param, 'name': full_param_name})
            else:
                params_with_decay.append({'params': param, 'name': full_param_name})

    # optimizer = torch.optim.AdamW(model.parameters(),lr=initial_lr, weight_decay=0.01)
    optimizer = torch.optim.AdamW(
    [
        {'params': [p['params'] for p in params_with_decay], 'weight_decay': wd},
        {'params': [p['params'] for p in params_without_decay], 'weight_decay': 0.0},
    ],
    lr=initial_lr,eps=1e-6
)
    
    writer_train = SummaryWriter(log_dir=os.path.join(logdir,'train'),filename_suffix=str(nowwhatturn))
    writer_vali = SummaryWriter(log_dir=os.path.join(logdir,'vali'),filename_suffix=str(nowwhatturn))
    writer_wegra = SummaryWriter(log_dir=os.path.join(logdir,'weightgrad_'+model_prefix),filename_suffix=str(nowwhatturn))
    # writer_we = SummaryWriter(log_dir=os.path.join(logdir,'weight'),filename_suffix=str(nowwhatturn))
    train_iou = JaccardIndex(num_classes=2, task="binary",ignore_index=-1).to(device)
    val_iou = JaccardIndex(num_classes=2, task="binary",ignore_index=-1).to(device)
    # bat_iou = JaccardIndex(num_classes=2, task="binary").to(device)
    scheduler_rlr = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=False) #5个epoch验证一次loss没少就减少还是不太行，可以试试factor0.1以下
    scheduler_wmu = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cosine_warmup_lambda(step, num_warmup_steps=num_wmp_steps))
    # scheduler_cosrest = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-4)
    # 设置最大学习率是增大10倍初值的余弦退火(10倍爆了，改3倍)
    # scheduler_cosan = lr_scheduler.CosineAnnealingLR(optimizer, T_max=30,eta_min=1e-4,last_epoch=-1)
    # scheduler_cosan.base_lrs = [initial_lr*3 for _ in scheduler_cosan.base_lrs]
    if contin:
        # path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/save_model/torch/s2filtn3_epoch380-valoss0.018.pth'
        checkpoint = torch.load(path,map_location=device)
        model.to(device)
        # 恢复模型和优化器状态
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_lr = optimizer.param_groups[0]['lr']
        # scheduler_cosrest.base_lrs = [initial_lr for _ in scheduler_cosrest.base_lrs]

        # 恢复训练的其他信息，如当前的epoch和损失
        start_epoch = checkpoint['epoch']
        pre_epoch = start_epoch
        loss = checkpoint['loss']
        pre_mvloss = 0.047
    else: 
        model.to(device)
        start_epoch = -1
        pre_mvloss = 0.1
    model.train()
    
    onetime=False
    for epoch in range(start_epoch+1, epoches+1):
        # last_loss = epoch_loss
        # epoch_loss=0.0
        
        train_iou.reset()
        for batch_index, (data, target) in enumerate(train_tensor):
            data, target = data.to(device), target.to(device)
            
            pred = model(data)
            loss = loss_f(pred, target)
            # loss.requires_grad_(True)
            epoch_loss+=loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            if epoch * len(train_tensor) + batch_index < num_wmp_steps:
                scheduler_wmu.step()
            # else: warmrestart = True
            predictions = torch.argmax(pred, dim=1)

            train_iou.update(predictions, target)
            iou_value = train_iou(predictions, target).item()
            # iou_value = bat_iou(predictions, target).item()
            # 获取学习率
            # lr_1 = scheduler_rlr.get_last_lr()[0]
            # lr_2 = scheduler_wmu.get_last_lr()[0]
            lr_3 = optimizer.param_groups[0]['lr']

            print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.12f} \tIoU: {:.5f} \tlearning rate: {:.6f}'.format(
                epoch, (batch_index+1) * len(data), len(train_tensor.dataset),
                       100. * (batch_index+1) / len(train_tensor), loss.item(), iou_value, lr_3))
            # if batch_index % 100 == 0:
            #     loss, current = loss.item(), (batch_index + 1) * len(data)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        # if warmrestart: 
        #     if scheduler_cosrest.T_cur == scheduler_cosrest.T_i - 1:
        #     # # 减小初始学习率
            
        #         initial_lr = scheduler_cosrest.base_lrs[0]* decay_factor
                                
        #     #     # 更新调度器的 base_lrs
        #         scheduler_cosrest.base_lrs = [initial_lr for _ in scheduler_cosrest.base_lrs]

        #     # 杰哥说过拟合时增大可能更有用

        #     scheduler_cosrest.step()
        # if warmrestart:
        #     scheduler_cosan.step()

        train_iou_value = train_iou.compute().item()
        epoch_loss = epoch_loss/(batch_index+1)
        print(f"Train Epoch: {epoch} - Jaccard Index (IoU): {train_iou_value:.4f}, CEloss: {epoch_loss:.4f}")
        writer_train.add_scalar('IoU', train_iou_value, epoch)
        writer_train.add_scalar('Learning Rate', lr_3, epoch)
        writer_train.add_scalar('loss', epoch_loss, epoch)
        
        # if last_loss!=0 and (epoch_loss-last_loss)/last_loss>=50:
        if epoch %8 == 0:
            print("writting model weight parameters and gradient ...")
            for name, params in model.named_parameters():
                writer_wegra.add_histogram(name + '_data', params.clone().cpu().data.numpy(), epoch)
                writer_wegra.add_histogram(name + '_grad', params.grad.clone().cpu().data.numpy(), epoch)
                
            print("Done")

        # fig = train_iou.plot()
        # writer.add_figure('Training IoU Plot', fig, epoch)
        # plt.close(fig)  # Close the figure to free memory

        if epoch %5 == 0:
            model.eval()
            val_iou.reset()
            # total = 0
            all_preds = []
            with torch.no_grad():
                epoch_loss=0.0
                for batch_index, (val_data, val_target) in enumerate(vali_tensor):
                    val_data, val_target = val_data.to(device), val_target.to(device)
                    pred = model(val_data)
                    
                    v_loss = loss_f(pred, val_target)
                    predictions = torch.argmax(pred, dim=1) 

                    val_iou.update(predictions, val_target)
                    iou_value = val_iou(predictions, val_target).item()
                    # iou_value = bat_iou(predictions, val_target).item()
                   
                    # print('val_loss: ', v_loss.item())
                    epoch_loss += v_loss.item()
                
                    all_preds.append(predictions.cpu())
                    print('Validation Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.12f} \tIoU: {:.5f}'.format(
                epoch, (batch_index+1) * len(val_data), len(vali_tensor.dataset),
                       100. * (batch_index+1) / len(vali_tensor), v_loss.item(), iou_value))

                val_iou_value = val_iou.compute().item()
                mvloss = epoch_loss / (batch_index+1)

                # lr_3 = optimizer.param_groups[0]['lr']
                scheduler_rlr.step(mvloss)
                # rlr = optimizer.param_groups[0]['lr']
                # if rlr != lr_3:
                #     print(f"Learning rate changed from {lr_3} to {rlr} due to validation loss plateau.")
                #     # 更新 CosineAnnealingWarmRestarts 调度器的 base_lrs 并重置其状态
                #     # scheduler_cosrest.base_lrs = [rlr for _ in scheduler_cosrest.base_lrs]
                #     # scheduler_cosrest.T_cur = 0  # 重置周期，使其从头开始
                #     scheduler_cosan.base_lrs = [rlr for _ in scheduler_cosan.base_lrs]
                #     scheduler_cosan.last_epoch=10

                print(f"Validation - Epoch: {epoch} - IoU: {val_iou_value:.5f}, CEloss: {mvloss:.12f}")
                writer_vali.add_scalar('IoU', val_iou_value, epoch)
                writer_vali.add_scalar('loss', mvloss, epoch)

                # pr curves:
                # all_preds = torch.cat(all_preds)
                # # writer_vali.add_pr_curve(f'PR_curve_epoch_{epoch}', tv_label, all_preds, global_step=epoch)
                # writer_vali.add_pr_curve(f'PR_curve', tv_label, all_preds, global_step=epoch)
                
                if mvloss <= pre_mvloss:
                    print(f"now loss :{mvloss:>7f} better than previous best epoch{pre_epoch:>3d}:{pre_mvloss:>7f} ")
                    model_name = model_prefix+f'_epoch{epoch:03d}-valoss{mvloss:.3f}.pth'
                    # torch.save(model, os.path.join(save_path, model_name))
                    torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                        }, os.path.join(save_path, model_name))
                    pre_mvloss = mvloss
                    pre_epoch = epoch
                    estop_count=0
                else:
                    estop_count+=1
                    # if estop_count==10 and not warmrestart: warmrestart=True
                    if mvloss-pre_mvloss/pre_mvloss>=2  and not onetime:
                        onetime = True
                        torch.save({
                            'model_state_dict':model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        },os.path.join(save_path, 'f{epoch:03d}_badstate.pth'))
                    if estop_count==30:
                        break
    torch.save(model, os.path.join(save_path, model_prefix+'_final_epoch'+ f'{epoch:04d}'+'.pth'))
    return


class focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes = 2, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma
        
        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))
        
    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数 这里要改(preds.view(-1,preds.size(-1)),-1改成1就行)
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # preds = preds.permute(0,)
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(1))
        alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = alpha.gather(0,labels.view(-1))
        # alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
    
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 扁平化输入和目标张量
        inputs = torch.argmax(inputs,dim=1)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算交集
        intersection = (inputs * targets).sum()
        
        # 计算 Dice 系数
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        # 返回 Dice Loss
        return 1 - dice

class Combine_loss(nn.Module):
    def __init__(self, device, cew = [2.0, 8.0]):
        super(Combine_loss,self).__init__()
        self.cew = torch.tensor(cew).to(device)
        # self.diceloss = DiceLoss()
        self.bceloss = nn.CrossEntropyLoss(weight=self.cew)
        self.focloss = focal_loss(alpha=0.25,gamma=1)
        # self.lova = lovasz_softmax()
    def forward(self,pred,tar):
        # dice = self.diceloss(pred, tar)
        bce = self.bceloss(pred, tar)
        foc = self.focloss(pred, tar)
        # lova = self.lova(pred, tar)
        lova = lovasz_softmax(pred, tar)
        # return (dice+bce+foc)/3
        return (bce+lova)/2

if __name__ == '__main__':
        
    # NUM_train = 507
    # NUM_tv = 200
    Batch_Size = 128
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # 根据具体情况设置
    torch.backends.cudnn.deterministic = False

    data_path = r'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_mat/bina_sicufk.mat'
    datar = h5py.File(data_path, 'r')
    data = datar['data'][:].transpose(0,3,1,2)
    label = datar['label_b'][:]
    train_data = torch.from_numpy(data).float()
    train_label = torch.from_numpy(label).long()
    train_tensor = TensorDataset(train_data,train_label)
    train_tensor = DataLoader(train_tensor, batch_size=Batch_Size,shuffle=True)

    test_path = r'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_mat/test_sicufk.mat'
    testr = h5py.File(test_path, 'r')
    td = testr['data'][:].transpose(0,3,1,2)
    tl = testr['label_b'][:]
    test_data = torch.from_numpy(td).float()
    test_label = torch.from_numpy(tl).long()
    test_tensor = TensorDataset(test_data,test_label)
    test_tensor = DataLoader(test_tensor, batch_size=Batch_Size)

    # # 原图像获取方式
    # output_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/exped_output/'
    # label_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/label/label/'
    # img_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/data1/data/'
    # img1_list = getimg(img_path,num=NUM_train, j=0)
    # img_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/data2/data/'
    # img2_list = getimg(img_path,num=NUM_train, j=900)
    # label_list = getimg(label_path,num=NUM_train, j=0)
    # img_list = np.stack((img1_list,img2_list),axis = 1)
    # img_path2 = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/test/predicted_input_data/'
    # label_path2 = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/test/predicted_label/'
    # # output_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/test/outputv2/'
    # img1_list = getimg(img_path2,num=100, j =0)
    # img2_list = getimg(img_path2,num=100, j =100)
    # test_img_list = np.stack((img1_list,img2_list),axis = 1)
    # test_label_list = getimg(label_path2, num=100, j=0)
    # img_list = np.vstack((img_list, test_img_list))
    # label_list = np.vstack((label_list, test_label_list))

    # Input = torch.from_numpy(img_list)
    # Output = torch.from_numpy(label_list.astype(np.int64))   
    # train_tensor = TensorDataset(Input, Output)
    # train_tensor = DataLoader(train_tensor, batch_size=Batch_Size, shuffle=True)     

    # test_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/test/test2/data/'
    # test_label_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/test/test2/label/'
    # test_output_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/test/outputv2/'
    # img1_list = getimg(test_path,num=NUM_tv, j =0)
    # img2_list = getimg(test_path,num=NUM_tv, j =200)
    # img_list = np.stack((img1_list,img2_list),axis = 1)
    # label_list = getimg(test_label_path,num=NUM_tv,j=0)

    # tv_image = torch.from_numpy(img_list)
    # tv_label = torch.from_numpy(np.array(label_list).astype(np.int64)) 
    # vali_tensor = TensorDataset(tv_image, tv_label)
    # vali_tensor = DataLoader(vali_tensor, batch_size=Batch_Size)
    '''_'''

    # model = MLPm(image_size=128,
    #       patch_size=16,
    #       dim=1024,
    #       depth=15,
    #       filters_num=[16,8]).to(device)
    # model = S2MLPv2(in_channels=20,d_model=[384,768],depth=[10,14],drop=dropt) 
    # we_loss = [0.1, 0.9]
    # path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/save_model/torch/s2filtn3_epoch380-valoss0.018.pth'
   
     #(weight=we_loss)
    # loss_fn = Combine_loss(device=device)
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters())
    # total_params = 0
    
    # checkpoint = torch.load(path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # # 如果有保存的epoch和损失，也可以加载
    # epoch = checkpoint.get('epoch', 0)
    # loss = checkpoint.get('loss', 0.0)
    
    
    save_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/save_model'
    logdir = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/output/logdep25'
    dropt = 0.5
    wd = 1e-2
    il = 1e-3
    # for param in model.parameters():
    #     num_params = param.numel()
    #     total_params += num_params
    #     print(f'Layer: {param.shape}, Parameters: {num_params}')
    # print('Total parameters:', total_params/1e6,'M')
    # model2 = S2MLPv2(filters_num=[64,32,8]) 
    # summary(model2, input_size=(Batch_Size, 2, 128, 128))
    # train(train_tensor=train_tensor, vali_tensor=vali_tensor,
    #       model=model2, loss_f=loss_fn,
    #       epoches=1000,
    #       save_path=save_path, logdir=logdir,
    #       device=device,model_prefix='s2filtn3_8k', path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/save_model/torch/s2filtn3_epoch410-valoss0.017.pth', contin=False)
    base_model = 's2dep25noLN'+'dp'+f"{dropt:.1f}"
    loss = 'bc0'
    weight_decay = 'wd'+f"{wd:.0e}"
    initial_lr = 'inlr+reduce'+f"{il:.0e}"
    d = 'data'+f'{data.shape[0]/1000:.0f}k'
    model_prefix = '_'.join((base_model,loss,'Adamw',weight_decay,initial_lr,d))
    # 把convffn block的LNRES取消了


    model = S2MLPv2(in_channels=20,d_model=[384,768],depth=[2,5],drop=dropt) 
    summary(model, input_size=(Batch_Size, 20, 20, 20))
    train(train_tensor=train_tensor, vali_tensor=test_tensor,
          model=model, loss_f=loss_fn,
          epoches=600,wd=wd,
          save_path=save_path, logdir=logdir,
          device=device,model_prefix=model_prefix, 
          path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/save_model/s2dep1014dp0.0_bc0_wd0e+00_inlr+reduce1e-03_data5k_epoch055-valoss0.047.pth', 
          contin=False,
          nowwhatturn=model_prefix)
    

    
    