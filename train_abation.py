import os
import time
import random
import numpy as np
import torch.nn as nn
import torch.autograd
from skimage import io
from torch import optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
working_path = os.path.dirname(os.path.abspath(__file__))

from utils.loss import CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity,SCA_Loss
from utils.utils import accuracy, SCDD_eval_all, AverageMeter
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#Data and model choose
torch.set_num_threads(4)
###############################################
from datasets import RS_ST as RS
# from models.BiSRNet import BiSRNet as Net
from models.cdsc import cdsc as Net
# 生成一个随机5位整数
import math
nums = math.floor(1e5 * random.random())
seed = 66536
print(seed)
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
    #  torch.backends.cudnn.deterministic = True
# # 设置随机数种子
setup_seed(seed)

# from models.SSCDl import SSCDl as Net
NET_NAME = 'cdsc_sca_wo_sce'
DATA_NAME = 'SECOND_'+"cdsc_aba"
###############################################    
rank = '1','2','3'
#Training options
###############################################
args = {
    'train_batch_size': 4,
    'val_batch_size': 4,
    'lr': 0.0001,
    'epochs': 50,
    'gpu': True,
    'lr_decay_power': 1.5,
    'weight_decay': 1e-2,
    'momentum': 0.9,
    'print_freq': 50,
    'predict_step': 5,
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', DATA_NAME),
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME, NET_NAME),
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'pretrained.pth')
}
###############################################

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
writer = SummaryWriter(args['log_dir'])

def main():        
    net = Net(3, output_nc=RS.num_classes).cuda()
    net = nn.DataParallel(net)
    print("Initializing backbone weights from: " + "/data/jw/semantic_cd/scd_jw/pretrained/backbone_weights_b2.pth")
    model_dict      = net.state_dict()
    pretrained_dict = torch.load("/data2/jw/semantic_cd/scd_jw/pretrained/backbone_weights_b2.pth")
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            if k in model_dict.keys():
                print(k,np.shape(model_dict[k]), np.shape(v))
            no_load_key.append(k)
#------------------------------------------------------#
#   显示没有匹配上的Key
#------------------------------------------------------#
    print("\nSuccessful Load Key:", str(load_key)[:5000], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:5000], "……\nFail To Load Key num:", len(no_load_key))
    model_dict.update(temp_dict)
    net.load_state_dict(model_dict)
        
    train_set = RS.Data('train', random_flip=True)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)
    val_set = RS.Data('val')
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)
    
    criterion = CrossEntropyLoss2d(ignore_index=0).cuda()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], betas=(0.9, 0.999),weight_decay=0.01, )
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)

    train(train_loader, net, criterion, optimizer, scheduler, val_loader)
    writer.close()
    print('Training finished.')

def train(train_loader, net, criterion, optimizer, scheduler, val_loader):
    bestaccT=0
    bestFscdV=0.0
    bestloss=1.0
    bestaccV = 0.0
    begin_time = time.time()
    all_iters = float(len(train_loader)*args['epochs'])
    criterion_sc = SCA_Loss().cuda()
    curr_epoch=0
    while True:
        torch.cuda.empty_cache()
        net.train()
        #freeze_model(net.FCN)
        start = time.time()
        acc_meter = AverageMeter()
        train_seg_loss = AverageMeter()
        train_bn_loss = AverageMeter()
        train_sc_loss = AverageMeter()
        
        curr_iter = curr_epoch*len(train_loader)
        for i, data in enumerate(train_loader):

            running_iter = curr_iter+i+1
            adjust_lr(optimizer, running_iter, all_iters)
            imgs_A, imgs_B, labels_A, labels_B = data
            if args['gpu']:
                imgs_A = imgs_A.cuda().float().detach()
                imgs_B = imgs_B.cuda().float().detach()
                labels_bn = (labels_A>0).unsqueeze(1).cuda().float().detach()
                labels_A = labels_A.cuda().long().detach()
                labels_B = labels_B.cuda().long().detach()

            optimizer.zero_grad()
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)
            
            assert outputs_A.size()[1] == RS.num_classes
            
            loss_seg = criterion(outputs_A*labels_bn, labels_A) * 0.5 +  criterion(outputs_B*labels_bn, labels_B) * 0.5 

            # print(out_change.shape)    
            loss_bn = weighted_BCE_logits(out_change, labels_bn)

            # print(outputs_A.size(),outputs_B.size(),labels_bn.size())
            # loss_sc = criterion_SCA(outputs_A, outputs_B, labels_bn)
            loss_sc = criterion_sc(outputs_A, outputs_B, labels_bn)
            loss = loss_seg + loss_bn + loss_sc
            loss.backward()
            optimizer.step()
 
            labels_A = labels_A.cpu().detach().numpy()
            labels_B = labels_B.cpu().detach().numpy()
            outputs_A = outputs_A.cpu().detach()
            outputs_B = outputs_B.cpu().detach()
            change_mask = F.sigmoid(out_change).cpu().detach()>0.5
            preds_A = torch.argmax(outputs_A, dim=1)
            preds_B = torch.argmax(outputs_B, dim=1)
            preds_A = (preds_A*change_mask.squeeze().long()).numpy()
            preds_B = (preds_B*change_mask.squeeze().long()).numpy()

            # batch_valid_sum = 0
            acc_curr_meter = AverageMeter()
            for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
                acc_A, valid_sum_A = accuracy(pred_A, label_A)
                acc_B, valid_sum_B = accuracy(pred_B, label_B)
                acc = (acc_A + acc_B)*0.5
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_seg_loss.update(loss_seg.cpu().detach().numpy())
            train_bn_loss.update(loss_bn.cpu().detach().numpy())
            train_sc_loss.update(loss_sc.cpu().detach().numpy())

            curr_time = time.time() - start

            if (i + 1) % args['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train seg_loss %.4f bn_loss %.4f sc_loss %.4f acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_seg_loss.val, train_bn_loss.val, train_sc_loss.val, acc_meter.val*100)) #sc_loss %.4f, train_sc_loss.val, 
                writer.add_scalar('train seg_loss', train_seg_loss.val, running_iter)
                writer.add_scalar('train sc_loss', train_sc_loss.val, running_iter)
                writer.add_scalar('train accuracy', acc_meter.val, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        Fscd_v, mIoU_v, Sek_v, acc_v, loss_v = validate(val_loader, net, criterion, curr_epoch)
        if acc_meter.avg>bestaccT: bestaccT=acc_meter.avg
        if Fscd_v>bestFscdV:
            bestFscdV=Fscd_v
            bestaccV=acc_v
            bestloss=loss_v
            torch.save(net.state_dict(), os.path.join(args['chkpt_dir'], NET_NAME+'_%de_mIoU%.2f_Sek%.2f_Fscd%.2f_OA%.2f.pth'\
                %(curr_epoch, mIoU_v*100, Sek_v*100, Fscd_v*100, acc_v*100)) )
        print('Total time: %.1fs Best rec: Train acc %.2f, Val Fscd %.2f acc %.2f loss %.4f' %(time.time()-begin_time, bestaccT*100, bestFscdV*100, bestaccV*100, bestloss))
        curr_epoch += 1
        #scheduler.step()
        if curr_epoch >= args['epochs']:
            return

def validate(val_loader, net, criterion, curr_epoch):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    acc_meter = AverageMeter()

    preds_all = []
    labels_all = []
    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels_A, labels_B = data
        if args['gpu']:
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            labels_A = labels_A.cuda().long()
            labels_B = labels_B.cuda().long()
            labels_bn = (labels_A>0).unsqueeze(1).cuda().float()
        with torch.no_grad():
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)
            loss_A = criterion(outputs_A*labels_bn, labels_A)
            loss_B = criterion(outputs_B*labels_bn, labels_B)
            loss = loss_A * 0.5 + loss_B * 0.5
        val_loss.update(loss.cpu().detach().numpy())
        
        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = F.sigmoid(out_change).cpu().detach()>0.5
        preds_A = torch.argmax(outputs_A, dim=1)
        preds_B = torch.argmax(outputs_B, dim=1)
        preds_A = (preds_A*change_mask.squeeze().long()).numpy()
        preds_B = (preds_B*change_mask.squeeze().long()).numpy()
        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, valid_sum_A = accuracy(pred_A, label_A)
            acc_B, valid_sum_B = accuracy(pred_B, label_B)
            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)
            acc = (acc_A + acc_B)*0.5
            acc_meter.update(acc)

        if curr_epoch%args['predict_step']==0 and vi==0:
            pred_A_color = RS.Index2Color(preds_A[0])
            pred_B_color = RS.Index2Color(preds_B[0])
            io.imsave(os.path.join(args['pred_dir'], NET_NAME+'_A.png'), pred_A_color)
            io.imsave(os.path.join(args['pred_dir'], NET_NAME+'_B.png'), pred_B_color)
            print('Prediction saved!')
    
    Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, RS.num_classes)

    curr_time = time.time() - start
    print('%.1fs Val loss: %.2f Fscd: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f'\
    %(curr_time, val_loss.average(), Fscd*100, IoU_mean*100, Sek*100, acc_meter.average()*100))

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Fscd', Fscd, curr_epoch)
    writer.add_scalar('val_Accuracy', acc_meter.average(), curr_epoch)

    return Fscd, IoU_mean, Sek, acc_meter.avg, val_loss.avg

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()            

def adjust_lr(optimizer, curr_iter, all_iter, init_lr=args['lr']):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args['lr_decay_power'])
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr
        
if __name__ == '__main__':
    main()