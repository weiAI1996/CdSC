import os
import time
import argparse
import numpy as np
import torch.autograd
from skimage import io, exposure
from torch.nn import functional as F
from torch.utils.data import DataLoader
#################################
from datasets import RS_ST as RS
# from models.BiSRNet import BiSRNet as Net
# from models.Daudt.HRSCD3 import HRSCD3 as Net
from models.cdsc import cdsc as Net
import cv2
# from models.SSCDl import SSCDl as Net
DATA_NAME = 'ST'
#################################

class PredOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        
    def initialize(self, parser):
        working_path = os.path.dirname(os.path.abspath(__file__))
        parser.add_argument('--pred_batch_size', required=False, default=1, help='prediction batch size')
        parser.add_argument('--test_dir', required=False, default='/data2/jw/semantic_cd/datasets/JL11_TEST/', help='directory to test images')
        parser.add_argument('--pred_dir', required=False, default='./res', help='directory to output masks')
        parser.add_argument('--chkpt_path', required=False, default='/data2/jw/semantic_cd/Bi-SRNet-main/checkpoints/JL1_111/cdsc_sca_46e_mIoU96.81_Sek88.89_Fscd97.50_OA98.60.pth')
        self.initialized = True
        return parser
        
    def gather_options(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        self.opt = self.gather_options()
        return self.opt
        
def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')        

def main():
    begin_time = time.time()
    opt = PredOptions().parse()
    net = Net(3, RS.num_classes).cuda()
    checkpoint = torch.load(opt.chkpt_path, map_location='cuda:0')
    saved_weights = checkpoint
    new_state_dict = {}
    for k, v in saved_weights.items():
        if k.startswith('module.'):
            name = k[7:]  # remove the "module." prefix
        else :
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    
    test_set = RS.Data_test(opt.test_dir)
    test_loader = DataLoader(test_set, batch_size=opt.pred_batch_size)
    predict(net, test_set, test_loader, opt.pred_dir, flip=False, index_map=True, intermediate=False)    
    time_use = time.time() - begin_time
    print('Total time: %.2fs'%time_use)

def create_change_mask(image1, image2):
    mask = np.zeros_like(image1)
    conditions = [
        ((image1 == 1) & (image2 == 2), 1),
        ((image1 == 1) & (image2 == 3), 2),
        ((image1 == 1) & (image2 == 4), 3),
        ((image1 == 1) & (image2 == 5), 4),
        ((image1 == 2) & (image2 == 1), 5),
        ((image1 == 3) & (image2 == 1), 6),
        ((image1 == 4) & (image2 == 1), 7),
        ((image1 == 5) & (image2 == 1), 8),
        # ... 其他条件
    ]
    for condition, value in conditions:
        mask = np.where(condition, value, mask)
    return mask

def predict(net, pred_set, pred_loader, pred_dir, flip=False, index_map=False, intermediate=False):

    for vi, data in enumerate(pred_loader):
        imgs_A, imgs_B = data
        #imgs = torch.cat([imgs_A, imgs_B], 1)
        imgs_A = imgs_A.cuda().float()
        imgs_B = imgs_B.cuda().float()
        mask_name = pred_set.get_mask_name(vi)
        with torch.no_grad(): 
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)#,aux
            out_change = F.sigmoid(out_change)
        if flip:
            outputs_A = F.softmax(outputs_A, dim=1)
            outputs_B = F.softmax(outputs_B, dim=1)
            
            imgs_A_v = torch.flip(imgs_A, [2])
            imgs_B_v = torch.flip(imgs_B, [2])
            out_change_v, outputs_A_v, outputs_B_v = net(imgs_A_v, imgs_B_v)
            outputs_A_v = torch.flip(outputs_A_v, [2])
            outputs_B_v = torch.flip(outputs_B_v, [2])
            out_change_v = torch.flip(out_change_v, [2])
            outputs_A += F.softmax(outputs_A_v, dim=1)
            outputs_B += F.softmax(outputs_B_v, dim=1)
            out_change += F.sigmoid(out_change_v)
            
            imgs_A_h = torch.flip(imgs_A, [3])
            imgs_B_h = torch.flip(imgs_B, [3])
            out_change_h, outputs_A_h, outputs_B_h = net(imgs_A_h, imgs_B_h)
            outputs_A_h = torch.flip(outputs_A_h, [3])
            outputs_B_h = torch.flip(outputs_B_h, [3])
            out_change_h = torch.flip(out_change_h, [3])
            outputs_A += F.softmax(outputs_A_h, dim=1)
            outputs_B += F.softmax(outputs_B_h, dim=1)
            out_change += F.sigmoid(out_change_h)
            
            imgs_A_hv = torch.flip(imgs_A, [2,3])
            imgs_B_hv = torch.flip(imgs_B, [2,3])
            out_change_hv, outputs_A_hv, outputs_B_hv = net(imgs_A_hv, imgs_B_hv)
            outputs_A_hv = torch.flip(outputs_A_hv, [2,3])
            outputs_B_hv = torch.flip(outputs_B_hv, [2,3])
            out_change_hv = torch.flip(out_change_hv, [2,3])
            outputs_A += F.softmax(outputs_A_hv, dim=1)
            outputs_B += F.softmax(outputs_B_hv, dim=1)
            out_change += F.sigmoid(out_change_hv)
            out_change = out_change/4
                        
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = out_change.cpu().detach()>0.5
        change_mask = change_mask.squeeze()
        pred_A = torch.argmax(outputs_A, dim=1).squeeze()
        pred_B = torch.argmax(outputs_B, dim=1).squeeze()
        pred_A = (pred_A*change_mask.long()).numpy()
        pred_B = (pred_B*change_mask.long()).numpy()      
        JL1_mask = create_change_mask(pred_A,pred_B)

        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)        
        save_path = os.path.join(pred_dir, mask_name)
        print(mask_name)
        cv2.imwrite(save_path,JL1_mask)




if __name__ == '__main__':
    main()