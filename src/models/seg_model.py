import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import models.networks as networks
from .base_model import BaseModel
from misc.metric_tools import ConfuseMatrixMeter
from misc.torchutils import get_scheduler
from typing import List
logger = logging.getLogger('base')
import numpy
import math 
from scipy import stats
from src.data_util import get_palette, get_class_names
from src.utils import colorize_mask

    
class CrossEntropyLoss2d(nn.Module): # for segmentation loss between label scd map & pred seg map 
    def __init__(self, weight=None, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index,
                                   reduction='mean')
    
    def forward(self, inputs, targets): # target label: 0부터 6 사이의 정수 값을 가짐 
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)
    
class ChangeSimilarity(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """
    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)
        
    def forward(self, x1, x2, label_change):
        b,c,h,w = x1.size()
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0,2,3,1)
        x2 = x2.permute(0,2,3,1)
        x1 = torch.reshape(x1,[b*h*w,c])
        x2 = torch.reshape(x2,[b*h*w,c])
        
        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        target = target - label_change.float()
        target = torch.reshape(target,[b*h*w])
        
        loss = self.loss_f(x1, x2, target)
        return loss    

class SS(BaseModel):
    def __init__(self, opt): #opt에 opt["seg_model"] 넣어줘야 함 
        super(SS, self).__init__(opt)
        # define network and load pretrained models
        self.netSS = self.set_device(networks.define_SS(opt))
        # set loss and load resume state
        self.loss_type = opt['model_ss']['loss_type']
        
        if self.loss_type == 'l_sem': # binary cross entropy 
            self.loss_func =nn.CrossEntropyLoss().to(self.device)
        else:
            raise NotImplementedError()
        
        if self.opt['phase'] == 'train':
            self.netSS.train()
            # find the parameters to optimize
            optim_ss_params = list(self.netSS.parameters())

            if opt['train']["optimizer"]["type"] == "adam":
                self.optSS = torch.optim.Adam(
                    optim_ss_params, lr=opt['train']["optimizer"]["lr"])
            elif opt['train']["optimizer"]["type"] == "adamw":
                self.optSS = torch.optim.AdamW(
                    optim_ss_params, lr=opt['train']["optimizer"]["lr"])
            else:
                raise NotImplementedError(
                    'Optimizer [{:s}] not implemented'.format(opt['train']["optimizer"]["type"]))
            self.log_dict = OrderedDict()
            
            #Define learning rate sheduler
            self.exp_lr_scheduler_netSS = get_scheduler(optimizer=self.optSS, args=opt['train'])
        else:
            self.netSS.eval()
            self.log_dict = OrderedDict()

        self.load_network()
        self.print_network()

        self.running_metric = ConfuseMatrixMeter(n_class=opt['model_ss']['out_channels'])
        self.len_train_dataloader = opt["len_train_dataloader"]
        self.len_val_dataloader = opt["len_val_dataloader"]

    # Feeding all data to the SS model
    def feed_data(self, feats_A, feats_B, data):
        self.fss_A = feats_A
        self.fss_B = feats_B
        self.data = self.set_device(data)
        
    def get_feature(self):
        self.optSS.zero_grad()
        self.dim = self.opt["model_ss"]["dim"]
        
        self.fss_A = self.collect_features(self.dim, self.fss_A) # input type은 feature map이 원소인 list 
        self.fss_B = self.collect_features(self.dim, self.fss_B) # return type은 (b, c, h, w) 
        
        b, c, h, w = self.fss_A.shape 
        # shape 확인 
        self.fss_A = self.fss_A.view(b, -1, c) # (b, c, h, w) -> (b, hxw, c)
        self.fss_B = self.fss_B.view(b, -1, c) 
        
    def get_label(self, data):
        pallete = get_palette(self.opt["datasets"]["train"]["name"])
        color_to_label = {}
        
        for i in range(0, len(pallete), 3):
            color_key = (pallete[i]/255., pallete[i+1]/255., pallete[i+2]/255.)
            color_to_label[color_key] = int(i/3)
        data = data.cpu()
        batch_size, _, height, width = data.shape
        label_tensor = torch.zeros((batch_size, height, width), dtype=torch.long)

        for batch_idx in range(batch_size):
            for color, label in color_to_label.items():
                mask = torch.all(torch.abs(data[batch_idx] - torch.tensor(color).view(1, 3, 1, 1))<0.001, dim=1).squeeze()
                label_tensor[batch_idx][mask] = label
    
        return label_tensor

        
    def optimize_parameters(self):
        b = self.feats_A.shape[0]
        loss_A = 0
        loss_B = 0
        self.pred_ss_A = []
        self.pred_ss_B = []
        
        label_A = self.get_label(self.data['L1']).cuda()
        label_B = self.get_label(self.data['L2']).cuda()
        if isinstance(self.netSS, nn.DataParallel):
            for i in range(b):
                self.pred_ss_A.append(self.netSS.module.forward(self.feats_A[i]))
                self.pred_ss_B.append(self.netSS.module.forward(self.feats_B[i]))
            
        else:
            for i in range(b):
                self.pred_ss_A.append(self.netSS(self.feats_A[i]))
                self.pred_ss_B.append(self.netSS(self.feats_B[i]))
                
        self.pred_ss_A = torch.stack(self.pred_ss_A)
        self.pred_ss_B = torch.stack(self.pred_ss_B)
        self.num_class = self.opt["model_ss"]["number_class"]
        
        b, h_w, c = self.pred_ss_A.shape
        h = int(np.sqrt(h_w))
        w = int(np.sqrt(h_w))
        predmap_A = self.pred_ss_A.view(b, c, h, w) # 4 x 7 x 256 x 256
        predmap_B = self.pred_ss_B.view(b, c, h, w)  
        loss_A = self.loss_sem(predmap_A, label_A) # 4 x 7 x 256 x 256, 4 x 7 (batch size 맞아야함)
        loss_B = self.loss_sem(predmap_B, label_B) 
        
        l_sem = loss_A* 0.5 + loss_B*0.5
        self.l_sem = l_sem
        self.log_dict['l_sem'] = l_sem.item()
        
    def loss_sem(self, seg_map, scd_label): # in BiSRNet
        loss_fun = CrossEntropyLoss2d(ignore_index=0) # 객체 initialize 
        return loss_fun(seg_map, scd_label) # 0번 (unlabel) 제외해서 segmentation loss 구함. 
    
    def test(self):
        
        ### self.get_feature 
        self.netSS.eval()
        self.dim = self.opt["model_ss"]["dim"]

        self.feats_A = self.collect_features(self.dim, self.feats_A) # input type은 feature map이 원소인 list 
        self.feats_B = self.collect_features(self.dim, self.feats_B) # return type은 (4, 8448, 256, 256) -> 4, 11520, 256, 256이어야 함. 
        
        b, c, h, w = self.feats_A.shape 
        # shape 확인 
        self.feats_A = self.feats_A.view(b, -1, c) # (b, c, h, w) -> (b, hxw, c)
        self.feats_B = self.feats_B.view(b, -1, c) 
    
        with torch.no_grad():
            self.pred_ss_A = []
            self.pred_ss_B = []

            if isinstance(self.netSS, nn.DataParallel):
                for i in range(b):
                    self.pred_ss_A.append(self.netSS.module.forward(self.feats_A[i]))
                    self.pred_ss_B.append(self.netSS.module.forward(self.feats_B[i]))
                
            else:
                for i in range(b):
                    self.pred_ss_A.append(self.netSS(self.feats_A[i]))
                    self.pred_ss_B.append(self.netSS(self.feats_B[i]))
                    
            self.pred_ss_A = torch.stack(self.pred_ss_A)
            self.pred_ss_B = torch.stack(self.pred_ss_B)
    
    # Get feature representations for a given image
    def get_feats(self, t):
        self.netSS.eval()
        A = self.data["A"]
        B = self.data["B"]
        with torch.no_grad():
            if isinstance(self.netSS, nn.DataParallel):
                fe_A_seg, fd_A_seg = self.netSS.module.feats(A, t)
                fe_B_seg, fd_B_seg = self.netSS.module.feats(B, t)
            else:
                fe_A_seg, fd_A_seg = self.netSS.feats(A, t)
                fe_B_seg, fd_B_seg = self.netSS.feats(B, t)
        self.netSS.train()
        return fe_A_seg, fd_A_seg, fe_B_seg, fd_B_seg

    def sample(self, batch_size=1, continous=False):
        self.netSS.eval()
        with torch.no_grad():
            if isinstance(self.netSS, nn.DataParallel):
                self.sampled_img = self.netSS.module.sample(batch_size, continous)
            else:
                self.sampled_img = self.netSS.sample(batch_size, continous)
        self.netSS.train()

    def collect_features(self, dim, activations: List[torch.Tensor]): #지금은 4개의 배치 중 0번만 사용하는데 이걸 다 사용해야 할듯 
        assert all([isinstance(acts, torch.Tensor) for acts in activations]) # timestep * featuremap 길이를 갖는 list임
        size = tuple(dim[:-1])
        resized_activations = []
        for feats in activations:
            # feats = feats[sample_idx][None] feature 전체를 사용 
            feats = nn.functional.interpolate( 
                feats, size=size, mode="bilinear"
            )
            resized_activations.append(feats)
            
        ret = torch.cat(resized_activations, dim=1) # batch 
        return ret
 

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, segMap_A, segMap_B, data):
        out_dict = OrderedDict()
        out_dict['pred_ss_A'] = torch.Tensor(segMap_A)
        out_dict['pred_ss_B'] = torch.Tensor(segMap_B)
        
        out_dict['gt_cm_A'] = data["L1"]
        out_dict['gt_cm_B'] = data["L2"]
        return out_dict
    
    def get_current_predmap(self):
        out_dict = OrderedDict()
        out_dict['pred_ss_A'] = self.pred_ss_A
        out_dict['pred_ss_B'] = self.pred_ss_B
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netSS)
        if isinstance(self.netSS, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netSS.__class__.__name__,
                                             self.netSS.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netSS.__class__.__name__)

        logger.info(
            'Network SS structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, is_best_model = False):
        ss_gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'gen_{}.pth'.format(epoch))
        ss_opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'opt_{}.pth'.format(epoch))
        # SS
        network = self.netSS
        if isinstance(self.netSS, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, ss_gen_path)
        
        # Save SS optimizer parameters
        opt_state = {'epoch': epoch,
                     'scheduler': None, 
                     'optimizer': None}
        opt_state['optimizer'] = self.optSS.state_dict()
        torch.save(opt_state, ss_opt_path)

        logger.info(
            'Saved current SSmodel in [{:s}] ...'.format(ss_gen_path))

    def load_network(self):
        load_path = self.opt['path_ss']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for SS [{:s}] ...'.format(load_path))
            gen_path = '{}.pth'.format(load_path)
            print(gen_path)
            # gen
            network = self.netSS
            if isinstance(self.netSS, nn.DataParallel):
                network = network.module
            
            loaded_state_dict = torch.load(gen_path)['model_state_dict']
                        
            state_dict = OrderedDict()
            for key, value in loaded_state_dict.items():
                if key.startswith('module.'):
                    new_key = key[len('module.'):]
                else:
                    new_key = key
                state_dict[new_key] = value
            
            
            network.load_state_dict(state_dict, strict=True)
            
                
    # Functions related to computing performance metrics for semantic segmentation 
    def _update_metric(self):
        """
        update metric
        """
        # compute mean IoU 
        
        # prediction map 
        SS_pred_A = self.pred_ss_A.detach()
        SS_pred_A = torch.argmax(SS_pred_A, dim=1)
        
        SS_pred_B = self.pred_ss_B.detach()
        SS_pred_B = torch.argmax(SS_pred_B, dim=1) # [256x256, 8] -> [256x256] 1 dimension tensor로 변환, voting하는 과정임 

        current_mIoU_A = self.running_metric.update_ss(pr=SS_pred_A.cpu().numpy(), gt=self.data['L1'].detach().cpu().numpy())
        current_mIoU_B = self.running_metric.update_ss(pr=SS_pred_B.cpu().numpy(), gt=self.data['L2'].detach().cpu().numpy())
        
        current_mIoU = (current_mIoU_A + current_mIoU_B)/2 
        return current_mIoU
    
    def fast_hist(self, a, b, n):
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

    def get_hist(self, image, label, num_class):
        hist = np.zeros((num_class, num_class))
        hist += self.fast_hist(image.flatten(), label.flatten(), num_class)
        return hist
    
    def cal_kappa(self, hist):
        if hist.sum() == 0:
            po = 0
            pe = 1
            kappa = 0
        else:
            po = np.diag(hist).sum() / hist.sum()
            pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
            if pe == 1:
                kappa = 0
            else:
                kappa = (po - pe) / (1 - pe)
        return kappa
    
    def _update_ss_metrics(self, seg_A, seg_B, data, dim): # preds랑 labels 파악하기
        preds = []
        labels = [] # pred_ss: (b, 256x256, 8) / # labels: (b, 3, 256, 256) 
        # 둘 다 index map (b, 256, 256) 으로 바꿔야 함. 
        num_classes = seg_A.shape[-1]
        pred_ss_A = torch.argmax(seg_A, dim = 2).reshape(-1, dim, dim) # 현재 1, 7, 256, 256 -> 1, 256, 256 
        pred_ss_B = torch.argmax(seg_B, dim = 2).reshape(-1, dim, dim)
        labels_A  = self.get_label(data['L1'])
        labels_B  = self.get_label(data['L2'])

        for pred_A, pred_B, label_A, label_B in zip(pred_ss_A, pred_ss_B, labels_A, labels_B):
            preds.append(pred_A)
            preds.append(pred_B)
            labels.append(label_A)
            labels.append(label_B)
            
        hist = np.zeros((num_classes, num_classes))
        
        for pred, label in zip(preds, labels):
            infer_array = np.array(pred.cpu())
            label_array = np.array(label.cpu())
            assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
            hist += self.get_hist(infer_array, label_array, num_classes)
        
        hist_fg = hist[1:, 1:]
        c2hist = np.zeros((2, 2))
        c2hist[0][0] = hist[0][0]
        c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
        c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
        c2hist[1][1] = hist_fg.sum()
        hist_n0 = hist.copy()
        hist_n0[0][0] = 0
        kappa_n0 = self.cal_kappa(hist_n0)
        iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
        IoU_fg = iu[1]
        mIoU = (iu[0] + iu[1]) / 2
        Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
        
        eps = 1e-5
        pixel_sum = hist.sum()
        change_pred_sum  = pixel_sum - hist.sum(1)[0].sum() + eps
        change_label_sum = pixel_sum - hist.sum(0)[0].sum() + eps
        change_ratio = change_label_sum/pixel_sum
        SC_TP = np.diag(hist[1:, 1:]).sum() + 1e-5
        SC_Precision = SC_TP/change_pred_sum
        SC_Recall = SC_TP/change_label_sum
        F_scd = stats.hmean([SC_Precision, SC_Recall])
        return F_scd, mIoU, Sek
    
    # Collecting status of the current running batch
    def _collect_running_batch_states(self):
        self.running_metrics = self._update_ss_metrics()
        self.log_dict['running_Fscd'] = self.running_metrics[0].item()
        self.log_dict['running_mIoU'] = self.running_metrics[1].item()
        self.log_dict['running_SeK'] = self.running_metrics[2].item()
        
        if 'epoch_sum_Fscd' not in self.log_dict:
            self.log_dict['epoch_sum_Fscd'] = 0
            self.log_dict['epoch_sum_mIoU'] = 0
            self.log_dict['epoch_sum_SeK'] = 0

        self.log_dict['epoch_sum_Fscd'] += self.log_dict['running_Fscd']
        self.log_dict['epoch_sum_mIoU'] += self.log_dict['running_mIoU']
        self.log_dict['epoch_sum_SeK'] += self.log_dict['running_SeK']

    # Collect the status of the epoch
    def _collect_epoch_states(self, steps_per_epoch):
        #scores = self.running_metric.get_scores()
        #self.epoch_IoU = scores['mIoU']
        self.log_dict['epoch_Fscd'] = self.log_dict['epoch_sum_Fscd'] / steps_per_epoch
        self.log_dict['epoch_mIoU'] = self.log_dict['epoch_sum_mIoU'] / steps_per_epoch
        self.log_dict['epoch_SeK'] = self.log_dict['epoch_sum_SeK'] / steps_per_epoch

        self.log_dict['epoch_sum_Fscd'] = 0
        self.log_dict['epoch_sum_mIoU'] = 0
        self.log_dict['epoch_sum_SeK'] = 0
        #for k, v in scores.items():
            #self.log_dict[k] = v
            #message += '%s: %.5f ' % (k, v)

    # Rest all the performance metrics
    def _clear_cache(self):
        self.running_metric.clear()

    # Finctions related to learning rate sheduler
    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_netSS.step()
        
    #utils about pixel classifier's output processing 
    def pixeltosegMap(self, seg_A, seg_B):
        
        batch = seg_A.shape[0]
        dim = int(seg_A.shape[1]**0.5)
        
        labelMap_A = torch.argmax(seg_A, dim=2, keepdim=False).reshape(-1, dim, dim).cpu().numpy()
        labelMap_B = torch.argmax(seg_B, dim=2, keepdim=False).reshape(-1, dim, dim).cpu().numpy()
        
        pallete = get_palette(self.opt["datasets"]["train"]["name"])
        segmap_A = np.zeros((batch, dim, dim, 3)) # 3 Channel 
        segmap_B = np.zeros((batch, dim, dim, 3))
        
        for i in range(batch):
            segmap_A[i] = colorize_mask(labelMap_A[i], pallete)
            segmap_B[i] = colorize_mask(labelMap_B[i], pallete)
        
        segmap_A = segmap_A / 255.0 # 0~1로 normalize
        segmap_B = segmap_B / 255.0 # 0~1로 normalize
        segmap_A = np.transpose(segmap_A, (0, 3, 1, 2)) 
        segmap_B = np.transpose(segmap_B, (0, 3, 1, 2)) 
        return segmap_A, segmap_B
        
    
    def labelMaptosegMap(self):
        pallete = get_palette(self.opt["datasets"]["train"]["name"])
        batch = self.pred_ss_A.shape[0]
        segmap_A = np.zeros((batch, self.dim[0], self.dim[1], 3)) # 3 Channel 
        segmap_B = np.zeros((batch, self.dim[0], self.dim[1], 3))
        
        for i in range(batch):
            segmap_A[i] = colorize_mask(self.pred_ss_A[i], pallete)
            segmap_B[i] = colorize_mask(self.pred_ss_B[i], pallete)
        
        segmap_A = segmap_A / 255.0 # 0~1로 normalize
        segmap_B = segmap_B / 255.0 # 0~1로 normalize
        segmap_A = np.transpose(segmap_A, (0, 3, 1, 2)) 
        segmap_B = np.transpose(segmap_B, (0, 3, 1, 2)) 
        return (segmap_A, segmap_B)
