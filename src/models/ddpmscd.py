import torch
import torch.nn as nn 
import models as Model




class ddpmscd(nn.Module):
    def __init__(self, opt):
        super(ddpmscd, self).__init__()
        self.semantic_segmentation = Model.create_SS_model(opt) # 모델 객체 생성 
        self.change_detection = Model.create_CD_model(opt) # 모델 객체 생성 
        self.ssNet = self.semantic_segmentation.netSS.module
        self.cdNet = self.change_detection.netCD.module
        
    def optimize_parameters(self, seg_A, seg_B, change, data):
        
        binary_image_tensor = 1 - torch.any(data['L1'] != 1, dim=1, keepdim=True).squeeze(dim=1).float()
        data['L'] = binary_image_tensor # tensor shape을 (n, c, h, w) 에서 (n, h, w) 로 맞춰줘야함. 
        data['L'] = 1 - data['L']
        
        label_A = self.semantic_segmentation.get_label(data['L1']).cuda()
        label_B = self.semantic_segmentation.get_label(data['L2']).cuda()
        
        dim = list(data['L'].shape)
        dim.append(-1) # 1, 256, 256, -1
        b, h, w, c = seg_A.reshape(dim).shape # 1, 256, 256, 7

        predmap_A = seg_A.view(b, c, h, w)
        predmap_B = seg_B.view(b, c, h, w)
            
        l_cd = self.change_detection.loss_func(change, data['L'].long()) # L1은 semantic change detection label, 흑백처리 하는 게 필요
        l_sem_A = self.semantic_segmentation.loss_sem(predmap_A, label_A)
        l_sem_B = self.semantic_segmentation.loss_sem(predmap_B, label_B)
        
        self.change_detection.log_dict['l_cd'] = l_cd.item()
        self.semantic_segmentation.log_dict['l_sem'] = (l_sem_A*0.5 + l_sem_B*0.5).item()
        
        loss = 0.01 * l_cd + (l_sem_A*0.5 + l_sem_B*0.5) # change에 대한 term을 적은 비중으로 업데이트하게 수정함. 
        return loss
        
        # net.cd, net.ss 에 각각 log 추가 
        
    
    def collect_running_batch_states(self, seg_A, seg_B, change, data):
        
        dim = data['L'].shape[-1]
        self.semantic_segmentation.running_metrics = self.semantic_segmentation._update_ss_metrics(seg_A, seg_B, data, dim)
        self.semantic_segmentation.log_dict['running_Fscd'] = self.semantic_segmentation.running_metrics[0].item()
        self.semantic_segmentation.log_dict['running_mIoU'] = self.semantic_segmentation.running_metrics[1].item()
        self.semantic_segmentation.log_dict['running_SeK'] = self.semantic_segmentation.running_metrics[2].item()
        
        if 'epoch_sum_Fscd' not in self.semantic_segmentation.log_dict:
            self.semantic_segmentation.log_dict['epoch_sum_Fscd'] = 0
            self.semantic_segmentation.log_dict['epoch_sum_mIoU'] = 0
            self.semantic_segmentation.log_dict['epoch_sum_SeK'] = 0

        self.semantic_segmentation.log_dict['epoch_sum_Fscd'] += self.semantic_segmentation.log_dict['running_Fscd']
        self.semantic_segmentation.log_dict['epoch_sum_mIoU'] += self.semantic_segmentation.log_dict['running_mIoU']
        self.semantic_segmentation.log_dict['epoch_sum_SeK'] += self.semantic_segmentation.log_dict['running_SeK']


        self.change_detection.running_acc = self.change_detection._update_metric(change, data)
        self.change_detection.log_dict['running_acc'] = self.change_detection.running_acc.item()

    
    def collect_epoch_states(self):
        return 

    def forward(self, fss_A, fss_B, fcd_A, fcd_B):
        
        # batch에 대한 처리 필요 
        
        pred_ss_A = []
        pred_ss_B = []
        
        fss_A = fss_A.squeeze(0)
        fss_B = fss_B.squeeze(0)
        
        if isinstance(self.ssNet, nn.DataParallel):
            pred_ss_A = self.ssNet.module.forward(fss_A).unsqueeze(0)
            pred_ss_B = self.ssNet.module.forward(fss_B).unsqueeze(0)
            
        else:
            pred_ss_A = self.ssNet.forward(fss_A).unsqueeze(0)
            pred_ss_B = self.ssNet.forward(fss_B).unsqueeze(0)

        change = self.cdNet(fcd_A, fcd_B)
        
        return pred_ss_A, pred_ss_B, change
