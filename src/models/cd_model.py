import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import models.networks as networks
import torchvision as tv 
from .base_model import BaseModel
from misc.metric_tools import ConfuseMatrixMeter
from misc.torchutils import get_scheduler
logger = logging.getLogger('base')


class CD(BaseModel):
    def __init__(self, opt):
        super(CD, self).__init__(opt)
        # define network and load pretrained models
        self.netCD = self.set_device(networks.define_CD(opt))

        # set loss and load resume state
        self.loss_type = opt['model_cd']['loss_type']
        if self.loss_type == 'l_change': # binary cross entropy 
            self.loss_func =nn.CrossEntropyLoss().to(self.device)
        else:
            raise NotImplementedError()
        
        if self.opt['phase'] == 'train':
            self.netCD.train()
            # find the parameters to optimize
            optim_cd_params = list(self.netCD.parameters())

            if opt['train']["optimizer"]["type"] == "adam":
                self.optCD = torch.optim.Adam(
                    optim_cd_params, lr=opt['train']["optimizer"]["lr"])
            elif opt['train']["optimizer"]["type"] == "adamw":
                self.optCD = torch.optim.AdamW(
                    optim_cd_params, lr=opt['train']["optimizer"]["lr"])
            else:
                raise NotImplementedError(
                    'Optimizer [{:s}] not implemented'.format(opt['train']["optimizer"]["type"]))
            self.log_dict = OrderedDict()
            
            #Define learning rate sheduler
            self.exp_lr_scheduler_netCD = get_scheduler(optimizer=self.optCD, args=opt['train'])
        else:
            self.netCD.eval()
            self.log_dict = OrderedDict()

        self.load_network()
        self.print_network()

        self.running_metric = ConfuseMatrixMeter(n_class=opt['model_cd']['out_channels'])
        self.len_train_dataloader = opt["len_train_dataloader"]
        self.len_val_dataloader = opt["len_val_dataloader"]

    # Feeding all data to the CD model
    def feed_data(self, feats_A, feats_B, data):
        self.fcd_A = feats_A
        self.fcd_B = feats_B
        self.data = self.set_device(data)
         
    # Optimize the parameters of the CD model
    def optimize_parameters(self):
        self.optCD.zero_grad()
        self.pred_cm = self.netCD(self.feats_A, self.feats_B)
        if "L" not in self.data:
            binary_image_tensor = 1 - torch.any(self.data['L1'] != 1, dim=1, keepdim=True).squeeze(dim=1).float()
            self.data['L'] = binary_image_tensor # tensor shape을 (n, c, h, w) 에서 (n, h, w) 로 맞춰줘야함. 
            self.data['L'] = 1 - self.data['L']
        
        l_cd = self.loss_func(self.pred_cm, self.data['L'].long()) # L1은 semantic change detection label, 흑백처리 하는 게 필요 
        self.optCD.step()
        self.l_cd = l_cd
        self.log_dict['l_cd'] = l_cd.item()  

    # Testing on given data
    def test(self):
        self.netCD.eval()
        with torch.no_grad():
            if isinstance(self.netCD, nn.DataParallel):
                self.pred_cm = self.netCD.module.forward(self.feats_A, self.feats_B)
            else:
                self.pred_cm = self.netCD(self.feats_A, self.feats_B)
           
            if "L" not in self.data:
                binary_image_tensor = 1 - torch.any(self.data["L1"] != 1, dim=1, keepdim=True).squeeze(dim=1).float()
                self.data["L"] = binary_image_tensor # tensor shape을 (n, c, h, w) 에서 (n, h, w) 로 맞춰줘야함. 
                self.data["L"] = 1 - self.data["L"]  
                                  
        l_cd = self.loss_func(self.pred_cm, self.data["L"].long()) # L1은 semantic change detection label, 흑백처리 하는 게 필요 
        self.log_dict['l_cd'] = l_cd.item()

    # Get current log
    def get_current_log(self):
        return self.log_dict

    # Get current visuals
    def get_current_visuals(self, change, data):
        out_dict = OrderedDict()
        out_dict['pred_cm'] = torch.argmax(change, dim=1, keepdim=False) # batch, 256, 256 
        out_dict['gt_cm'] = data['L'] # batch, 256, 256  / 0~1 사이로 normalize 
        return out_dict

    # Printing the CD network
    def print_network(self):
        s, n = self.get_network_description(self.netCD)
        if isinstance(self.netCD, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netCD.__class__.__name__,
                                             self.netCD.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netCD.__class__.__name__)

        logger.info(
            'Change Detection Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    # Saving the network parameters
    def save_network(self, epoch, is_best_model = False):
        cd_gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'cd_model_E{}_gen.pth'.format(epoch))
        cd_opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'cd_model_E{}_opt.pth'.format(epoch))
        
        if is_best_model:
            best_cd_gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'best_cd_model_gen.pth'.format(epoch))
            best_cd_opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'best_cd_model_opt.pth'.format(epoch))

        # Save CD model pareamters
        network = self.netCD
        if isinstance(self.netCD, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, cd_gen_path)
        if is_best_model:
            torch.save(state_dict, best_cd_gen_path)


        # Save CD optimizer paramers
        opt_state = {'epoch': epoch,
                     'scheduler': None, 
                     'optimizer': None}
        opt_state['optimizer'] = self.optCD.state_dict()
        torch.save(opt_state, cd_opt_path)
        if is_best_model:
            torch.save(opt_state, best_cd_opt_path)

        # Print info
        logger.info(
            'Saved current CD model in [{:s}] ...'.format(cd_gen_path))
        if is_best_model:
            logger.info(
            'Saved best CD model in [{:s}] ...'.format(best_cd_gen_path))

    # Loading pre-trained CD network
    def load_network(self):
        load_path = self.opt['path_cd']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for CD model [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            
            # change detection model
            network = self.netCD
            if isinstance(self.netCD, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=True)
            
            if self.opt['phase'] == 'train':
                opt = torch.load(opt_path)
                self.optCD.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter'] if 'iter' in opt else 0 
                self.begin_epoch = opt['epoch'] if 'epoch' in opt else 0 
    
    # Functions related to computing performance metrics for CD
    def _update_metric(self, change, data):
        """
        update metric
        """
        G_pred = change.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=data['L'].detach().cpu().numpy())
        return current_score
    
    # Collecting status of the current running batch
    def _collect_running_batch_states(self, change, data):
        self.running_acc = self._update_metric(change, data)
        self.log_dict['running_acc'] = self.running_acc.item()

    # Collect the status of the epoch
    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.log_dict['epoch_acc'] = self.epoch_acc.item()

        for k, v in scores.items():
            self.log_dict[k] = v
            #message += '%s: %.5f ' % (k, v)

    # Rest all the performance metrics
    def _clear_cache(self):
        self.running_metric.clear()

    # Finctions related to learning rate sheduler
    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_netCD.step()

        
