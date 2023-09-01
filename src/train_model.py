# code for train & validate semantic change detection model
# compute loss function and semantic change detection metric

import torch
import data as Data
import models as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from tensorboardX import SummaryWriter
import os
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist 
from models.ddpmscd import ddpmscd as ddpmscd
from src.feature_extractors import create_feature_extractor, collect_features

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

def train(opt, Net, diffusion, optimizer, lr_scheduler, train_loader, logger):

    feature_extractor = create_feature_extractor(**opt['segmentation'])
    
    # n_epoch = opt['train']['n_epoch']
    n_epoch = 500
    start_epoch = 0
    
    cd = Net.module.change_detection
    ss = Net.module.semantic_segmentation
    
    for current_epoch in range(start_epoch, n_epoch):   
           
        cd._clear_cache()
        ss._clear_cache()
        
        steps_per_epoch = 0
        train_result_path = '{}/train/{}'.format(opt['path']
                                            ['results'], current_epoch)
        os.makedirs(train_result_path, exist_ok=True)
        torch.cuda.empty_cache()
        for current_step, train_data in enumerate(train_loader):
            steps_per_epoch += 1
            diffusion.feed_data(train_data)

            # for change detection 
            fcd_A=[] 
            fcd_B=[]
            for t in opt['model_cd']['t']:
                fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t) 
                if opt['model_cd']['feat_type'] == "dec": #
                    fcd_A.append(fd_A_t)
                    fcd_B.append(fd_B_t)
                else:
                    fcd_A.append(fe_A_t)
                    fcd_B.append(fe_B_t)
                    del fd_A_t, fd_B_t 
            
            cd.feed_data(fcd_A, fcd_B, train_data) 
            
            fss_A = feature_extractor(train_data['A'], noise = None)
            fss_B = feature_extractor(train_data['B'], noise = None)
            
            fss_A = collect_features(opt['model_ss'], fss_A)
            fss_B = collect_features(opt['model_ss'], fss_B)

            x1 = fss_A.view(opt['model_ss']['dim'][-1], -1).permute(1, 0).unsqueeze(dim = 0)
            x2 = fss_B.view(opt['model_ss']['dim'][-1], -1).permute(1, 0).unsqueeze(dim = 0)

                
            ss.feed_data(x1, x2, train_data) # 마찬가지
      
            seg_A, seg_B, change = Net(ss.fss_A, ss.fss_B, cd.fcd_A, cd.fcd_B)
            
            loss = Net.module.optimize_parameters(seg_A, seg_B, change, train_data)
            Net.module.collect_running_batch_states(seg_A, seg_B, change, train_data)
        
            # log running batch status
            # if current_step % opt['train']['train_print_freq'] == 0:
            if current_step % 100 == 0:
                # message
                logs = cd.get_current_log()
                message = '[Training CD]. epoch: [%d/%d]. Iter: [%d/%d], CD_loss: %.5f, running_mf1: %.5f\n' %\
                (current_epoch, n_epoch-1, current_step, len(train_loader), logs['l_cd'], logs['running_acc'])
                logger.info(message)
                
                #visuals
                visuals_cd = cd.get_current_visuals(change, train_data)

                # Converting to uint8
                img_A   = Metrics.tensor2img(train_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                img_B   = Metrics.tensor2img(train_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                
                gt_cm   = Metrics.tensor2img(visuals_cd['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
                pred_cm = Metrics.tensor2img(visuals_cd['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
            
                #save imgs
                Metrics.save_img(
                    img_A, '{}/img_A_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                Metrics.save_img(
                    img_B, '{}/img_B_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                
                Metrics.save_img(
                    pred_cm, '{}/pred_cm_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                Metrics.save_img(
                    gt_cm, '{}/gt_cm_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                
                
                # for semantic segmentation
                logs = ss.get_current_log()
                message = '[Training SS]. epoch: [%d/%d]. Iter: [%d/%d], SS_loss: %.5f, running_mIoU: %.5f\n' %\
                (current_epoch, n_epoch-1, current_step, len(train_loader), logs['l_sem'], logs['running_mIoU'])
                logger.info(message)
            
                #visuals
                segMap_A, segMap_B = ss.pixeltosegMap(seg_A, seg_B) 
                
                visuals_ss = ss.get_current_visuals(segMap_A, segMap_B, train_data) 
                
                visuals_ss['pred_ss_A'] = visuals_ss['pred_ss_A'] *2.0-1.0 # b, 3, 256, 256, -1~1 로 normalize되어있음 
                visuals_ss['pred_ss_B'] = visuals_ss['pred_ss_B'] *2.0-1.0 # b, 3, 256, 256 
                
                train_data['L1'] = train_data['L1']*2.0-1.0
                train_data['L2'] = train_data['L2']*2.0-1.0
                
                pred_ss_A = Metrics.tensor2img(visuals_ss['pred_ss_A'], out_type=np.uint8, min_max=(-1, 1)) # uint8
                pred_ss_B = Metrics.tensor2img(visuals_ss['pred_ss_B'], out_type=np.uint8, min_max=(-1, 1)) # uint8
                    
                gt_scm_A = Metrics.tensor2img(train_data['L1'], out_type=np.uint8, min_max=(-1, 1))
                gt_scm_B = Metrics.tensor2img(train_data['L2'], out_type=np.uint8, min_max=(-1, 1))
                
                Metrics.save_img(
                    pred_ss_A, '{}/pred_ss_A_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                Metrics.save_img(
                    pred_ss_B, '{}/pred_ss_B_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                    
                pred_scm_A_tensor, pred_scm_B_tensor = Metrics.make_SCD_map(pred_ss_A, pred_ss_B, pred_cm)
                                
                pred_scm_A = Metrics.tensor2img(pred_scm_A_tensor, out_type=np.uint8, min_max=(0, 255))
                pred_scm_B = Metrics.tensor2img(pred_scm_B_tensor, out_type=np.uint8, min_max=(0, 255))
                    
                Metrics.save_img(
                    pred_scm_A, '{}/pred_scm_A_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                Metrics.save_img(
                    pred_scm_B, '{}/pred_scm_B_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                            
                Metrics.save_img(
                    gt_scm_A, '{}/gt_scm_A_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                Metrics.save_img(
                    gt_scm_B, '{}/gt_scm_B_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))    
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        lr_scheduler.step()
        
        ### log epoch status ###
    
        cd._collect_epoch_states()
        logs = cd.get_current_log()
        message = '[Training CD (epoch summary)]: epoch: [%d/%d]. epoch_mF1=%.5f \n' %\
                (current_epoch, n_epoch-1, logs['epoch_acc'])
        for k, v in logs.items():
            message += '{:s}: {:.4e} '.format(k, v)
            tb_logger.add_scalar(k, v, current_step)
        message += '\n'
        logger.info(message)
        cd._clear_cache()
        
        ss._collect_epoch_states(steps_per_epoch)
        logs = ss.get_current_log()
        message = '[Training SS (epoch summary)]: epoch: [%d/%d]. epoch_mIoU=%.5f \n' %\
                (current_epoch, n_epoch-1, logs['epoch_mIoU'])
        message = '[Training SCD (epoch summary)]: epoch: [%d/%d]. epoch_Fscd=%.5f. epoch_SeK=%5f \n' %\
                (current_epoch, n_epoch-1, logs['epoch_Fscd'], logs['epoch_SeK'])
        for k, v in logs.items():
            message += '{:s}: {:.4e} '.format(k, v)
            tb_logger.add_scalar(k, v, current_step)
        message += '\n'
        logger.info(message)
        ss._clear_cache()
        
        if current_epoch % 10 == 0: 
            ss.save_network(current_epoch, is_best_model = True)
            cd.save_network(current_epoch, is_best_model = True)
            
        torch.cuda.empty_cache()  
    logger.info('End of training.')
    
    
def validation(opt, Net, diffusion, validation_loader, logger):
    # TBD... 
    return 

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/SCD/ddpm_scd_train.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train'],
                        help='Run either train(training + validation) or testing', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None
                        )
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])


    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'test':
            print("Creating [train] semantic-change-detection dataloader.")
            train_set   = Data.create_scd_dataset(dataset_opt, phase)
            train_loader= Data.create_scd_dataloader(
                train_set, dataset_opt, phase)
            opt['len_train_dataloader'] = len(train_loader)
        
    logger.info('Initial Dataset Finished')

    # Loading diffusion model
    diffusion = Model.create_model(opt)
    logger.info('Initial Diffusion Model Finished')

    # Set noise schedule for the diffusion model
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
    # Creating Model 
    Net = torch.nn.DataParallel(ddpmscd(opt), device_ids=opt['gpu_ids'])
    
    # Creating feature extractor 
    diffusion.netG.module.set_feature_extractor(False, opt['model_ss']['steps'], opt['model_ss']['blocks'])

    optimizer = torch.optim.SGD(Net.parameters(), lr=opt['model_scd']['lr'], \
        weight_decay=opt['model_scd']['weight_decay'], momentum=opt['model_scd']['momentum'], nesterov=True)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)
    
    train(opt, Net, diffusion, optimizer, lr_scheduler, train_loader, logger)
    
   