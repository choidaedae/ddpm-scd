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
from models import ddpmscd

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
        
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
        
        elif phase == 'test' and args.phase == 'test':
            print("Creating [test] semantic-change-detection dataloader.")
            print(phase)
            test_set   = Data.create_scd_dataset(dataset_opt, phase)
            test_loader= Data.create_scd_dataloader(
                test_set, dataset_opt, phase)
            opt['len_test_dataloader'] = len(test_loader)
    
    logger.info('Initial Dataset Finished')

    # Loading diffusion model
    diffusion = Model.create_model(opt)
    logger.info('Initial Diffusion Model Finished')

    # Set noise schedule for the diffusion model
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
    Net = ddpmscd(opt)
    
    # Creating change-detection model
    change_detection = Model.create_CD_model(opt)
    logger.info('Initial Change Network Finished')
    
    # Creating semantic-segmentation model
    semantic_segmentation = Model.create_SS_model(opt)
    logger.info('Initial Pixel Classifier Finished')
    
    diffusion.netG.module.set_feature_extractor(False, opt['model_ss']['steps'], opt['model_ss']['blocks'])
    
    # pretrained model을 불러와서 추가적인 학습 (diffusion model은 원래 불러옴)
    

    params = {**semantic_segmentation.netSS.state_dict(), **change_detection.netCD.state_dict()}
        
    optimizer = torch.optim.SGD(params.values(), lr=opt['model_scd']['lr'], \
        weight_decay=opt['model_scd']['weight_decay'], momentum=opt['model_scd']['momentum'], nesterov=True)
    
    #################
    # Training loop #
    #################
    n_epoch = opt['train']['n_epoch']
    best_mF1 = 0.0
    start_epoch = 0
    for current_epoch in range(start_epoch, n_epoch):         
        change_detection._clear_cache()
        semantic_segmentation._clear_cache()
        
        steps_per_epoch = 0
        train_result_path = '{}/train/{}'.format(opt['path']
                                            ['results'], current_epoch)
        os.makedirs(train_result_path, exist_ok=True)
        
        ################
        ### training ###
        ################
        torch.cuda.empty_cache()
        message = 'lr: %0.7f\n \n' % change_detection.optCD.param_groups[0]['lr'] # SCD로 업데이트  
        logger.info(message)
        for current_step, train_data in enumerate(train_loader): # for change detection 
            # Feeding data to diffusion model and get features
            
            cd_params = {}
            ss_params = {}
            
            prev_cd_params = cd_params
            prev_ss_params = ss_params
                
            for name, param in change_detection.netCD.named_parameters():      
                cd_params[name] = param
                
            if (cd_params.items() != prev_cd_params.items()): print('changed')
        
            for name, param in semantic_segmentation.netSS.named_parameters():      
                ss_params[name] = param
                
            if (ss_params.items() != prev_ss_params.items()): print('changed')
        
            if (params.items() != {**ss_params, **cd_params}.items()): print('changed')
                
            steps_per_epoch += 1
            diffusion.feed_data(train_data)
 
            # for change detection 
            f_A=[] 
            f_B=[]
            for t in opt['model_cd']['t']:
                fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t) #np.random.randint(low=2, high=8)
                if opt['model_cd']['feat_type'] == "dec": # decoder의 feature map을 가져온다면
                    f_A.append(fd_A_t)
                    f_B.append(fd_B_t)
                else: # encoder의 feature map을 가져온다면 
                    f_A.append(fe_A_t)
                    f_B.append(fe_B_t)
                    del fd_A_t, fd_B_t 
            
            change_detection.feed_data(f_A, f_B, train_data) # 2중 list, 각 원소들은 bxcxhxw 형태의 feature map 
            change_detection.optimize_parameters() # parameter 업데이트 하는 부분 
            change_detection._collect_running_batch_states()
            
            f_A = diffusion.netG.module.feats_for_pc(train_data['A'])
            f_B = diffusion.netG.module.feats_for_pc(train_data['B'])
            
            semantic_segmentation.feed_data(f_A, f_B, train_data) # 마찬가지 
            semantic_segmentation.get_feature()
            semantic_segmentation.optimize_parameters()
            semantic_segmentation._collect_running_batch_states()
        
            # log running batch status
            if current_step % opt['train']['train_print_freq'] == 0:
                # message
                logs = change_detection.get_current_log()
                message = '[Training CD]. epoch: [%d/%d]. Iter: [%d/%d], CD_loss: %.5f, running_mf1: %.5f\n' %\
                (current_epoch, n_epoch-1, current_step, len(train_loader), logs['l_cd'], logs['running_acc'])
                logger.info(message)
                
                l_cd = logs['l_cd']

                #visuals
                visuals_cd = change_detection.get_current_visuals()

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
                logs = semantic_segmentation.get_current_log()
                
                message = '[Training SS]. epoch: [%d/%d]. Iter: [%d/%d], SS_loss: %.5f, running_mIoU: %.5f\n' %\
                (current_epoch, n_epoch-1, current_step, len(train_loader), logs['l_sem'], logs['running_mIoU'])
                logger.info(message)
            
                l_ss = logs['l_sem']

                #visuals
                semantic_segmentation.pred_ss_A, semantic_segmentation.pred_ss_B = semantic_segmentation.pixeltolabelMap() 
                semantic_segmentation.pred_ss_A, semantic_segmentation.pred_ss_B = semantic_segmentation.labelMaptosegMap() # 0 ~ 255 값        
                
                visuals_ss = semantic_segmentation.get_current_visuals() 
                
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
                                
                pred_scm_A = Metrics.tensor2img(pred_scm_A_tensor, out_type=np.uint8, min_max=(-1, 1))
                pred_scm_B = Metrics.tensor2img(pred_scm_B_tensor, out_type=np.uint8, min_max=(-1, 1))
                    
                Metrics.save_img(
                    pred_scm_A, '{}/pred_scm_A_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                Metrics.save_img(
                    pred_scm_B, '{}/pred_scm_B_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                            
                Metrics.save_img(
                    gt_scm_A, '{}/gt_scm_A_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                Metrics.save_img(
                    gt_scm_B, '{}/gt_scm_B_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))    
            
            loss = semantic_segmentation.l_sem + change_detection.l_cd
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        ### log epoch status ###
    
        change_detection._collect_epoch_states()
        logs = change_detection.get_current_log()
        message = '[Training CD (epoch summary)]: epoch: [%d/%d]. epoch_mF1=%.5f \n' %\
                (current_epoch, n_epoch-1, logs['epoch_acc'])
        for k, v in logs.items():
            message += '{:s}: {:.4e} '.format(k, v)
            tb_logger.add_scalar(k, v, current_step)
        message += '\n'
        logger.info(message)

        change_detection._clear_cache()
        change_detection._update_lr_schedulers()
        
        semantic_segmentation._collect_epoch_states(steps_per_epoch)
        logs = semantic_segmentation.get_current_log()
        message = '[Training SS (epoch summary)]: epoch: [%d/%d]. epoch_mIoU=%.5f \n' %\
                (current_epoch, n_epoch-1, logs['epoch_mIoU'])
        message = '[Training SCD (epoch summary)]: epoch: [%d/%d]. epoch_Fscd=%.5f. epoch_SeK=%5f \n' %\
                (current_epoch, n_epoch-1, logs['epoch_Fscd'], logs['epoch_SeK'])
        for k, v in logs.items():
            message += '{:s}: {:.4e} '.format(k, v)
            tb_logger.add_scalar(k, v, current_step)
        message += '\n'
        logger.info(message)

        semantic_segmentation._clear_cache()
        semantic_segmentation._update_lr_schedulers()
        
        
                
        if current_epoch % 10 == 0: 
            semantic_segmentation.save_network(current_epoch, is_best_model = True)
            change_detection.save_network(current_epoch, is_best_model = True)
            
        torch.cuda.empty_cache()  
            
    logger.info('End of training.')

    