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
import cv2
from models.cd.cd_head_diff import cd_head_diff 

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/SCD/ddpm_scd_train.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train'],
                        help='Run either train(training + validation) or testing', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
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

        elif phase == 'val' and args.phase != 'test':
            print("Creating [val] semantic-change-detection dataloader.")
            val_set   = Data.create_scd_dataset(dataset_opt, phase)
            val_loader= Data.create_scd_dataloader(
                val_set, dataset_opt, phase)
            opt['len_val_dataloader'] = len(val_loader)
        
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
    
    # Creating change-detection model
    change_detection = Model.create_CD_model(opt)
    logger.info('Initial Change Network Finished')
    
    # Creating semantic-segmentation model
    semantic_segmentation = Model.create_SS_model(opt)
    logger.info('Initial Pixel Classifier Finished')
    
    # pretrained model을 불러와서 추가적인 학습
    if args['path_cd']['load_model']:
        change_detection.netCD.load_state_dict(torch.load(opt['path_cd']['resume_state']))
    if args['path_ss']['load_ss']:
        semantic_segmentation.netSS.load_state_dict(torch.load(opt['path_ss']['resume_state']))
    
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
        message = 'lr: %0.7f\n \n' % change_detection.optCD.param_groups[0]['lr'] # SCD로 업데이트  
        logger.info(message)
        for current_step, train_data in enumerate(train_loader): # for change detection 
            # Feeding data to diffusion model and get features
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
            
            # for semantic segmentation 
            for t in opt['model_ss']['t']:
                fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t) #np.random.randint(low=2, high=8)
                if opt['model_ss']['feat_type'] == "dec": # decoder의 feature map을 가져온다면
                    f_A.append(fd_A_t)
                    f_B.append(fd_B_t)
                else: # encoder의 feature map을 가져온다면 
                    f_A.append(fe_A_t)
                    f_B.append(fe_B_t)
                    del fd_A_t, fd_B_t
            
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
            
                l_scd = 0
                loss = l_ss + l_cd + l_scd
                loss.backward()
                optimizer.step()
                
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
        
        ##################
        ### validation ###
        ##################
        if current_epoch % opt['train']['val_freq'] == 0:
            val_result_path = '{}/val/{}'.format(opt['path']
                                            ['results'], current_epoch)
            os.makedirs(val_result_path, exist_ok=True)
            
            for current_epoch in range(start_epoch, n_epoch):         
                change_detection._clear_cache()
                val_result_path = '{}/val/{}'.format(opt['path']
                                                    ['results'], current_epoch)
                os.makedirs(val_result_path, exist_ok=True)
                message = 'lr: %0.7f\n \n' % change_detection.optCD.param_groups[0]['lr'] # SCD로 업데이트  
                logger.info(message)
                
                for current_step, val_data in enumerate(val_loader):
                # Feed data to diffusion model
                    diffusion.feed_data(val_data)

                    # for change detection 
                    f_A=[] 
                    f_B=[]
                    for t in opt['model_cd']['t']:
                        fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t) #np.random.randint(low=2, high=8)
                        if opt['model_cd']['feat_type'] == "dec": 
                            f_A.append(fd_A_t)
                            f_B.append(fd_B_t)
                        else: 
                            f_A.append(fe_A_t)
                            f_B.append(fe_B_t)
                            del fd_A_t, fd_B_t
                    
                    # Feed data to CD model
                    change_detection.feed_data(f_A, f_B, val_data)
                    change_detection.test()
                    change_detection._collect_running_batch_states()
                    
                    for t in opt['model_ss']['t']:
                        fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t) #np.random.randint(low=2, high=8)
                        if opt['model_ss']['feat_type'] == "dec": # decoder의 feature map을 가져온다면
                            f_A.append(fd_A_t)
                            f_B.append(fd_B_t)
                        else: # encoder의 feature map을 가져온다면 
                            f_A.append(fe_A_t)
                            f_B.append(fe_B_t)
                            del fd_A_t, fd_B_t
                            
                    semantic_segmentation.feed_data(f_A, f_B, val_data)
                    semantic_segmentation.get_feature()
                    semantic_segmentation.optimize_parameters()
                    semantic_segmentation._collect_running_batch_states()
                    
                    # log running batch status for val data
                    if current_step % opt['train']['val_print_freq'] == 0:
                        logs        = change_detection.get_current_log()
                        message = '[Validation CD]. epoch: [%d/%d]. Iter: [%d/%d], CD_loss: %.5f, running_mf1: %.5f\n' %\
                        (current_epoch, n_epoch-1, current_step, len(val_loader), logs['l_cd'], logs['running_acc'])
                        logger.info(message)
                        
                        #visuals
                        visuals_cd = change_detection.get_current_visuals()

                        # Converting to uint8
                        img_A   = Metrics.tensor2img(val_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        img_B   = Metrics.tensor2img(val_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        
                        gt_cm   = Metrics.tensor2img(visuals_cd['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
                        pred_cm = Metrics.tensor2img(visuals_cd['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
                    
                        #save imgs
                        Metrics.save_img(
                            img_A, '{}/img_A_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                        Metrics.save_img(
                            img_B, '{}/img_B_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                        
                        Metrics.save_img(
                            pred_cm, '{}/pred_cm_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                        Metrics.save_img(
                            gt_cm, '{}/gt_cm_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                        
                        
                        # for semantic segmentation
                        logs = semantic_segmentation.get_current_log()
                        message = '[Validating SS]. epoch: [%d/%d]. Iter: [%d/%d], SS_loss: %.5f, running_mIoU: %.5f\n' %\
                        (current_epoch, n_epoch-1, current_step, len(val_loader), logs['l_sem'], logs['running_mIoU'])
                        logger.info(message)

                        #visuals
                        semantic_segmentation.pred_ss_A, semantic_segmentation.pred_ss_B = semantic_segmentation.pixeltolabelMap() 
                        semantic_segmentation.pred_ss_A, semantic_segmentation.pred_ss_B = semantic_segmentation.labelMaptosegMap() # 0 ~ 255 값        
                        
                        visuals_ss = semantic_segmentation.get_current_visuals() 
                    
                        visuals_ss['pred_ss_A'] = visuals_ss['pred_ss_A'] *2.0-1.0 # b, 256, 256, 0~1 로 normalize되어있음 
                        visuals_ss['pred_ss_B'] = visuals_ss['pred_ss_B'] *2.0-1.0 # b, 256, 256 
                        
                        val_data['L1'] = val_data['L1']*2.0-1.0
                        val_data['L2'] = val_data['L2']*2.0-1.0
                        
                        pred_ss_A = Metrics.tensor2img(visuals_ss['pred_ss_A'], out_type=np.uint8, min_max=(-1, 1)) # uint8
                        pred_ss_B = Metrics.tensor2img(visuals_ss['pred_ss_B'], out_type=np.uint8, min_max=(-1, 1)) # uint8
                            
                        gt_scm_A = Metrics.tensor2img(val_data['L1'], out_type=np.uint8, min_max=(-1, 1))
                        gt_scm_B = Metrics.tensor2img(val_data['L2'], out_type=np.uint8, min_max=(-1, 1))
                        
                        Metrics.save_img(
                            pred_ss_A, '{}/pred_ss_A_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                        Metrics.save_img(
                            pred_ss_B, '{}/pred_ss_B_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                            
                        pred_scm_A_tensor, pred_scm_B_tensor = Metrics.make_SCD_map(pred_ss_A, pred_ss_B, pred_cm)
                                        
                        pred_scm_A = Metrics.tensor2img(pred_scm_A_tensor, out_type=np.uint8, min_max=(-1, 1))
                        pred_scm_B = Metrics.tensor2img(pred_scm_B_tensor, out_type=np.uint8, min_max=(-1, 1))
                            
                        Metrics.save_img(
                            pred_scm_A, '{}/pred_scm_A_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                        Metrics.save_img(
                            pred_scm_B, '{}/pred_scm_B_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                                    
                        Metrics.save_img(
                            gt_scm_A, '{}/gt_scm_A_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                        Metrics.save_img(
                            gt_scm_B, '{}/gt_scm_B_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                    
            change_detection._collect_epoch_states()
            logs = change_detection.get_current_log()
            message = '[Validating CD (epoch summary)]: epoch: [%d/%d]. epoch_mF1=%.5f \n' %\
                (current_epoch, n_epoch-1, logs['epoch_acc'])
            for k, v in logs.items():
                message += '{:s}: {:.4e} '.format(k, v)
                tb_logger.add_scalar(k, v, current_step)
            message += '\n'
            logger.info(message)

            change_detection._clear_cache()
            change_detection._update_lr_schedulers()
        
            semantic_segmentation._collect_epoch_states()
            logs = semantic_segmentation.get_current_log()
            message = '[Validating SS (epoch summary)]: epoch: [%d/%d]. epoch_mIoU=%.5f \n' %\
                (current_epoch, n_epoch-1, logs['epoch_mIoU'])
            message += '[Validating SCD (epoch summary)]: epoch: [%d/%d]. epoch_Fscd=%.5f. epoch_SeK=%.5f. \n' %\
                (current_epoch, n_epoch-1, logs['epoch_mFscd'], logs['epoch_SeK'])
            for k, v in logs.items():
                message += '{:s}: {:.4e} '.format(k, v)
                tb_logger.add_scalar(k, v, current_step)
            message += '\n'
            logger.info(message)
            
            semantic_segmentation._clear_cache()
            semantic_segmentation._update_lr_schedulers()
            
            if logs['epoch_acc'] > best_mF1: 
                cd_best_model = True
                best_mF1 = logs['epoch_acc']
                logger.info('[Validation CD] Best model updated. Saving the models (current + best) and training states.')
            else:
                cd_best_model = False
                logger.info('[Validation CD] Saving the current cd model and training states.')
            logger.info('--- Proceed To The Next Epoch ----\n \n')

            change_detection.save_network(current_epoch, is_best_model = cd_best_model) 
            change_detection._clear_cache()
            
            if logs['epoch_mIoU'] > best_mIoU: 
                seg_best_model = True
                best_mIoU = logs['epoch_mIoU']
                logger.info('[Validation SS] Best model updated. Saving the models (current + best) and training states.')
            else:
                seg_best_model = False
                logger.info('[Validation SS] Saving the current seg model and training states.')
            logger.info('--- Proceed To The Next Epoch ----\n \n')
            
            semantic_segmentation.save_network(current_epoch, is_best_model = seg_best_model)
            semantic_segmentation._clear_cache()   
            
            if logs['epoch_Fscd'] > best_Fscd:
                scd_best_model = True
                best_Fscd = logs['epoch_Fscd']
                logger.info('[Validation SCD] Best model updated. Saving the models (current + best) and training states.')
            else:
                seg_best_model = False
                logger.info('[Validation SCD] Saving the current whole model and training states.')
            logger.info('--- Proceed To The Next Epoch ----\n \n')
            
    logger.info('End of training.')
    print(f'Training is done. The best F_scd score is = {best_Fscd}.')
    
