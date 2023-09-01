# code for test & evaluate semantic change detection 

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
    parser.add_argument('-c', '--config', type=str, default='config/SCD/ddpm_scd.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['test'],
                        help='Run either train(training + validation) or testing', default='test')
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


    # Loading change-detction datasets. -> semantic change detection으로 바꿔야 함. 
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
    
    #################
    # Training loop #
    #################
    n_epoch = opt['train']['n_epoch']
    best_mF1 = 0.0
    start_epoch = 0
    if opt['phase'] == 'train':
        for current_epoch in range(start_epoch, n_epoch):         
            change_detection._clear_cache()
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
                diffusion.feed_data(train_data)
                # diffusion model에서 특정 feature map 담아와서 change detection 하는 데 사용. 
                
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
                
                change_detection.feed_data(f_A, f_B, train_data)
                change_detection.optimize_parameters()
                change_detection._collect_running_batch_states()

                # log running batch status
                if current_step % opt['train']['train_print_freq'] == 0:
                    # message
                    logs = change_detection.get_current_log()
                    message = '[Training CD]. epoch: [%d/%d]. Iter: [%d/%d], CD_loss: %.5f, running_mf1: %.5f\n' %\
                      (current_epoch, n_epoch-1, current_step, len(train_loader), logs['l_cd'], logs['running_acc'])
                    logger.info(message)

                    #visuals
                    visuals = change_detection.get_current_visuals()

                    img_mode = "single"
                    if img_mode == "single":
                        # Converting to uint8
                        img_A   = Metrics.tensor2img(train_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        img_B   = Metrics.tensor2img(train_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        
                        gt_cm   = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
                        pred_cm = Metrics.tensor2img(visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
                    
                        #save imgs
                        Metrics.save_img(
                            img_A, '{}/img_A_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        Metrics.save_img(
                            img_B, '{}/img_B_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        
                        Metrics.save_img(
                            pred_cm, '{}/img_pred_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        Metrics.save_img(
                            gt_cm, '{}/img_gt_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                            
                    '''
                    else:
                        # grid img
                        visuals['pred_cm'] = visuals['pred_cm']*2.0-1.0
                        visuals['gt_cm'] = visuals['gt_cm']*2.0-1.0
                        grid_img = torch.cat((  train_data['A'], 
                                    train_data['B'], 
                                    visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), 
                                    visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                    dim = 0)
                        grid_img = Metrics.tensor2img(grid_img)  # uint8
                        Metrics.save_img(
                            grid_img, '{}/img_A_B_pred_gt_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                    
                    '''
                    
                # for semantic segmentation
                f_A=[] 
                f_B=[]
                for t in opt['model_ss']['t']:
                    fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t) #np.random.randint(low=2, high=8)
                    if opt['model_ss']['feat_type'] == "dec": # decoder의 feature map을 가져온다면
                        f_A.append(fd_A_t)
                        f_B.append(fd_B_t)
                    else: # encoder의 feature map을 가져온다면 
                        f_A.append(fe_A_t)
                        f_B.append(fe_B_t)
                        del fd_A_t, fd_B_t
                
                semantic_segmentation.feed_data(f_A, f_B, train_data)
                semantic_segmentation.optimize_parameters()
                semantic_segmentation._collect_running_batch_states()
                
                if current_step % opt['train']['train_print_freq'] == 0:
                    # message
                    logs = semantic_segmentation.get_current_log()
                    message = '[Training SS]. epoch: [%d/%d]. Iter: [%d/%d], SS_loss: %.5f, running_mIoU: %.5f\n' %\
                      (current_epoch, n_epoch-1, current_step, len(train_loader), logs['l_sem'], logs['running_acc'])
                    logger.info(message)

                    #visuals
                    visuals_ss = semantic_segmentation.get_current_visuals()
                    
                    if img_mode == "single":
                        # Converting to uint8
                        
                        visuals_ss['pred_ss_A'] = visuals_ss['pred_ss_A']*2.0-1.0
                        visuals_ss['pred_ss_B'] = visuals_ss['pred_ss_B']*2.0-1.0
                        pred_ss_A = Metrics.tensor2img(visuals_ss['pred_ss_A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        pred_ss_B = Metrics.tensor2img(visuals_ss['pred_ss_B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                         
                        gt_scm_A = Metrics.tensor2img(train_data['L1'], out_type=np.uint8, min_max=(-1, 1))
                        gt_scm_B = Metrics.tensor2img(train_data['L2'], out_type=np.uint8, min_max=(-1, 1))
                        
                        #save results (semantic segmentation & semantic change map)
                    
                        Metrics.save_img(
                            pred_ss_A, '{}/pred_ss_A_{}.png'.format(train_result_path, current_step))
                        Metrics.save_img(
                            pred_ss_B, '{}/pred_ss_B_{}.png'.format(train_result_path, current_step))
                        
                        pred_scm_tensors = Metrics.make_SCD_map(pred_ss_A, pred_ss_B, pred_cm)
                            
                        pred_scm_A_tensor = pred_scm_tensors[0], pred_scm_B_tensor = pred_scm_tensors[1] 
                            
                        pred_scm_A = Metrics.tensor2img(pred_scm_A_tensor, out_type=np.uint8, min_max=(-1, 1))
                        pred_scm_B = Metrics.tensor2img(pred_scm_B_tensor, out_type=np.uint8, min_max=(-1, 1))
                    
                        Metrics.save_img(
                            pred_scm_A, '{}/pred_scm_A_{}.png'.format(train_result_path, current_step))
                        Metrics.save_img(
                            pred_scm_B, '{}/pred_scm_B_{}.png'.format(train_result_path, current_step))
                        
                        Metrics.save_img(
                            gt_scm_A, '{}/gt_scm_A_{}.png'.format(train_result_path, current_step))
                        Metrics.save_img(
                            gt_scm_B, '{}/gt_scm_B_{}.png'.format(train_result_path, current_step))
                        
                    '''
                    else:
                        # grid img
                        visuals['pred_cm'] = visuals['pred_cm']*2.0-1.0
                        visuals['gt_cm'] = visuals['gt_cm']*2.0-1.0
                        grid_img = torch.cat((  train_data['A'], 
                                    train_data['B'], 
                                    visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), 
                                    visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                    dim = 0)
                        grid_img = Metrics.tensor2img(grid_img)  # uint8
                        Metrics.save_img(
                            grid_img, '{}/img_A_B_pred_gt_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                    '''
                    
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
            
            semantic_segmentation._collect_epoch_states()
            logs = semantic_segmentation.get_current_log()
            message = '[Training SS (epoch summary)]: epoch: [%d/%d]. epoch_mIoU=%.5f \n' %\
                      (current_epoch, n_epoch-1, logs['epoch_acc'])
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

                for current_step, val_data in enumerate(val_loader):
                    # Feed data to diffusion model
                    diffusion.feed_data(val_data)
                    f_A=[]
                    f_B=[]
                    for t in opt['model_cd']['t']:
                        fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t) #np.random.randint(low=2, high=8)
                        if opt['model_cd']['feat_type'] == "dec":
                            f_A.append(fd_A_t)
                            f_B.append(fd_B_t)
                            del fe_A_t, fe_B_t
                        else:
                            f_A.append(fe_A_t)
                            f_B.append(fe_B_t)
                            del fd_A_t, fd_B_t

                    # Feed data to CD model
                    change_detection.feed_data(f_A, f_B, val_data)
                    change_detection.test()
                    change_detection._collect_running_batch_states()
                    
                    # log running batch status for val data
                    if current_step % opt['train']['val_print_freq'] == 0:
                        # message
                        logs        = change_detection.get_current_log()
                        message     = '[Validation CD]. epoch: [%d/%d]. Iter: [%d/%d], running_mf1: %.5f\n' %\
                                    (current_epoch, n_epoch-1, current_step, len(val_loader), logs['running_acc'])
                        logger.info(message)

                        #visuals
                        visuals_cd = change_detection.get_current_visuals()

                        img_mode = "grid"
                        if img_mode == "single":
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
                                pred_cm, '{}/img_pred_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                            Metrics.save_img(
                                gt_cm, '{}/img_gt_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                        
                        '''
                        else:
                            # grid img
                            visuals['pred_cm'] = visuals['pred_cm']*2.0-1.0
                            visuals['gt_cm'] = visuals['gt_cm']*2.0-1.0
                            grid_img = torch.cat((  val_data['A'], 
                                        val_data['B'], 
                                        visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), 
                                        visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                        dim = 0)
                            grid_img = Metrics.tensor2img(grid_img)  # uint8
                            Metrics.save_img(
                            grid_img, '{}/img_A_B_pred_gt_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                        '''
                    
                    # for semantic segmentation & semantic change detection 
                    f_A=[]
                    f_B=[]
                    for t in opt['model_ss']['t']:
                        fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t) #np.random.randint(low=2, high=8)
                        if opt['model_ss']['feat_type'] == "dec":
                            f_A.append(fd_A_t)
                            f_B.append(fd_B_t)
                            del fe_A_t, fe_B_t
                        else:
                            f_A.append(fe_A_t)
                            f_B.append(fe_B_t)
                            del fd_A_t, fd_B_t

                    # Feed data to SS model
                    semantic_segmentation.feed_data(f_A, f_B, val_data)
                    semantic_segmentation.test()
                    semantic_segmentation._collect_running_batch_states()
                    
                    # log running batch status for val data
                    if current_step % opt['train']['val_print_freq'] == 0:
                        # message
                        logs        = semantic_segmentation.get_current_log()
                        message     = '[Validation SS]. epoch: [%d/%d]. Iter: [%d/%d], running_IoU: %.5f\n' %\
                                    (current_epoch, n_epoch-1, current_step, len(val_loader), logs['running_acc'])
                        logger.info(message)

                        #visuals
                        visuals_ss = semantic_segmentation.get_current_visuals()

                        img_mode = "grid"
                        if img_mode == "single":
                            # Converting to uint8
                            visuals_ss['pred_ss_A'] = visuals_ss['pred_ss_A']*2.0-1.0
                            visuals_ss['pred_ss_B'] = visuals_ss['pred_ss_B']*2.0-1.0
                            pred_ss_A = Metrics.tensor2img(visuals_ss['pred_ss_A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                            pred_ss_B = Metrics.tensor2img(visuals_ss['pred_ss_B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                         
                            gt_scm_A = Metrics.tensor2img(val_data['L1'], out_type=np.uint8, min_max=(-1, 1))
                            gt_scm_B = Metrics.tensor2img(val_data['L2'], out_type=np.uint8, min_max=(-1, 1))
                            
                            #save imgs
                            Metrics.save_img(
                                pred_ss_A, '{}/pred_ss_A_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                            Metrics.save_img(
                                pred_ss_B, '{}/pred_ss_B_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                            
                            pred_scm_tensors = Metrics.make_SCD_map(pred_ss_A, pred_ss_B, pred_cm)
                            
                            pred_scm_A_tensor = pred_scm_tensors[0], pred_scm_B_tensor = pred_scm_tensors[1] 
                            
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
                      (current_epoch, n_epoch-1, logs['epoch_acc'])
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    tb_logger.add_scalar(k, v, current_step)
                message += '\n'
                logger.info(message)

                semantic_segmentation._clear_cache()
                semantic_segmentation._update_lr_schedulers()
                
                if logs['epoch_acc'] > best_mF1: # mFSCD로 바꿔야 함 
                    is_best_model = True
                    best_mF1 = logs['epoch_acc']
                    logger.info('[Validation CD] Best model updated. Saving the models (current + best) and training states.')
                else:
                    is_best_model = False
                    logger.info('[Validation CD]Saving the current cd model and training states.')
                logger.info('--- Proceed To The Next Epoch ----\n \n')

                change_detection.save_network(current_epoch, is_best_model = is_best_model) #
                change_detection._clear_cache()
                
                semantic_segmentation.save_network(current_epoch, is_best_model = is_best_model)
                semantic_segmentation._clear_cache()   
                
        logger.info('End of training.')
        
    else: # evaluation, if opt == 'test': 
        logger.info('Begin Model Evaluation (testing).')
        test_result_path = '{}/test/'.format(opt['path']
                                                 ['results'])
        os.makedirs(test_result_path, exist_ok=True)
        logger_test = logging.getLogger('test')  # test logger
        change_detection._clear_cache()
        semantic_segmentation._clear_cache()
        for current_step, test_data in enumerate(test_loader): # 이 때 test_data는 semantic change detection 용도의 data. segmentation과 change_detectio에 같은 데이터셋을 씀 
            
            # for change detection 
            # Feed data to diffusion model
            diffusion.feed_data(test_data) # test data를 diffusion에게 전달해주고, feature map을 뽑아옴. 
            f_A=[]
            f_B=[]
            for t in opt['model_cd']['t']:
                fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t) #np.random.randint(low=2, high=8)
                if opt['model_cd']['feat_type'] == "dec":
                    f_A.append(fd_A_t)
                    f_B.append(fd_B_t)
                    del fe_A_t, fe_B_t
                else:
                    f_A.append(fe_A_t)
                    f_B.append(fe_B_t)
                    del fd_A_t, fd_B_t
            
            # feature 가져옴 
            
            # pixel_classifier test 부분 추가 

            # Feed data to CD model 
            change_detection.feed_data(f_A, f_B, test_data) # change detection에 feature 쌍으로 넣어줌. 
            change_detection.test()
            change_detection._collect_running_batch_states()
            
            # Logs
            logs        = change_detection.get_current_log()
            message     = '[Testing CD]. Iter: [%d/%d], running_mf1: %.5f\n' %\
                                    (current_step, len(test_loader), logs['running_acc'])
            logger_test.info(message)

            # Visuals
            visuals_cd = change_detection.get_current_visuals()
    
            img_mode = 'single'
            
            if img_mode == 'single':
                # Converting to uint8
                visuals_cd['pred_cm'] = visuals_cd['pred_cm']*2.0-1.0
                visuals_cd['gt_cm'] = visuals_cd['gt_cm']*2.0-1.0
                img_A   = Metrics.tensor2img(test_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                img_B   = Metrics.tensor2img(test_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                gt_cm   = Metrics.tensor2img(visuals_cd['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
                pred_cm = Metrics.tensor2img(visuals_cd['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8

                # Save imgs
                Metrics.save_img(
                    img_A, '{}/img_A_{}.png'.format(test_result_path, current_step))
                Metrics.save_img(
                    img_B, '{}/img_B_{}.png'.format(test_result_path, current_step))
                Metrics.save_img(
                    pred_cm, '{}/img_pred_cm{}.png'.format(test_result_path, current_step))
                Metrics.save_img(
                    gt_cm, '{}/img_gt_cm{}.png'.format(test_result_path, current_step))
            '''
            else:
                # grid img
                visuals['pred_cm'] = visuals['pred_cm']*2.0-1.0
                visuals['gt_cm'] = visuals['gt_cm']*2.0-1.0
                grid_img = torch.cat((  test_data['A'], 
                                        test_data['B'], 
                                        visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), 
                                        visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                        dim = 0)
                grid_img = Metrics.tensor2img(grid_img)  # uint8
                Metrics.save_img(
                    grid_img, '{}/img_A_B_pred_gt_{}.png'.format(test_result_path, current_step))
            '''
            
            f_A=[] 
            f_B=[]
            for t in opt['model_ss']['t']:
                fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t) #np.random.randint(low=2, high=8)
                if opt['model_ss']['feat_type'] == "dec": # decoder의 feature map을 가져온다면
                    f_A.append(fd_A_t)
                    f_B.append(fd_B_t)
                else: # encoder의 feature map을 가져온다면 
                    f_A.append(fe_A_t)
                    f_B.append(fe_B_t)
                    del fd_A_t, fd_B_t
                    
            semantic_segmentation.feed_data(f_A, f_B, test_data)
            semantic_segmentation.test(3, 256)
            semantic_segmentation._collect_running_batch_states()
            
            # Logs
            logs        = semantic_segmentation.get_current_log()
            message     = '[Testing SS]. Iter: [%d/%d], running_mIoU: %.5f\n' %\
                                    (current_step, len(test_loader), logs['running_acc'])
            logger_test.info(message)
                    
            visuals_ss = semantic_segmentation.get_current_visuals()
    
            
            if img_mode == 'single':
                # Converting to uint8
                visuals_ss['pred_ss_A'] = visuals_ss['pred_ss_A']*2.0-1.0
                visuals_ss['pred_ss_B'] = visuals_ss['pred_ss_B']*2.0-1.0
                pred_ss_A = Metrics.tensor2img(visuals_ss['pred_ss_A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                pred_ss_B = Metrics.tensor2img(visuals_ss['pred_ss_B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                
                gt_scm_A = Metrics.tensor2img(test_data['L1'], out_type=np.uint8, min_max=(-1, 1))
                gt_scm_B = Metrics.tensor2img(test_data['L2'], out_type=np.uint8, min_max=(-1, 1))
                
                # Save imgs
                Metrics.save_img(
                    pred_ss_A, '{}/pred_ss_A_{}.png'.format(test_result_path, current_step))
                Metrics.save_img(
                    pred_ss_B, '{}/pred_ss_B_{}.png'.format(test_result_path, current_step))
                
                pred_scm_tensors = Metrics.make_SCD_map(pred_ss_A, pred_ss_B, pred_cm)
                            
                pred_scm_A_tensor = pred_scm_tensors[0], pred_scm_B_tensor = pred_scm_tensors[1] 
                            
                pred_scm_A = Metrics.tensor2img(pred_scm_A_tensor, out_type=np.uint8, min_max=(-1, 1))
                pred_scm_B = Metrics.tensor2img(pred_scm_B_tensor, out_type=np.uint8, min_max=(-1, 1))
                
                Metrics.save_img(
                    pred_scm_A, '{}/pred_scm_A_{}.png'.format(test_result_path, current_step))
                Metrics.save_img(
                    pred_scm_B, '{}/pred_scm_B_{}.png'.format(test_result_path, current_step))
                        
                Metrics.save_img(
                    gt_scm_A, '{}/gt_scm_A_{}.png'.format(test_result_path, current_step))
                Metrics.save_img(
                    gt_scm_B, '{}/gt_scm_B_{}.png'.format(test_result_path, current_step))
                           
        change_detection._collect_epoch_states()
        logs = change_detection.get_current_log()
        message = '[Test CD summary]: Test mF1=%.5f \n' %\
                      (logs['epoch_acc'])
        for k, v in logs.items():
            message += '{:s}: {:.4e} '.format(k, v)
            message += '\n'
        logger_test.info(message)
        
        semantic_segmentation._collect_epoch_states()
        logs = semantic_segmentation.get_current_log()
        message = '[Test SS summary]: Test mIoU=%.5f \n' %\
                      (logs['epoch_acc'])
        for k, v in logs.items():
            message += '{:s}: {:.4e} '.format(k, v)
            message += '\n'
        logger_test.info(message)
        
        logger.info('End of testing...')
