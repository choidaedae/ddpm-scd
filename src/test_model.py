# code for test & evaluate semantic change detection with pre trained modules

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
from models.ddpmscd import ddpmscd as ddpmscd
from src.feature_extractors import create_feature_extractor, collect_features

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def test(opt, Net, diffusion, test_loader, logger, evaluate):
    
    feature_extractor = create_feature_extractor(**opt['segmentation'])
    
    cd = Net.module.change_detection
    ss = Net.module.semantic_segmentation
    
    cd._clear_cache()
    ss._clear_cache()

    test_result_path = '{}/test'.format(opt['path']
                                        ['results'])
    os.makedirs(test_result_path, exist_ok=True)
    torch.cuda.empty_cache()
    
    for current_step, test_data in enumerate(test_loader): # 이 때 test_data는 semantic change detection 용도의 data. segmentation과 change_detectio에 같은 데이터셋을 씀 
        # Feed data to diffusion model
        
        diffusion.feed_data(test_data) # test data를 diffusion에게 전달해주고, feature map을 뽑아옴. 
        
        # for change detection 
        fcd_A=[] 
        fcd_B=[]
        for t in opt['model_cd']['t']:
            fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t) #np.random.randint(low=2, high=8)
            if opt['model_cd']['feat_type'] == "dec":
                fcd_A.append(fd_A_t)
                fcd_B.append(fd_B_t)
                del fe_A_t, fe_B_t
            else:
                
                fcd_A.append(fe_A_t)
                fcd_B.append(fe_B_t)
                del fd_A_t, fd_B_t
                
        cd.feed_data(fcd_A, fcd_B, test_data) 
        
        fss_A = feature_extractor(test_data['A'], noise = None)
        fss_B = feature_extractor(test_data'B'], noise = None)
        
        fss_A = collect_features(opt['model_ss'], fss_A)
        fss_B = collect_features(opt['model_ss'], fss_B)

        x1 = fss_A.view(opt['model_ss']['dim'][-1], -1).permute(1, 0).unsqueeze(dim = 0)
        x2 = fss_B.view(opt['model_ss']['dim'][-1], -1).permute(1, 0).unsqueeze(dim = 0)

                
        ss.feed_data(x1, x2, test_data)
        
        ss.get_feature()
        
        seg_A, seg_B, change = Net(ss.fss_A, ss.fss_B, cd.fcd_A, cd.fcd_B)
        Net.module.collect_running_batch_states(seg_A, seg_B, change, test_data)
        
        '''
        if evaluate: # evaluate한다면 성능 측정 
            Net.module.collect_running_batch_states(seg_A, seg_B, change, test_data)

            logs    = cd.get_current_log()
            message = '[Testing SCD]. Iter: [%d/%d], running_mf1: %.5f\n' %\
                    (current_step, len(test_loader), logs['running_acc'])
            logger_test.info(message)

        '''

        # Visuals
        visuals_cd = cd.get_current_visuals(change, test_data)
        
        # Converting to uint8
        img_A   = Metrics.tensor2img(test_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8 # b, 3, 256, 256 
        img_B   = Metrics.tensor2img(test_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8 # b, 3, 256, 256
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
        
         # for semantic segmentation
        logs = ss.get_current_log()
        message = '[Test SS]. Iter: [%d/%d], SS_loss: %.5f, running_mIoU: %.5f\n' %\
        (current_step, len(test_loader), logs['l_sem'], logs['running_mIoU'])
        logger.info(message)
        
        #visuals
        segMap_A, segMap_B = ss.pixeltosegMap(seg_A, seg_B)     
        visuals_ss = ss.get_current_visuals(segMap_A, segMap_B, test_data) 
        
        visuals_ss['pred_ss_A'] = visuals_ss['pred_ss_A'] *2.0-1.0 # b, 256, 256, 0~1 로 normalize되어있음 
        visuals_ss['pred_ss_B'] = visuals_ss['pred_ss_B'] *2.0-1.0 # b, 256, 256 
        test_data['L1'] = test_data['L1']*2.0-1.0
        test_data['L2'] = test_data['L2']*2.0-1.0
        
        pred_ss_A = Metrics.tensor2img(visuals_ss['pred_ss_A'], out_type=np.uint8, min_max=(-1, 1)) # uint8
        pred_ss_B = Metrics.tensor2img(visuals_ss['pred_ss_B'], out_type=np.uint8, min_max=(-1, 1)) # uint8
            
        gt_scm_A = Metrics.tensor2img(test_data['L1'], out_type=np.uint8, min_max=(-1, 1))
        gt_scm_B = Metrics.tensor2img(test_data['L2'], out_type=np.uint8, min_max=(-1, 1))
        
        Metrics.save_img(
            pred_ss_A, '{}/pred_ss_A_{}.png'.format(test_result_path, current_step))
        Metrics.save_img(
            pred_ss_B, '{}/pred_ss_B_{}.png'.format(test_result_path, current_step))
            
        pred_scm_A_tensor, pred_scm_B_tensor = Metrics.make_SCD_map(pred_ss_A, pred_ss_B, pred_cm)
                        
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
     
    logger.info('End of testing...')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/SCD/ddpm_scd_test.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['test'],
                        help='Run either train(training + validation) or testing', default='test')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    #parser.add_argument('eval', default = 'false')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
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

        if phase == 'test' and args.phase == 'test':
            print("Creating [test] semantic-change-detection dataloader.")
            print(phase)
            test_set   = Data.create_scd_dataset(dataset_opt, phase)
            test_loader= Data.create_scd_dataloader(
                test_set, dataset_opt, phase)
            opt['len_test_dtaloader'] = len(test_loader)
    
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
   
    test(opt, Net, diffusion, test_loader, logger)