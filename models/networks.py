import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import modules
logger = logging.getLogger('base')
#from model.cd.cd_head import cd_head
from models.cd.cd_head_diff import cd_head_diff
from guided_diffusion import gaussian_diffusion, diffusion, unet
from src.pixel_classifier import pixel_classifier
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def define_G(opt): # Image Denoising하는 module (DDPM)
    
    model_opt = opt['model'] # diffusion model을 ddpm 논문에서의 ddpm과 sr3 논문에서의 ddpm 중 하나를 갖다 쓸 수 있는 것 같다. 여기에 ddpm-ip를 갖다 붙이면 좀 더 잘 될수도? 

    if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None: model_opt['unet']['norm_groups']=32
    model = unet.UNet( # ddpm = unet 기반, UNet instance 정의 
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        inner_channel=model_opt['unet']['inner_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        channel_mults=model_opt['unet']['channel_mults'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['diffusion']['image_size']
    )
    netG = diffusion.GaussianDiffusion( # Diffusion model instance 정의 
        model,
        image_size=model_opt['diffusion']['image_size'],
        channels=model_opt['diffusion']['channels'],
        loss_type=model_opt['diffusion']['loss'],    # L1 or L2
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train']
    )
    if opt['phase'] == 'train':
        # init_weights(netG, init_type='kaiming', scale=0.1) # weight initialization option 선택 가능 
        init_weights(netG, init_type='orthogonal')
    if opt['gpu_ids'] and opt['distributed']: # gpu training 
        assert torch.cuda.is_available() 
        print("Distributed training")
        netG = nn.DataParallel(netG)
    return netG


def define_SS(opt): #pixel classifier module
    seg_model_opt = opt['model_ss']
    
    netSS = pixel_classifier(numpy_class=seg_model_opt["number_class"], 
                             dim = seg_model_opt["dim"][-1])
    
    if opt['phase'] == 'train':
        init_weights(netSS, init_type='normal')
        
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netSS = nn.DataParallel(netSS, opt['gpu_ids'])
    
    return netSS

    
# Change Detection Network
def define_CD(opt): #change detection module
    cd_model_opt = opt['model_cd']
    diffusion_model_opt = opt['model']
    rank = opt['rank']
    
    netCD = cd_head_diff(feat_scales=cd_model_opt['feat_scales'], 
                    out_channels=cd_model_opt['out_channels'], 
                    inner_channel=diffusion_model_opt['unet']['inner_channel'], 
                    channel_mults=diffusion_model_opt['unet']['channel_mults'],
                    img_size=cd_model_opt['output_cm_size'],
                    time_steps=cd_model_opt["t"])
    
    # Initialize the change detection head if it is 'train' phase 
    if opt['phase'] == 'train':
        init_weights(netCD, init_type='orthogonal')
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netCD = nn.parallel.DistributedDataParallel(netCD, device_ids=opt['gpu_ids'])
    
    return netCD
