import sys
import torch
from torch import nn
from typing import List
from collections import OrderedDict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"current device: {device}")


def create_feature_extractor(model_type, **kwargs):
    """ Create the feature extractor for <model_type> architecture. """
    if model_type == 'ddpm':
        print("Creating DDPM Feature Extractor...")
        feature_extractor = FeatureExtractorDDPM(**kwargs)
    else:
        raise Exception(f"Wrong model type: {model_type}")
    return feature_extractor


def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    if type(features) in [list, tuple]:
        features = [f.detach().float() if f is not None else None 
                    for f in features]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())


def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out


def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'activations')
    return out

def pt_processing(model_path):
    pt = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k,v in pt.items():
        if k[:11] == "denoise_fn.":
            key = k.replace("denoise_fn.", "")
            new_state_dict[key] = v
            
    return new_state_dict

class FeatureExtractor(nn.Module):
    def __init__(self, model_path: str, input_activations: bool, **kwargs):
        ''' 
        Parent feature extractor class.
        
        param: model_path: path to the pretrained model
        param: input_activations: 
            If True, features are input activations of the corresponding blocks
            If False, features are output activations of the corresponding blocks
        '''
        print(model_path)
        super().__init__()
        self._load_pretrained_model(model_path, **kwargs)
        print(f"Pretrained model is successfully loaded from {model_path}")
        self.save_hook = save_input_hook if input_activations else save_out_hook
        self.feature_blocks = []

    def _load_pretrained_model(self, model_path: str, **kwargs):
        pass


class FeatureExtractorDDPM(FeatureExtractor):
    ''' 
    Wrapper to extract features from pretrained DDPMs.
            
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''
    
    def __init__(self, steps: List[int], blocks: List[int], **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        # blocks 뭔지 보기 
        # Save decoder activations # encoder를 선택할 수도 있음. 
        for idx, block in enumerate(self.model.ups):
            if idx in blocks:
                block.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block) # feature block 리스트에 저장 

    def _load_pretrained_model(self, model_path, **kwargs):
        import inspect
        import guided_diffusion.dist_util as dist_util
        # import guided_diffusion.guided_diffusion.dist_util as dist_util
        from guided_diffusion.script_util import create_model_and_diffusion
        # from guided_diffusion.guided_diffusion.script_util import create_model_and_diffusion

        # Needed to pass only expected args to the function
        argnames = inspect.getfullargspec(create_model_and_diffusion)[0]
        
        unet_args = kwargs['model']['unet'] 
        diffusion_args = kwargs['model']['diffusion']
        args = {**unet_args, **diffusion_args}
        expected_args = {name: args[name] for name in argnames}
        self.model, self.diffusion = create_model_and_diffusion(**expected_args)

        
        state_dict = pt_processing(model_path)
        self.model.load_state_dict(state_dict)
        
        self.model.to(dist_util.dev())
        print(dist_util.dev())
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, noise=None):
        activations = []
        for t in self.steps:
            # Compute x_t and run DDPM
            t = torch.tensor([t]).to(x.device)
            noisy_x = self.diffusion.q_sample(x, t, noise=noise)
            self.model(noisy_x, self.diffusion._scale_timesteps(t))

            # Extract activations
            for block in self.feature_blocks:
                activations.append(block.activations)
                block.activations = None

        # Per-layer list of activations [N, C, H, W]
        return activations


def collect_features(args, activations: List[torch.Tensor], sample_idx=0):
    """ Upsample activations and concatenate them to form a feature tensor """
    assert all([isinstance(acts, torch.Tensor) for acts in activations])
    size = tuple(args['dim'][:-1])
    resized_activations = []
    for feats in activations:
        feats = feats[sample_idx][None]
        feats = nn.functional.interpolate( 
            feats, size=size, mode=args["upsample_mode"] # size = 256으로 모든 featuremap을 일괄적으로 \
            # bilinear interpolation해서 붙임.
        )
        resized_activations.append(feats[0])
    ret = torch.cat(resized_activations, dim=0)
    return ret




