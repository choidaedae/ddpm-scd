# code for train pixel classifier 
# It needs to pre-trained DDPM Weight 

import torch
import torch.nn as nn
from tqdm import tqdm
import json
import os
import gc

from torch.utils.data import DataLoader

import argparse
from src.utils import setup_seed, multi_acc
from src.pixel_classifier import  load_ensemble, compute_iou, predict_labels, save_predictions, save_predictions, pixel_classifier
from src.datasets import ImageLabelDataset, FeatureDataset, make_transform
from src.feature_extractors import create_feature_extractor, collect_features

from guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from guided_diffusion.dist_util import dev

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def prepare_data(args):
    feature_extractor = create_feature_extractor(**args)
    
    print(f"Preparing the train set for {args['category']}...")
    dataset = ImageLabelDataset(
        data_dir=args['training_path'],
        resolution=args['image_size'],
        num_images=args['training_number'],
        transform=make_transform(
            args['model_type'],
            args['image_size']
        )
    )
    
    X = torch.zeros((len(dataset), *args['dim'][::-1]), dtype=torch.float32) 
    y = torch.zeros((len(dataset), *args['dim'][:-1]), dtype=torch.uint8)

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=device).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=device)
    else:
        noise = None 

    for row, (img, label) in enumerate(tqdm(dataset)):
        img = img[None].to(device)
        features = feature_extractor(img, noise=noise)
        X[row] = collect_features(args, features).cpu()
        
        for target in range(args['number_class']):
            if target == args['ignore_label']: continue
            if 0 < (label == target).sum() < 20:
                print(f'Delete small annotation from image {dataset.image_paths[row]} | label {target}')
                label[label == target] = args['ignore_label']
        y[row] = label
    
    d = X.shape[1]
    print(f'Total dimension {d}')
    # 여기서 문제 발생 
    X = X.permute(1,0,2,3).reshape(d, -1).permute(1, 0)
    y = y.flatten()
    return X[y != args['ignore_label']], y[y != args['ignore_label']]


def evaluation(args, models):
    feature_extractor = create_feature_extractor(**args)
   
    
    dataset = ImageLabelDataset(
        data_dir=args['testing_path'],
        resolution=args['image_size'],
        num_images=args['testing_number'],
        transform=make_transform(
            args['model_type'],
            args['image_size']
        )
    )
    
    

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=device).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=device)
    else:
        noise = None 

    preds, gts, uncertainty_scores = [], [], []
    a = True
    for img, label in tqdm(dataset):       
        img = img[None].to(device)
        min_value = img.min()
        max_value = img.max()

        # Normalize the image tensor using min-max normalization
        normalized_image = (img - min_value) / (max_value - min_value)

        print(normalized_image)
        features = feature_extractor(img, noise=noise)
    
        features = collect_features(args, features)
    

        x = features.view(args['dim'][-1], -1).permute(1, 0)
        torch.save(x, 'x_1.pt')
        pred, uncertainty_score = predict_labels(
            models, x, size=args['dim'][:-1]
        )
        gts.append(label.numpy())
        preds.append(pred.numpy())
        uncertainty_scores.append(uncertainty_score.item())
    
    save_predictions(args, dataset.image_paths, preds)
    miou = compute_iou(args, preds, gts)
    print(f'Overall mIoU: ', miou)
    print(f'Mean uncertainty: {sum(uncertainty_scores) / len(uncertainty_scores)}')


# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L434

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())
    
    parser.add_argument('--exp', type=str, default='config/SS/segmentation_test.json')
    parser.add_argument('--seed', type=int,  default=0)

    args = parser.parse_args()
    setup_seed(args.seed)

    # Load the experiment config
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))
    opts['image_size'] = opts['dim'][0] #image_size = 256

    # Prepare the experiment folder 
    if len(opts['steps']) > 0:
        suffix = '_'.join([str(step) for step in opts['steps']])
        suffix += '_' + '_'.join([str(step) for step in opts['blocks']])
        opts['exp_dir'] = os.path.join(opts['exp_dir'], suffix)

    path = opts['exp_dir']
    os.makedirs(path, exist_ok=True)
    print('Experiment folder: %s' % (path))
    os.system('cp %s %s' % (args.exp, opts['exp_dir']))

    model_list = ['best_model.pth', 'model_0_e_1_b_50.pth', 'model_0_e_2_b_50.pth', 'model_1_e_0_b_50.pth']
    model_num = 1
    # Check whether all models in ensemble are trained 
    pretrained = [os.path.exists(os.path.join(opts['exp_dir'], model_list[i]))
                  for i in range(model_num)]
    
    print(pretrained)
    
    print('Loading pretrained models...') 
    models = load_ensemble(opts, model_list, device='cuda') # 10개 모델 앙상블
    evaluation(opts, models)
    
    