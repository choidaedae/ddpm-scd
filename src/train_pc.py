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
from src.datasets import ImageLabelDataset, FeatureDataset, DividedImageLabelDataset, make_transform
from src.feature_extractors import create_feature_extractor, collect_features

from guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from guided_diffusion.dist_util import dev

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4, 5"

def prepare_data(args, idx):
    feature_extractor = create_feature_extractor(**args)
    
    print(f"Preparing the train set for {args['category']}...")
    dataset = DividedImageLabelDataset(
        data_dir=args['training_path'],
        resolution=args['image_size'],
        loop_idx = idx,
        num_images=args['data_load_size'],
        transform=make_transform(
            args['model_type'],
            args['image_size']
        )
    )
    
    X = torch.zeros((len(dataset), *args['dim'][::-1]), dtype=torch.float32) 
    y = torch.zeros((len(dataset), *args['dim'][:-1]), dtype=torch.uint8)

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None 

    for row, (img, label) in enumerate(tqdm(dataset)):
        img = img[None].to(dev())
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
    X = X.permute(1,0,2,3).contiguous().view(d, -1).permute(1, 0) # 327680
    y = y.flatten() # 5x256x256
    
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
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None 

    preds, gts, uncertainty_scores = [], [], []
    for img, label in tqdm(dataset):        
        img = img[None].to(dev())
        features = feature_extractor(img, noise=noise)
        features = collect_features(args, features)

        x = features.view(args['dim'][-1], -1).permute(1, 0)
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
def train(args):
    
    num_batches = int(args['training_number'] / args['data_load_size'])
    
    print(torch.cuda.device_count())
    
    print(f'number of batches: {num_batches}')
 
    classifier = pixel_classifier(numpy_class=(args['number_class']), dim=args['dim'][-1])
    classifier.init_weights()
    if args['distributed']:
        gpu_ids = args['gpu_ids']
        print(f"distributed learning for gpu ids = {gpu_ids}...")
        classifier = nn.DataParallel(classifier, device_ids=args['gpu_ids']).cuda()
    else : classifier = nn.DataParallel(classifier, device_ids=[0]).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    
    iteration = 0
    
    # pre-trained model load하는 부분 
    
    for i in range(args['start_model_num'], args['model_num'], 1):
        
        for epoch in range(3):
        
            for batch_idx in range(num_batches): 
                
                iteration = 0

                features, labels = prepare_data(args, batch_idx) 
                train_data = FeatureDataset(features, labels)
                
                print(f" ********* train for the batch {batch_idx}, for the data {args['data_load_size']*batch_idx} ~ {args['data_load_size']*(batch_idx+1)-1}***********")
                print(f" ********* max_label {args['number_class']} *** ignore_label {args['ignore_label']} ***********")
                print(f" *********************** Current number data {len(features)} ***********************")
                train_loader = DataLoader(dataset=train_data, batch_size=args['inner_batch_size'], shuffle=True, drop_last=True, num_workers=16)

                print(" *********************** Current dataloader length " +  str(len(train_loader)) + " ***********************")

                classifier.train()
                
                for X_batch, y_batch in train_loader: 
                    X_batch, y_batch = X_batch.to(dev()), y_batch.to(dev())
                    y_batch = y_batch.type(torch.long)
                    
                    optimizer.zero_grad()
                    y_pred = classifier(X_batch)
                    loss = criterion(y_pred, y_batch) 
                    acc = multi_acc(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    iteration += 1
                    print(f"{iteration} iters...")
                
                print('Model Num: ', str(i), 'Epoch : ', str(epoch), 'iteration', iteration, 'training loss', loss.item(), 'acc', acc)
                
                if(batch_idx % 50 == 0): # save model per 50 batch 
                    model_path = os.path.join(args['exp_dir'], 
                                        f'model_{i}_e_{str(epoch)}_b_{str(batch_idx)}.pth')
                    print('save to:', model_path)
                    torch.save({'model_state_dict': classifier.state_dict()},
                        model_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())
    
    parser.add_argument('--exp', type=str, default='config/SS/segmentation.json')
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
        opts['exp_dir'] = os.path.join(opts['exp_dir'], 'segmentation', suffix)

    path = opts['exp_dir']
    os.makedirs(path, exist_ok=True)
    print('Experiment folder: %s' % (path))
    os.system('cp %s %s' % (args.exp, opts['exp_dir']))

    # Check whether all models in ensemble are trained 
    pretrained = [os.path.exists(os.path.join(opts['exp_dir'], f'model_{i}.pth')) 
                  for i in range(opts['model_num'])]
              
    if not all(pretrained):
        # train all remaining models
        opts['start_model_num'] = sum(pretrained) # ensemble 10 models 
        train(opts)
    
    