# Semantic Change Detection Using Denoising Diffusion Probabilistic Model (DDPM-scd)
* This work

## How to use it?
It requires Docker, Pytorch 2.0.1, CUDA 11.8, CUDNN 8700. 

### Docker build

### Docker run 

### Docker

## DDPM-SCD train script 

1. If you want to training DDPM, 
python train_ddpm.py

2. If you want to training Pixel Classifier(Semantic Segmentation)
* It needs to pre-trained DDPM weight.
python train_pc.py 

3. If you want to training CD Network
* It needs to pre-trained DDPM weight.
python train_cd.py

4. If you want to test your DDPM (Sampling Images), 
python train_ddpm.py --config config/DDPM/ddpm_sampling.json --phase val

5. If you want to test your Pixel Classifier(Semantic Segmentation)
python test_pc.py --config config/PC/PC_{dataset_name}.json 

6. If you want to test your CD Network
python test_cd.py --config config/CD/{dataset_name}.json --phase test -log_eval

7. If you want to train your whole model
python train_model.py

8. If you want to test your whole model
python test_model.py  
