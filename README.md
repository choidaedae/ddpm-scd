# Semantic Change Detection Using Denoising Diffusion Probabilistic Model (DDPM-scd)
* This work is done during the internship in Meissa Planet.
* This model is large vision model which can do some remote sensing image analysis tasks, such as change detection, semantic segmentation, and semantic change detection.

![DDPM-scd Architecture]((https://github.com/choidaedae/ddpm-scd/assets/105369646/5c98135c-32f5-4539-85ca-b185586e02cd)


  
## How to use it?
* It requires Docker, Pytorch 2.0.1, CUDA 11.8, CUDNN 8700. 
* To make environment to get ready for running model, build Docker following below
* Or you can also use conda environment

### if you want to use Docker 
#### 1. Docker build
```
  docker build {image_name} .
```
#### 2. Docker run 
```
  docker run --gpus all
```

### If you want to use conda 
#### 1. Create Conda virtual environment
```
  conda create -n {environment_name} python=3.8.10
```
#### 2. Activate Conda environment
```
  conda activate {environment_name}
```
#### 3. Install all requirements to run model 
```
  pip install -r requirements.txt
```
## DDPM-SCD train script 
#### 1. If you want to training DDPM, 
```
  python train_ddpm.py
```
#### 2. If you want to training Pixel Classifier(Semantic Segmentation)
* It needs to pre-trained DDPM weight.
```
  python train_pc.py 
```
#### 3. If you want to training CD Network
* It needs to pre-trained DDPM weight.
```
python train_cd.py
```
#### 4. If you want to test your DDPM (Sampling Images), 
```
python train_ddpm.py --config config/DDPM/ddpm_sampling.json --phase val
```

#### 5. If you want to test your Pixel Classifier(Semantic Segmentation)
```
python test_pc.py --config config/PC/PC_{dataset_name}.json 
```
#### 6. If you want to test your CD Network
```
python test_cd.py --config config/CD/{dataset_name}.json --phase test -log_eval
```
#### 7. If you want to train your whole model
```
python train_model.py
```
#### 8. If you want to test your whole model
```
python test_model.py  
```
