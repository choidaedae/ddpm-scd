DDPM-SCD train script 

1. If you want to training DDPM, 
python3 train_ddpm.py --config config/DDPM/ddpm_train.json --phase train -log_eval

2. If you want to training Pixel Classifier(Semantic Segmentation)
* It needs to pre-trained DDPM weight.
python3 train_pc.py --config config/PC/PC_{dataset_name}.json 

3. If you want to training CD Network
* It needs to pre-trained DDPM weight.
python3 train_cd.py --config config/CD/{dataset_name}.json --phase train -log_eval

4. If you want to test your DDPM (Sampling Images), 
python3 train_ddpm.py --config config/DDPM/ddpm_sampling.json --phase val

5. If you want to test your Pixel Classifier(Semantic Segmentation)
python3 eval_pc.py --config config/PC/PC_{dataset_name}.json 

6. If you want to test your CD Network
python3 eval_cd.py --config config/CD/{dataset_name}.json --phase test -log_eval

7. If you want to test your whole model
python3 test_model.py --config  config/SCD/sampling.json 