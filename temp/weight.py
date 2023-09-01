import torch

# Load pretrained weights from a .pth file
pth_path_1 = "experiments/DDPM-scd-CDHead-SCD_230825_171227/checkpoint/cd_model_E0_gen.pth"
pth_path_2 = "experiments/DDPM-scd-CDHead-SCD_230825_171227/checkpoint/cd_model_E10_gen.pth"
pretrained_dict_1 = torch.load(pth_path_1)
pretrained_dict_2 = torch.load(pth_path_2)

# Print the keys and shapes of the loaded weights
for key in pretrained_dict_1:
    param_1 = pretrained_dict_1[key]
    param_2 = pretrained_dict_2[key]
    print(f"Layer: {key}, Size: {param_1.size()}, Values: {param_1}")
    print(f"Layer: {key}, Size: {param_2.size()}, Values: {param_2}")
    
    print(param_1 ==param_2)

