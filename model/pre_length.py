import os
import pandas as pd
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from model import Multitasking

def make_symmetric_and_zero_diag_numpy(matrix):
    matrix_abs = np.abs(matrix)
    np.fill_diagonal(matrix_abs, 0)
    matrix_symmetric = (matrix_abs + matrix_abs.T) / 2
    return matrix_symmetric


def predict_and_save(model, img_folder, transform, device, output_folder):
    model.eval()
    with torch.no_grad():
        for img_name in os.listdir(img_folder):
            prefix = img_name.split('.')[0]
            img_path = os.path.join(img_folder, img_name)
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            outputs_dist, outputs_num = model(img)
            output_matrix = outputs_dist.squeeze().cpu().numpy()
            output_matrix_symmetric = make_symmetric_and_zero_diag_numpy(output_matrix)
            predicted_num = outputs_num.squeeze().cpu().item()
            output_matrix_symmetric[0, 0] = predicted_num
            output_csv_path = os.path.join(output_folder, prefix + ".csv")
            pd.DataFrame(output_matrix_symmetric).to_csv(output_csv_path, index=False, header=False)

def remove_module_prefix(state_dict):
    """
    Remove the 'module.' prefix from each key in the state dictionary.
    """
    return {k.replace('module.', ''): v for k, v in state_dict.items()}

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Multitasking().to(device)
    state_dict = (torch.load("./result_length/best_model.pth", map_location=device))
    new_state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(new_state_dict)

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_folder = "./data/valA"
    output_folder = "./data/out_length"
        # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    predict_and_save(model, img_folder, val_transform, device, output_folder)
