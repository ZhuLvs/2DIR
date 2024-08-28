import os
import pandas as pd
import numpy as np
import torch
from PIL import Image

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        outputs = torch.abs(outputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            outputs = torch.abs(outputs)
            outputs = (outputs + outputs.transpose(-1, -2)) / 2
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

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
            output = model(img)
            output_matrix = output.squeeze().cpu().numpy()
            output_matrix = make_symmetric_and_zero_diag_numpy(output_matrix)
            output_csv_path = os.path.join(output_folder, prefix + ".csv")
            pd.DataFrame(output_matrix).to_csv(output_csv_path, index=False, header=False)

def Mulmake_symmetric_and_zero_diag(matrix):

    matrix_symmetric = (matrix + matrix.transpose(-2, -1)) / 2
    batch_size, num_rows, num_cols = matrix_symmetric.shape
    mask = torch.ones(num_rows, num_cols, device=matrix.device) - torch.eye(num_rows, num_cols, device=matrix.device)
    matrix_processed = matrix_symmetric * mask
    return matrix_processed


def Multrain_one_epoch(model, dataloader, criterion_dist, criterion_num, optimizer, device):
    model.train()
    total_loss_dist = 0
    total_loss_num = 0
    for images, targets_dist, targets_num in dataloader:
        images = images.to(device)
        targets_dist = targets_dist.to(device)
        targets_num = targets_num.to(device).view(-1, 1)

        optimizer.zero_grad()
        outputs_dist, outputs_num = model(images)

        outputs_dist_processed = torch.abs(outputs_dist)
        loss_dist = criterion_dist(outputs_dist_processed, targets_dist)
        loss_num = criterion_num(outputs_num, targets_num)
        total_loss = loss_dist + loss_num

        total_loss.backward()
        optimizer.step()
        total_loss_dist += loss_dist.item()
        total_loss_num += loss_num.item()

    avg_loss_dist = total_loss_dist / len(dataloader)
    avg_loss_num = total_loss_num / len(dataloader)
    avg_train_loss_total = avg_loss_dist + avg_loss_num

    return avg_loss_dist, avg_loss_num, avg_train_loss_total



def Mulevaluate(model, dataloader, criterion_dist, criterion_num, device):
    model.eval()
    total_loss_dist = 0
    total_loss_num = 0
    with torch.no_grad():
        for images, targets_dist, targets_num in dataloader:
            images = images.to(device)
            targets_dist = targets_dist.to(device)
            targets_num = targets_num.to(device).view(-1, 1)

            outputs_dist, outputs_num = model(images)

            outputs_dist_processed = torch.abs(outputs_dist)
            loss_dist = criterion_dist(outputs_dist_processed, targets_dist)
            loss_num = criterion_num(outputs_num, targets_num)

            total_loss_dist += loss_dist.item()
            total_loss_num += loss_num.item()

    avg_loss_dist = total_loss_dist / len(dataloader)
    avg_loss_num = total_loss_num / len(dataloader)
    avg_val_loss_total = avg_loss_dist + avg_loss_num
    return avg_loss_dist, avg_loss_num, avg_val_loss_total

def Mulmake_symmetric_and_zero_diag_numpy(matrix):

    matrix_abs = np.abs(matrix)
    matrix_symmetric = (matrix_abs + matrix_abs.T) / 2
    np.fill_diagonal(matrix_symmetric, 0)

    return matrix_symmetric



def Mulpredict_and_save(model, img_folder, transform, device, output_folder):
    model.eval()
    with torch.no_grad():
        for img_name in os.listdir(img_folder):
            prefix = img_name.split('.')[0]
            img_path = os.path.join(img_folder, img_name)
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            outputs_dist, outputs_num = model(img)
            output_matrix = outputs_dist.squeeze().cpu().numpy()
            output_matrix_symmetric = Mulmake_symmetric_and_zero_diag_numpy(output_matrix)
            predicted_num = outputs_num.squeeze().cpu().item()
            output_matrix_symmetric[0, 0] = predicted_num
            output_csv_path = os.path.join(output_folder, prefix + ".csv")
            pd.DataFrame(output_matrix_symmetric).to_csv(output_csv_path, index=False, header=False)