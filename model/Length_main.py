import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from model import Multitasking
from Dataset import MulDataset, transform, val_transform
from Maskloss import MaskedL1Loss
from Inference import Multrain_one_epoch, Mulevaluate
import torch.optim as optim

if __name__ == '__main__':
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    main_device = torch.device("cuda:0")
    model = Multitasking().to(main_device)
    model = nn.DataParallel(model, device_ids=device_ids)
    criterion_dist = MaskedL1Loss()
    criterion_num = MaskedL1Loss()
    optimizer = torch.optim.Adam([
        {'params': model.module.backbone.parameters(), 'lr': 1e-4},
        {'params': model.module.up.parameters(), 'lr': 1e-3},
        {'params': model.module.fc.parameters(), 'lr': 1e-3},
        {'params': model.module.fl.parameters(), 'lr': 1e-3},
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_val_loss = float('inf')
    best_model_save_path = "./result_length/best_model.pth"
    final_model_save_path = "./result_length/final_model.pth"
    num_epochs = 340


    train_dataset = MulDataset("./data/2DIR", "./data/contact", transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=14)
    val_dataset = MulDataset("./data/valA", "./data/valB", transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=14)

    for epoch in range(num_epochs):
        avg_train_loss_dist, avg_train_loss_num, avg_train_loss_total = Multrain_one_epoch(
            model, train_dataloader, criterion_dist, criterion_num, optimizer, main_device)
        avg_val_loss_dist, avg_val_loss_num, avg_val_loss_total = Mulevaluate(
            model, val_dataloader, criterion_dist, criterion_num, main_device)


        scheduler.step()
        if avg_val_loss_total < best_val_loss:
            best_val_loss = avg_val_loss_total
            torch.save(model.state_dict(), best_model_save_path)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Total Loss: {avg_train_loss_total:.4f}, Train Dist Loss: {avg_train_loss_dist:.4f}, Train Num Loss: {avg_train_loss_num:.4f}, Val Total Loss: {avg_val_loss_total:.4f}, Val Dist Loss: {avg_val_loss_dist:.4f}, Val Num Loss: {avg_val_loss_num:.4f}")

    torch.save(model.state_dict(), final_model_save_path)
