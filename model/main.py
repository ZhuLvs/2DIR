import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import CustomDeepLabV3
#from feature_fusion_model import CustomDeepLabV3
from Dataset import CustomDataset, transform, val_transform
from Maskloss import MaskedL1Loss
from Inference import train_one_epoch, evaluate

if __name__ == '__main__':
    device = torch.device("cuda:0")
    model = CustomDeepLabV3().to(device)
    criterion = MaskedL1Loss()
    optimizer = torch.optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-4},
        {'params': model.up.parameters(), 'lr': 1e-3}
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    writer = SummaryWriter()

    best_val_loss = float('inf')
    best_model_save_path = "../result/best_model.pth"
    final_model_save_path = "../result/final_model.pth"
    num_epochs = 340

    train_dataset = CustomDataset("../data/2DIR", "../data/contact", transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = CustomDataset("../data/valA", "../data/valB", transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    for epoch in range(num_epochs):
        avg_train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        avg_val_loss = evaluate(model, val_dataloader, criterion, device)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_save_path)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train MaskLoss: {avg_train_loss:.4f}, Val MaskLoss: {avg_val_loss:.4f}")



    writer.close()
    torch.save(model.state_dict(), final_model_save_path)
