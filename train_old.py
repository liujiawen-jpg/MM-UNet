import torch
import torch.nn as nn
import torch.nn.functional as F
from data import RetinaDataset
from model import Unet
# from loss import DiceBCELoss, dice_coeff
import numpy as np
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt



train_augmented_path_images = sorted(glob('data/augmented/training/images/*'))
train_augmented_path_1st_manual = sorted(glob('data/augmented/training/1st_manual/*'))

test_augmented_path_images = sorted(glob('data/augmented/test/images/*'))
test_augmented_path_1st_manual = sorted(glob('data/augmented/test/1st_manual/*'))

train_dataset = RetinaDataset(train_augmented_path_images, train_augmented_path_1st_manual)

test_dataset = RetinaDataset(test_augmented_path_images, test_augmented_path_1st_manual)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

set_seed(seed)
def train(model, trainloader, optimizer, loss, epochs):
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        train_dice = 0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            l = loss(logits, masks)
            l.backward()
            optimizer.step()
            train_loss += l.item()
            train_dice += dice_coeff(logits, masks)
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_losses.append(train_loss)
        train_dices.append(train_dice)
        
        model.eval() #Validation
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for i, (images, masks) in enumerate(test_loader):
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                l = loss(logits, masks)
                val_loss += l.item()
                val_dice += dice_coeff(logits, masks)
        val_loss /= len(test_loader)
        val_dice /= len(test_loader)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        print(f"Epoch: {epoch + 1}  Train Loss: {train_loss:.4f} | Train DICE Coeff: {train_dice:.4f} | Val Loss: {val_loss:.4f} | Val DICE Coeff: {val_dice:.4f}")
        
    return train_losses, train_dices, val_losses, val_dices

class DICE_BCE_Loss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits= torch.sigmoid(logits)
        intersection = 2*(logits * targets).sum() + self.smooth
        union = (logits + targets).sum() + self.smooth
        dice_loss = 1. - intersection / union

        loss = nn.BCELoss()
        bce_loss = loss(logits, targets)

        return dice_loss + bce_loss
    
def dice_coeff(logits, targets):
    logits=torch.sigmoid(logits)
    intersection = 2*(logits * targets).sum()
    union = (logits + targets).sum()
    if union == 0:
        return 1
    dice_coeff = intersection / union
    return dice_coeff.item()

epochs = 30
loss =  DICE_BCE_Loss()
model = Unet(3, 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses, train_dices, val_losses, val_dices = train(model, train_loader, optimizer, loss, epochs)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_dices, label="Train DICE")
plt.plot(range(epochs), val_dices, label="Val DICE")
plt.xlabel("Epoch")
plt.ylabel("DICE Coeff")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(epochs), train_losses, label="Train Loss")
plt.plot(range(epochs), val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
