import torch
import matplotlib.pyplot as plt


images, masks = next(iter(test_loader))
with torch.no_grad():
    pred = model(images.to(device)).cpu().detach()
    pred = pred > 0.5

def display_batch(images, masks, pred):
    fig, axes = plt.subplots(len(images), 3, figsize=(20, 6 * len(images)))
    fig.tight_layout()

    for i in range(len(images)):
        axes[i, 0].imshow(images[i].permute(1,2,0))
        axes[i, 0].set_title(f'Image {i+1}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(masks[i].squeeze(), cmap='gray')
        axes[i, 1].set_title(f'Mask {i+1}')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred[i].squeeze(), cmap='gray')
        axes[i, 2].set_title(f'Prediction {i+1}')
        axes[i, 2].axis('off')

    plt.show()

display_batch(images, masks, pred)
