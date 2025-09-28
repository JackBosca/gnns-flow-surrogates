import argparse
import matplotlib.pyplot as plt
import torch


def plot_losses(losses_path='checkpoint/losses.pt', save_path="loss_plot.png", show=False):
    losses = torch.load(losses_path, weights_only=True)

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Batch Index')
    plt.ylabel('Loss')
    plt.title('Training Loss over Batches')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)  
    if show:
        plt.show()           
    plt.close()              

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--losses-path', type=str, default='checkpoint/losses.pt', help='Path to the losses tensor file')
    parser.add_argument('--save-path', type=str, default='loss_plot.png', help='Path to save the loss plot image')
    parser.add_argument('--show', action='store_true', help='Whether to display the plot interactively')
    args = parser.parse_args()

    plot_losses(args.losses_path, args.save_path, args.show)
