import argparse
import matplotlib.pyplot as plt
import torch

def plot_losses(losses_path='checkpoint/losses.pt', save_path="loss_plot.png", show=False, log_scale=False, smooth=0):
    losses = torch.load(losses_path, weights_only=True)

    losses = torch.as_tensor(losses)

    if smooth > 1:
        if len(losses) < smooth:
            print(f"Warning: Not enough data points ({len(losses)}) for smoothing window ({smooth}). Skipping smoothing.")
        else:
            kernel = torch.ones(smooth) / smooth
            losses = torch.nn.functional.conv1d(
                losses.view(1,1,-1), kernel.view(1,1,-1), padding=smooth//2
            ).view(-1)

    plt.figure(figsize=(10, 5))
    
    plt.plot(losses, label='Training Loss', color='#005b96', linewidth=2)
    
    plt.xlabel('Batch Index', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    if log_scale:
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    
    if show:
        plt.show()
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--losses-path', type=str, default='checkpoint/losses.pt', help='Path to the losses tensor file')
    parser.add_argument('--save-path', type=str, default='loss_plot.png', help='Path to save the loss plot image')
    parser.add_argument('--show', action='store_true', help='Whether to display the plot interactively')
    parser.add_argument('--log-scale', action='store_true', help='Whether to use logarithmic scale for y-axis')
    parser.add_argument('--smooth', type=int, default=0, help='Rolling average window size; 0=disabled')

    args = parser.parse_args()

    plot_losses(args.losses_path, args.save_path, args.show, args.log_scale, args.smooth)