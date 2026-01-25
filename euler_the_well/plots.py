import argparse
import matplotlib.pyplot as plt
import torch
import os
import glob
import re

def plot_losses(losses_path='checkpoint', save_path="loss_plot.png", show=False, log_scale=False, smooth=0):
    pattern = os.path.join(losses_path, "*_epoch_*.pt")
    files = glob.glob(pattern)

    if len(files) == 0:
        raise FileNotFoundError(f"No files matching *_epoch_*.pt found in {losses_path}")

    try:
        files.sort(key=lambda f: int(re.search(r'epoch_(\d+)\.pt', f).group(1)))
    except AttributeError:
        print("Warning: Some files did not match the 'epoch_N.pt' format and may be sorted incorrectly.")
        files.sort()

    print(f"Found {len(files)} files. Processing...")

    all_losses = []
    for f in files:
        epoch_losses = torch.load(f, weights_only=True)
        all_losses.extend(epoch_losses.tolist())

    losses = torch.tensor(all_losses)

    if smooth > 1:
        if len(losses) < smooth:
            print(f"Warning: Not enough data points ({len(losses)}) for smoothing window ({smooth}). Skipping smoothing.")
        else:
            kernel = torch.ones(smooth) / smooth
            losses = torch.nn.functional.conv1d(
                losses.view(1,1,-1), kernel.view(1,1,-1), padding=smooth//2
            ).view(-1)

    plt.figure(figsize=(10, 5))
    
    # Professional Scientific Style: Deep blue color, slightly thicker line
    plt.plot(losses, label='Training Loss', color='#005b96', linewidth=2)
    
    plt.xlabel('Batch Index', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Remove top and right spines for a cleaner look
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    if log_scale:
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--losses-path', type=str, default='checkpoints', help='Directory containing losses .pt files')
    parser.add_argument('--save-path', type=str, default='loss_plot.png', help='Path to save the loss plot image')
    parser.add_argument('--show', action='store_true', help='Whether to display the plot interactively')
    parser.add_argument('--log-scale', action='store_true', help='Whether to use logarithmic scale for y-axis')
    parser.add_argument('--smooth', type=int, default=0, help='Rolling average window size; 0=disabled')

    args = parser.parse_args()

    plot_losses(args.losses_path, args.save_path, args.show, args.log_scale, args.smooth)
    