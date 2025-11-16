import argparse
import matplotlib.pyplot as plt
import torch
import os
import glob


def plot_losses(losses_path='checkpoint', save_path="loss_plot.png", show=False, log_scale=False):
    # load and concatenate all losses_epoch_*.pt
    pattern = os.path.join(losses_path, "losses_epoch_*.pt")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No files matching losses_epoch_*.pt found in {losses_path}")

    all_losses = []
    for f in files:
        epoch_losses = torch.load(f, weights_only=True)
        all_losses.extend(epoch_losses.tolist())

    losses = torch.tensor(all_losses)

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Batch Index')
    plt.ylabel('Loss')
    plt.title('Training Loss over Batches')
    plt.legend()
    plt.grid(True)

    if log_scale:
        plt.yscale('log')
    
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--losses-path', type=str, default='checkpoint', help='Directory containing losses_epoch_*.pt')
    parser.add_argument('--save-path', type=str, default='loss_plot.png', help='Path to save the loss plot image')
    parser.add_argument('--show', action='store_true', help='Whether to display the plot interactively')
    parser.add_argument('--log-scale', action='store_true', help='Whether to use logarithmic scale for y-axis')
    args = parser.parse_args()

    plot_losses(args.losses_path, args.save_path, args.show, args.log_scale)
