"""
render.py

Render videos from rollout results saved by rollout.py.

Expected result file format (pickle):
    with open('results_X.pkl', 'rb') as f:
        result, crds = pickle.load(f)

where result == [predicteds, targets]
  - predicteds: shape (T, N, vel_dim)
  - targets:    shape (T, N, vel_dim)
and crds: (N, 2) or (N, D) with at least first two columns = x,y coordinates.

Produces an mp4 file for each pkl in --out-dir (default: videos).
"""
import os
import argparse
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from PIL import Image
import cv2


def fig2data(fig):
    """
    Convert a Matplotlib figure to an HxWx4 RGBA numpy array.
    Uses tostring_argb and rolls to RGBA (RGB + alpha).
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # tostring_argb returns width*height*4 bytes in ARGB order; reshape to (h,w,4)
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    try:
        buf.shape = (h, w, 4)
    except Exception:
        # fallback: try transpose if width/height are swapped
        buf.shape = (w, h, 4)
        buf = buf.transpose((1, 0, 2))
    # ARGB -> RGBA
    buf = np.roll(buf, -1, axis=2)  # move A from front to last position
    return buf  # uint8 RGBA


def compute_global_scales(predicteds, targets):
    """
    Compute global min/max velocity magnitudes and their differences.
    Args:
        predicteds: (T, N, d) array of predicted velocities
        targets:    (T, N, d) array of target velocities
    Returns:
        vmin, vmax: float, global min and max velocity magnitudes
        diff_vmax:  float, maximum difference magnitude between predicted and target
    """
    pred_mag = np.linalg.norm(predicteds, axis=-1)  # (T, N)
    targ_mag = np.linalg.norm(targets, axis=-1)
    all_mag = np.concatenate([pred_mag.ravel(), targ_mag.ravel()])

    vmin = float(np.nanmin(all_mag))
    vmax = float(np.nanmax(all_mag))
    # if vmax==vmin, expand a tiny bit to avoid matplotlib warnings
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    
    # compute diff max
    diff = np.linalg.norm(predicteds - targets, axis=-1)
    diff_vmax = float(np.nanmax(diff))
    if np.isclose(diff_vmax, 0.0):
        diff_vmax = 1e-6
    return vmin, vmax, diff_vmax


def render_file(pkl_path, out_dir, fps=20, size=(1700, 800), skip=1,
                cmap='viridis', codec='mp4v', write_frames=False):
    """Render a single result pickle to a video (and optionally frame PNGs)."""
    basename = os.path.splitext(os.path.basename(pkl_path))[0]
    out_path = os.path.join(out_dir, f'{basename}.mp4')

    with open(pkl_path, 'rb') as f:
        loaded = pickle.load(f)

    if isinstance(loaded, (list, tuple)) and len(loaded) == 2:
        result, crds = loaded
    else:
        raise RuntimeError(f'Unexpected pickle format in {pkl_path}')

    predicteds = np.asarray(result[0])  # (T, N, d)
    targets = np.asarray(result[1])     # (T, N, d)
    crds = np.asarray(crds)             # (N, 2) or (N, D)

    if crds.ndim != 2 or crds.shape[1] < 2:
        raise RuntimeError('crds must be shape (N, 2) or (N, D>=2)')

    triang = mtri.Triangulation(crds[:, 0], crds[:, 1])

    T_tot = predicteds.shape[0]
    if targets.shape[0] != T_tot:
        raise RuntimeError('predicteds and targets must have same time dimension')

    # compute global scales
    vmin, vmax, diff_vmax = compute_global_scales(predicteds, targets)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(out_path, fourcc, float(fps), (int(size[0]), int(size[1])))

    frame_dir = None
    if write_frames:
        frame_dir = os.path.join(out_dir, f'{basename}_frames')
        os.makedirs(frame_dir, exist_ok=True)

    dpi = 100
    fig_w = size[0] / dpi
    fig_h = size[1] / dpi

    indices = list(range(0, T_tot, skip))
    pbar = tqdm(indices, desc=f'Rendering {basename}', unit='frame')
    for step in pbar:
        pred = predicteds[step]   # (N, d)
        targ = targets[step]      # (N, d)

        pred_v = np.linalg.norm(pred, axis=-1)
        targ_v = np.linalg.norm(targ, axis=-1)
        diff_v = np.linalg.norm(targ - pred, axis=-1)

        # 3-row layout: Target, Prediction, Difference
        fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_h), dpi=dpi, constrained_layout=True)
        for ax in axes:
            ax.cla()
            ax.triplot(triang, 'o-', color='k', ms=0.4, lw=0.3)

        h1 = axes[0].tripcolor(triang, targ_v, shading='gouraud', vmin=vmin, vmax=vmax, cmap=cmap)
        h2 = axes[1].tripcolor(triang, pred_v, shading='gouraud', vmin=vmin, vmax=vmax, cmap=cmap)
        h3 = axes[2].tripcolor(triang, diff_v, shading='gouraud', vmin=0.0, vmax=diff_vmax, cmap=cmap)

        axes[0].set_title(f'Target — Time @ {step * 0.01:.2f}s')
        axes[1].set_title(f'Prediction — Time @ {step * 0.01:.2f}s')
        axes[2].set_title(f'|Target - Prediction| — Time @ {step * 0.01:.2f}s')

        # colorbars: one for top two (shared scale), and one for diff
        cbar1 = fig.colorbar(h1, ax=[axes[0], axes[1]], orientation='vertical', shrink=0.8)
        cbar2 = fig.colorbar(h3, ax=axes[2], orientation='vertical', shrink=0.8)

        img_rgba = fig2data(fig)
        plt.close(fig)

        img_rgb = img_rgba[:, :, :3].astype(np.uint8)
        img_rgb = cv2.resize(img_rgb, (int(size[0]), int(size[1])))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        out.write(img_bgr)

        if write_frames:
            frame_path = os.path.join(frame_dir, f'{basename}_frame_{step:05d}.png')
            Image.fromarray(img_rgb).save(frame_path)

    out.release()
    print(f'Wrote video: {out_path}')
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', type=str, default='rollout/', required=True,
                   help='Directory containing result .pkl files (e.g. rollout/)')
    p.add_argument('--out-dir', type=str, default='videos', help='Output directory for videos')
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--size', nargs=2, type=int, default=[1700, 800], help='Output video size: width height (pixels)')
    p.add_argument('--skip', type=int, default=5, help='Render every `skip` timesteps (skip=1 means every frame)')
    p.add_argument('--cmap', type=str, default='viridis', help='Matplotlib colormap for plots')
    p.add_argument('--codec', type=str, default='mp4v', help='FourCC codec for cv2 VideoWriter (e.g. mp4v, XVID)')
    p.add_argument('--write-frames', action='store_true', help='Also save each frame as PNG (in subdir)')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # list .pkl files inside results-dir
    if not os.path.isdir(args.results_dir):
        print('results-dir does not exist or is not a directory:', args.results_dir)
        return

    files = sorted([
        os.path.join(args.results_dir, fname)
        for fname in os.listdir(args.results_dir)
        if fname.lower().endswith('.pkl')
    ])

    if len(files) == 0:
        print('No .pkl result files found in:', args.results_dir)
        return

    for file in files:
        try:
            render_file(file, args.out_dir, fps=args.fps, size=tuple(args.size),
                        skip=args.skip, cmap=args.cmap, codec=args.codec,
                        write_frames=args.write_frames)
        except Exception as e:
            print(f'Failed to render {file}: {e}')


if __name__ == '__main__':
    main()
