"""
Rendering file used for report images.
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
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    
    # compute diff max
    diff = np.linalg.norm(predicteds - targets, axis=-1)
    diff_vmax = float(np.nanmax(diff))
    if np.isclose(diff_vmax, 0.0):
        diff_vmax = 1e-6
    return vmin, vmax, diff_vmax

def render_file(pkl_path, out_dir, fps=None, size=(1920, 1080), skip=1,
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

    predicteds = np.asarray(result[0])
    targets = np.asarray(result[1])

    coords, triangles = crds 
    coords = np.asarray(coords)
    tri_arr = np.asarray(triangles)

    if tri_arr.ndim == 2 and tri_arr.shape[0] == 3 and tri_arr.shape[1] != 3:
        tri_arr = tri_arr.T
    tri_arr = tri_arr.astype(np.int64)

    triang = mtri.Triangulation(coords[:, 0], coords[:, 1], triangles=tri_arr)
    T_tot = predicteds.shape[0]
    if targets.shape[0] != T_tot:
        raise RuntimeError('predicteds and targets must have same time dimension')

    vmin, vmax, diff_vmax = compute_global_scales(predicteds, targets)
    dt = 0.01

    if fps is None:
        fps = 1.0 / (skip * dt)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(out_path, fourcc, float(fps), (int(size[0]), int(size[1])))

    frame_dir = None
    if write_frames:
        frame_dir = os.path.join(out_dir, f'{basename}_frames')
        os.makedirs(frame_dir, exist_ok=True)

    dpi = 200
    fig_w = size[0] / dpi
    fig_h = size[1] / dpi

    # --- FONT SIZES ---
    title_fs = 13
    cbar_label_fs = 12
    cbar_tick_fs = 9
    axis_label_fs = 11
    tick_label_fs = 11

    indices = list(range(0, T_tot, skip))
    pbar = tqdm(indices, desc=f'Rendering {basename}', unit='frame')
    last_step = indices[-1]

    for step in pbar:
        pred = predicteds[step]
        targ = targets[step]

        pred_v = np.linalg.norm(pred, axis=-1)
        targ_v = np.linalg.norm(targ, axis=-1)
        diff_v = np.linalg.norm(targ - pred, axis=-1)

        fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_h), dpi=dpi)
        
        fig.tight_layout(pad=3.0) 
        fig.subplots_adjust(left=0.1, right=0.88, bottom=0.12, hspace=0.4) 

        for ax in axes:
            ax.cla()
            ax.set_aspect('equal', adjustable='box')

        tri_nodes = triang.triangles
        targ_tri = targ_v[tri_nodes].mean(axis=1)
        pred_tri = pred_v[tri_nodes].mean(axis=1)
        diff_tri = diff_v[tri_nodes].mean(axis=1)

        h1 = axes[0].tripcolor(triang, targ_tri, shading='flat', vmin=vmin, vmax=vmax, cmap=cmap)
        h2 = axes[1].tripcolor(triang, pred_tri, shading='flat', vmin=vmin, vmax=vmax, cmap=cmap)
        h3 = axes[2].tripcolor(triang, diff_tri, shading='flat', vmin=0.0, vmax=diff_vmax, cmap=cmap)

        for ax in axes:
            ax.triplot(triang, '-', color='k', lw=max(0.2 * (dpi / 100.0), 0.1), ms=0)

        axes[0].set_title(fr'$\| u_{{target}} \|$ at {step * dt:.2f} s', fontsize=title_fs)
        axes[1].set_title(fr'$\| u_{{pred}} \|$ at {step * dt:.2f} s', fontsize=title_fs)
        axes[2].set_title(fr'$\| u_{{target}} - u_{{pred}} \|$ at {step * dt:.2f} s', fontsize=title_fs)

        for ax in axes:
            ax.set_ylabel(r'$y \; [m]$', fontsize=axis_label_fs, labelpad=12)
            ax.tick_params(axis='both', which='major', labelsize=tick_label_fs)
            ax.tick_params(axis='both', which='minor', labelsize=max(6 * (dpi/100.0), 4))

        axes[0].set_xticklabels([])
        axes[1].set_xticklabels([])

        axes[2].set_xlabel(r'$x \; [m]$', fontsize=axis_label_fs, labelpad=12)

        cbar1 = fig.colorbar(h1, ax=[axes[0], axes[1]], orientation='vertical', shrink=0.8)
        cbar1.set_label('$\| u \|$ [m/s]', fontsize=cbar_label_fs, labelpad=10)
        cbar1.ax.tick_params(labelsize=cbar_tick_fs)

        cbar2 = fig.colorbar(h3, ax=axes[2], orientation='vertical', shrink=1.0)
        cbar2.set_label('Error [m/s]', fontsize=cbar_label_fs, labelpad=10)
        cbar2.ax.tick_params(labelsize=cbar_tick_fs)

        img_rgba = fig2data(fig)
        img_rgb = img_rgba[:, :, :3].astype(np.uint8)

        if write_frames:
            frame_path = os.path.join(frame_dir, f'{basename}_frame_{step:05d}.png')
            Image.fromarray(img_rgb).save(frame_path)
        
        if step == last_step:
            last_frame_path = os.path.join(out_dir, f'{basename}_last_frame.png')
            Image.fromarray(img_rgb).save(last_frame_path)

        plt.close(fig)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        out.write(img_bgr)

    out.release()
    print(f'Wrote video: {out_path}')
    return out_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', type=str, default='rollout/', required=True,
                   help='Directory containing result .pkl files (e.g. rollout/)')
    p.add_argument('--out-dir', type=str, default='videos', help='Output directory for videos')
    p.add_argument('--fps', type=int, default=None, help='Output video frames per second (default: match sim time)')
    p.add_argument('--size', nargs=2, type=int, default=[1920, 1080], help='Output video size: width height (pixels)')
    p.add_argument('--skip', type=int, default=1, help='Render every `skip` timesteps (skip=1 means every frame)')
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