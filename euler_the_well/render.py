"""
render.py

Render videos from Euler rollout results (.npz).
Produces a high-quality video with 8 panels.

Style: 
- High Resolution (1920x1080) & Simulation Matched FPS
- ROBUST SCALING: Color limits anchored to GT percentiles (2%-98%).
- DISTINCT COLORMAPS: Specific scientific colormaps for each variable.
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import cv2


def fig2data(fig):
    """
    Convert a Matplotlib figure to an HxWx4 RGBA numpy array.
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    try:
        buf.shape = (h, w, 4)
    except Exception:
        buf.shape = (w, h, 4)
        buf = buf.transpose((1, 0, 2))
    # ARGB -> RGBA
    buf = np.roll(buf, -1, axis=2)
    return buf


def compute_momentum_mag(mom):
    return np.sqrt(mom[..., 0]**2 + mom[..., 1]**2)


def resize_to_target(arr, target_hw):
    Hc, Wc = target_hw
    if arr.ndim == 2:
        return cv2.resize(arr, (Wc, Hc), interpolation=cv2.INTER_LINEAR)
    elif arr.ndim == 3:
        out = np.zeros((arr.shape[0], Hc, Wc), arr.dtype)
        for i in range(arr.shape[0]):
            out[i] = cv2.resize(arr[i], (Wc, Hc), interpolation=cv2.INTER_LINEAR)
        return out
    else:
        raise ValueError("Invalid array dimension for resize.")


def _load_from_npz(path):
    P = np.load(path)
    if "preds_d" not in P:
        raise KeyError(f"Prediction arrays not found in {path}.")

    p_d = P["preds_d"]
    p_e = P["preds_e"]
    p_p = P["preds_p"]
    p_m = P["preds_m"]

    if "gts_d" in P:
        g_d = P["gts_d"]
        g_e = P["gts_e"]
        g_p = P["gts_p"]
        g_m = P["gts_m"]
    else:
        _, Hc, Wc = p_d.shape
        g_d = np.zeros((0, Hc, Wc), dtype=p_d.dtype)
        g_e = np.zeros((0, Hc, Wc), dtype=p_e.dtype)
        g_p = np.zeros((0, Hc, Wc), dtype=p_p.dtype)
        g_m = np.zeros((0, Hc, Wc, 2), dtype=p_m.dtype)

    return {"preds": {"d": p_d, "e": p_e, "p": p_p, "m": p_m},
            "gts":   {"d": g_d, "e": g_e, "p": g_p, "m": g_m}}


def get_robust_limits(arr_main, arr_fallback=None, p_low=2, p_high=98):
    """
    Compute robust vmin/vmax using percentiles of the Ground Truth.
    Ignores extreme outliers (shockwaves) to keep bulk fluid visible.
    """
    if arr_main.size > 0:
        data = arr_main
    elif arr_fallback is not None and arr_fallback.size > 0:
        data = arr_fallback
    else:
        return 0.0, 1.0

    flat = data.ravel()
    vmin = float(np.percentile(flat, p_low))
    vmax = float(np.percentile(flat, p_high))
    
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
        
    return vmin, vmax


def render_file(npz_path, out_dir, fps=None, size=(1920, 1080), skip=1, 
                codec='mp4v', write_frames=False):
    
    basename = os.path.splitext(os.path.basename(npz_path))[0]
    out_path = os.path.join(out_dir, f'{basename}.mp4')

    # Load Data
    data = _load_from_npz(npz_path)
    
    p_d = data["preds"]["d"]
    p_e = data["preds"]["e"]
    p_p = data["preds"]["p"]
    p_m = data["preds"]["m"]
    p_mmag = compute_momentum_mag(p_m)
    
    g_d = data["gts"]["d"]
    g_e = data["gts"]["e"]
    g_p = data["gts"]["p"]
    g_m = data["gts"]["m"]

    p_T, Hc, Wc = p_d.shape

    if g_d.size > 0:
        g_mmag = compute_momentum_mag(g_m)
        if g_d.ndim == 3 and g_d.shape[1:] == (Hc, Wc):
            g_d_r, g_e_r, g_p_r, g_mmag_r = g_d, g_e, g_p, g_mmag
        else:
            g_d_r = resize_to_target(g_d, (Hc, Wc))
            g_e_r = resize_to_target(g_e, (Hc, Wc))
            g_p_r = resize_to_target(g_p, (Hc, Wc))
            g_mmag_r = resize_to_target(g_mmag, (Hc, Wc))
    else:
        g_d_r = g_e_r = g_p_r = g_mmag_r = np.zeros((0, Hc, Wc))

    T = min(p_T, g_d_r.shape[0]) if g_d_r.shape[0] > 0 else p_T
    
    p_d, p_e, p_p, p_mmag = p_d[:T], p_e[:T], p_p[:T], p_mmag[:T]
    g_d_r, g_e_r, g_p_r, g_mmag_r = g_d_r[:T], g_e_r[:T], g_p_r[:T], g_mmag_r[:T]

    # compute scales
    lims_d = get_robust_limits(g_d_r, p_d)
    lims_e = get_robust_limits(g_e_r, p_e)
    lims_m = get_robust_limits(g_mmag_r, p_mmag)
    lims_p = get_robust_limits(g_p_r, p_p)

    # fps
    dt = 0.015
    if fps is None:
        fps = 1.0 / (skip * dt)
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (int(size[0]), int(size[1])))

    frame_dir = None
    if write_frames:
        frame_dir = os.path.join(out_dir, f'{basename}_frames')
        os.makedirs(frame_dir, exist_ok=True)

    dpi = 100
    fig_w = size[0] / dpi
    fig_h = size[1] / dpi
    
    base_fs = 12 * (size[0] / 1920.0) 
    title_fs = base_fs * 1.2
    cbar_fs = base_fs * 0.9

    indices = list(range(0, T, skip))
    pbar = tqdm(indices, desc=f'Rendering {basename}', unit='frame')

    # Define per-variable colormaps
    # Index 0: Density, 1: Energy, 2: Momentum, 3: Pressure
    VAR_CMAPS = ["viridis", "magma", "plasma", "cividis"]

    for t in pbar:
        fig, axes = plt.subplots(2, 4, figsize=(fig_w, fig_h), dpi=dpi)
        fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.3, hspace=0.2)
        
        time_str = f"{t * dt:.2f} s"

        columns = [
            (p_d[t], g_d_r[t], lims_d, "Density"),
            (p_e[t], g_e_r[t], lims_e, "Energy"),
            (p_mmag[t], g_mmag_r[t], lims_m, "|Momentum|"),
            (p_p[t], g_p_r[t], lims_p, "Pressure")
        ]

        for col_idx, (p_data, g_data, (vmin, vmax), title) in enumerate(columns):
            current_cmap = VAR_CMAPS[col_idx]

            # Row 0: Prediction
            ax_p = axes[0, col_idx]
            im_p = ax_p.imshow(p_data, origin="lower", cmap=current_cmap, vmin=vmin, vmax=vmax, aspect='auto')
            ax_p.set_xticks([])
            ax_p.set_yticks([])
            if col_idx == 0: 
                ax_p.set_ylabel("Prediction", fontsize=title_fs, fontweight='bold')
            ax_p.set_title(f"Pred {title}\n{time_str}", fontsize=title_fs)

            # Row 1: Ground Truth
            ax_g = axes[1, col_idx]
            im_g = ax_g.imshow(g_data, origin="lower", cmap=current_cmap, vmin=vmin, vmax=vmax, aspect='auto')
            ax_g.set_xticks([])
            ax_g.set_yticks([])
            if col_idx == 0: 
                ax_g.set_ylabel("Ground Truth", fontsize=title_fs, fontweight='bold')
            ax_g.set_title(f"GT {title}", fontsize=title_fs)

            # Shared Colorbar
            cbar = fig.colorbar(im_g, ax=[ax_p, ax_g], orientation='horizontal', fraction=0.05, pad=0.05)
            cbar.ax.tick_params(labelsize=cbar_fs)

        img_rgba = fig2data(fig)
        plt.close(fig)

        img_rgb = img_rgba[:, :, :3]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        if img_bgr.shape[0] != size[1] or img_bgr.shape[1] != size[0]:
            img_bgr = cv2.resize(img_bgr, (size[0], size[1]))

        writer.write(img_bgr)

        if write_frames:
            frame_path = os.path.join(frame_dir, f'{basename}_frame_{t:05d}.png')
            Image.fromarray(img_rgb).save(frame_path)

    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Render Euler rollout results to high-quality video.")
    parser.add_argument("--rollout-dir", type=str, required=True,
                        help="Directory containing rollout .npz files.")
    parser.add_argument("--out-dir", type=str, default="videos",
                        help="Output directory for videos.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Frames per second. If not set, matches sim time.")
    parser.add_argument('--size', nargs=2, type=int, default=[1920, 1080], 
                        help='Output video size: width height (pixels)')
    parser.add_argument('--skip', type=int, default=1, 
                        help='Render every `skip` timesteps.')
    parser.add_argument('--codec', type=str, default='avc1', 
                        help='FourCC codec for cv2 VideoWriter (default avc1 for H.264)')
    parser.add_argument('--write-frames', action='store_true', 
                        help='Also save each frame as PNG')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted([
        f for f in os.listdir(args.rollout_dir) if f.endswith(".npz")
    ])

    if not files:
        print("No .npz rollout files found in:", args.rollout_dir)
        return

    print(f"Found {len(files)} rollout files.")

    for fname in files:
        path = os.path.join(args.rollout_dir, fname)
        try:
            render_file(path, args.out_dir, fps=args.fps, size=tuple(args.size),
                        skip=args.skip, codec=args.codec,
                        write_frames=args.write_frames)
        except Exception as e:
            print(f"Failed to render {fname}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
