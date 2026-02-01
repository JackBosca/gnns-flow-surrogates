"""
Rendering script for Flux Conservative GNN Euler gases report images and videos.
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


def render_file(npz_path, out_dir, fps=None, size=(1600, 1600), skip=1, 
                codec='mp4v', write_frames=False):
    
    basename = os.path.splitext(os.path.basename(npz_path))[0]
    out_path = os.path.join(out_dir, f'{basename}.mp4')

    # 1.Load Data
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
            g_m_x_r = g_m[..., 0]
            g_m_y_r = g_m[..., 1]
        else:
            g_d_r = resize_to_target(g_d, (Hc, Wc))
            g_e_r = resize_to_target(g_e, (Hc, Wc))
            g_p_r = resize_to_target(g_p, (Hc, Wc))
            g_mmag_r = resize_to_target(g_mmag, (Hc, Wc))
            g_m_x_r = resize_to_target(g_m[..., 0], (Hc, Wc))
            g_m_y_r = resize_to_target(g_m[..., 1], (Hc, Wc))
    else:
        g_d_r = g_e_r = g_p_r = g_mmag_r = np.zeros((0, Hc, Wc))
        g_m_x_r = g_m_y_r = np.zeros((0, Hc, Wc))

    T = min(p_T, g_d_r.shape[0]) if g_d_r.shape[0] > 0 else p_T
    
    p_d, p_e, p_p, p_mmag = p_d[:T], p_e[:T], p_p[:T], p_mmag[:T]
    p_m = p_m[:T]
    g_d_r, g_e_r, g_p_r, g_mmag_r = g_d_r[:T], g_e_r[:T], g_p_r[:T], g_mmag_r[:T]
    g_m_x_r, g_m_y_r = g_m_x_r[:T], g_m_y_r[:T]

    # compute errors and scales
    err_d = np.abs(g_d_r - p_d)
    err_e = np.abs(g_e_r - p_e)
    err_p = np.abs(g_p_r - p_p)
    err_m = np.sqrt((g_m_x_r - p_m[...,0])**2 + (g_m_y_r - p_m[...,1])**2)

    lims_d = get_robust_limits(g_d_r, p_d)
    lims_e = get_robust_limits(g_e_r, p_e)
    lims_m = get_robust_limits(g_mmag_r, p_mmag)
    lims_p = get_robust_limits(g_p_r, p_p)
    
    _, vmax_err_d = get_robust_limits(err_d, p_high=98)
    _, vmax_err_e = get_robust_limits(err_e, p_high=98)
    _, vmax_err_m = get_robust_limits(err_m, p_high=98)
    _, vmax_err_p = get_robust_limits(err_p, p_high=98)
    
    lims_err_d = (0.0, vmax_err_d)
    lims_err_e = (0.0, vmax_err_e)
    lims_err_m = (0.0, vmax_err_m)
    lims_err_p = (0.0, vmax_err_p)

    # FPS and setup
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
    
    title_fs = 60
    cbar_label_fs = 60
    cbar_err_label_fs = 48
    cbar_tick_fs = 44
    
    indices = list(range(0, T, skip))
    pbar = tqdm(indices, desc=f'Rendering {basename}', unit='frame')

    VAR_CMAPS = ["viridis", "magma", "plasma", "cividis"]

    for t in pbar:
        fig, axes = plt.subplots(3, 4, figsize=(fig_w, fig_h), dpi=dpi)
        
        fig.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.1, wspace=0.15, hspace=0.15)
        
        columns = [
            (g_d_r[t], p_d[t], err_d[t], lims_d, lims_err_d, "Density [-]"),
            (g_e_r[t], p_e[t], err_e[t], lims_e, lims_err_e, "Energy [-]"),
            (g_mmag_r[t], p_mmag[t], err_m[t], lims_m, lims_err_m, r"$\|$Momentum$\|$ [-]"),
            (g_p_r[t], p_p[t], err_p[t], lims_p, lims_err_p, "Pressure [-]")
        ]

        for col_idx, (g_data, p_data, err_data, (vmin, vmax), (emin, emax), cbar_title) in enumerate(columns):
            current_cmap = VAR_CMAPS[col_idx]

            # --- Row 0: Target ---
            ax_g = axes[0, col_idx]
            im_g = ax_g.imshow(g_data, origin="lower", cmap=current_cmap, vmin=vmin, vmax=vmax, aspect='equal')
            ax_g.axis('off') 
            if col_idx == 0: 
                ax_g.axis('on')
                ax_g.set_xticks([])
                ax_g.set_yticks([])
                for spine in ax_g.spines.values(): spine.set_visible(False)
                ax_g.set_ylabel("Target", fontsize=title_fs, fontweight='bold')

            # --- Row 1: Prediction ---
            ax_p = axes[1, col_idx]
            im_p = ax_p.imshow(p_data, origin="lower", cmap=current_cmap, vmin=vmin, vmax=vmax, aspect='equal')
            ax_p.axis('off')
            if col_idx == 0: 
                ax_p.axis('on')
                ax_p.set_xticks([])
                ax_p.set_yticks([])
                for spine in ax_p.spines.values(): spine.set_visible(False)
                ax_p.set_ylabel("Prediction", fontsize=title_fs, fontweight='bold')

            # --- Row 2: Error ---
            ax_e = axes[2, col_idx]
            im_e = ax_e.imshow(err_data, origin="lower", cmap=current_cmap, vmin=emin, vmax=emax, aspect='equal')
            ax_e.axis('off')
            if col_idx == 0: 
                ax_e.axis('on')
                ax_e.set_xticks([])
                ax_e.set_yticks([])
                for spine in ax_e.spines.values(): spine.set_visible(False)
                ax_e.set_ylabel("Error", fontsize=title_fs, fontweight='bold')
            
            # Main clorbar (Rows 0 & 1)
            cbar_main = fig.colorbar(im_g, ax=[ax_g, ax_p], orientation='horizontal', fraction=0.046, pad=0.04)
            cbar_main.set_label(cbar_title, fontsize=cbar_label_fs)
            cbar_main.ax.tick_params(labelsize=cbar_tick_fs)

            # Error colorbar (Row 2)
            cbar_err = fig.colorbar(im_e, ax=ax_e, orientation='horizontal', fraction=0.046, pad=0.04)
            
            err_title = cbar_title.replace(" [-]", " Err [-]")
            cbar_err.set_label(err_title, fontsize=cbar_err_label_fs)
            
            cbar_err.ax.tick_params(labelsize=cbar_tick_fs)
            cbar_err.ax.locator_params(nbins=4)

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
        
        if t == 30:
            snapshot_path = os.path.join(out_dir, f'{basename}_frame_30.png')
            Image.fromarray(img_rgb).save(snapshot_path)
            print(f"Saved snapshot at step {t}: {snapshot_path}")
        
        if t == indices[-1]:
            # t is the last one
            snapshot_path = os.path.join(out_dir, f'{basename}_frame_final.png')
            Image.fromarray(img_rgb).save(snapshot_path)
            print(f"Saved final snapshot at step {t}: {snapshot_path}")

    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Render Euler rollout results.")
    parser.add_argument("--rollout-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="videos")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument('--size', nargs=2, type=int, default=[3200, 3200])
    parser.add_argument('--skip', type=int, default=1)
    parser.add_argument('--codec', type=str, default='avc1')
    parser.add_argument('--write-frames', action='store_true')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(args.rollout_dir) if f.endswith(".npz")])

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