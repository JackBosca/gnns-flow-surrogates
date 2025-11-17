# render.py: Euler rollout renderer
# Produces a single video with 8 panels:
# (Pred: density, energy, |momentum|, pressure)
# (Ground Truth: density, energy, |momentum|, pressure)
import numpy as np
import matplotlib.pyplot as plt
import cv2


def compute_momentum_mag(mom):
    # mom: (T, H, W, 2)
    return np.sqrt(mom[..., 0]**2 + mom[..., 1]**2)


def resize_to_target(arr, target_hw):
    """Resize numpy 2D or 3D arrays to (Hc, Wc)."""
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
    """
    Load the arrays from a rollout .npz.
    Returns dict with keys:
      preds: dict with density, energy, pressure, momentum
      gts:   dict with density, energy, pressure, momentum
    """
    P = np.load(path)
    # predictions
    if "preds_d" in P:
        p_d = P["preds_d"]
        p_e = P["preds_e"]
        p_p = P["preds_p"]
        p_m = P["preds_m"]
    else:
        raise KeyError(f"Prediction arrays not found in {path}.")

    # ground truths
    if "gts_d" in P:
        g_d = P["gts_d"]
        g_e = P["gts_e"]
        g_p = P["gts_p"]
        g_m = P["gts_m"]
    else:
        # create empty arrays shaped to preds (so rendering still works)
        _, Hc, Wc = p_d.shape
        g_d = np.zeros((0, Hc, Wc), dtype=p_d.dtype)
        g_e = np.zeros((0, Hc, Wc), dtype=p_e.dtype)
        g_p = np.zeros((0, Hc, Wc), dtype=p_p.dtype)
        g_m = np.zeros((0, Hc, Wc, 2), dtype=p_m.dtype)

    return {"preds": {"d": p_d, "e": p_e, "p": p_p, "m": p_m},
            "gts":   {"d": g_d, "e": g_e, "p": g_p, "m": g_m}}


def render_file(npz_file, out_path="rollout.mp4", fps=25):
    """
    npz_file: .npz produced by rollout_one_simulation(...) containing:
        preds_* arrays and gts_* arrays (see rollout script)
    """
    data = _load_from_npz(npz_file)
    p_d = data["preds"]["d"]         # (T, Hc, Wc)
    p_e = data["preds"]["e"]
    p_p = data["preds"]["p"]
    p_m = data["preds"]["m"]         # (T, Hc, Wc, 2)
    p_T = p_d.shape[0]
    _, Hc, Wc = p_d.shape

    p_mmag = compute_momentum_mag(p_m)  # (T, Hc, Wc)

    g_d = data["gts"]["d"]            # (T_gt, H_gt, W_gt) or (T_gt, Hc, Wc)
    g_e = data["gts"]["e"]
    g_p = data["gts"]["p"]
    g_m = data["gts"]["m"]            # (T_gt, H_gt, W_gt, 2)

    # determine if GT needs resizing to match coarse preds
    if g_d.ndim == 3 and g_d.shape[1:] == (Hc, Wc):
        g_d_r = g_d
        g_e_r = g_e
        g_p_r = g_p
        g_mmag_r = compute_momentum_mag(g_m)
    else:
        # resize GT to match coarse grid (Hc, Wc)
        g_d_r = resize_to_target(g_d, (Hc, Wc))
        g_e_r = resize_to_target(g_e, (Hc, Wc))
        g_p_r = resize_to_target(g_p, (Hc, Wc))
        g_mmag = compute_momentum_mag(g_m)
        g_mmag_r = resize_to_target(g_mmag, (Hc, Wc))

    # ensure all arrays have same length T
    T = min(p_T, g_d_r.shape[0])
    p_d = p_d[:T]
    p_e = p_e[:T]
    p_p = p_p[:T]
    p_mmag = p_mmag[:T]

    g_d_r = g_d_r[:T]
    g_e_r = g_e_r[:T]
    g_p_r = g_p_r[:T]
    g_mmag_r = g_mmag_r[:T]

    # set up video writer (8 panels arranged in 2×4 grid)
    panel_h, panel_w = Hc, Wc

    # final video dimensions = 4*Wc wide, 2*Hc tall
    frame_h = 2 * panel_h
    frame_w = 4 * panel_w

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h)
    )

    # render each frame
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    plt.set_cmap("viridis")

    for t in range(T):
        # build 8 panels as images
        panels = [
            p_d[t], p_e[t], p_mmag[t], p_p[t],     # predictions
            g_d_r[t], g_e_r[t], g_mmag_r[t], g_p_r[t]   # ground truth
        ]

        for ax, img in zip(axes.flat, panels):
            ax.clear()
            ax.imshow(img, origin="lower", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])

        axes[0, 0].set_title("Pred density")
        axes[0, 1].set_title("Pred energy")
        axes[0, 2].set_title("|Pred momentum|")
        axes[0, 3].set_title("Pred pressure")

        axes[1, 0].set_title("GT density")
        axes[1, 1].set_title("GT energy")
        axes[1, 2].set_title("|GT momentum|")
        axes[1, 3].set_title("GT pressure")

        # draw canvas to array (use tostring_argb -> convert ARGB to RGB)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (h, w, 4)                    # (H, W, 4) in ARGB order
        buf = np.roll(buf, -1, axis=2)           # ARGB -> RGBA (move A to last)
        img = buf[:, :, :3].copy()               # take RGB channels (uint8)

        # resize the rendered matplotlib frame to final video size
        img = cv2.resize(img, (frame_w, frame_h), interpolation=cv2.INTER_AREA)

        # convert RGB to BGR for OpenCV
        img = img[:, :, ::-1]

        writer.write(img)

    writer.release()
    plt.close(fig)

    print(f"\nSaved video to {out_path}\n")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Render Euler rollout results to video(s).")
    parser.add_argument("--rollout-dir", type=str, required=True,
                        help="Directory containing rollout .npz files.")
    parser.add_argument("--out-dir", type=str, default="videos",
                        help="Where to save rendered mp4 videos.")
    parser.add_argument("--fps", type=int, default=25,
                        help="Frames per second for output video.")
    args = parser.parse_args()

    # ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # list rollout files
    files = sorted(
        f for f in os.listdir(args.rollout_dir)
        if f.endswith(".npz")
    )

    if len(files) == 0:
        print("No .npz rollout files found in:", args.rollout_dir)
        exit(0)

    print(f"Found {len(files)} rollout files. Rendering...")

    for fname in files:
        path = os.path.join(args.rollout_dir, fname)
        try:
            print(f"\nRendering {fname} ...")
            out_path = os.path.join(args.out_dir, fname.replace(".npz", ".mp4"))

            render_file(npz_file=path, out_path=out_path, fps=args.fps)

            print(f"Saved video to {out_path}")

        except Exception as e:
            print(f"Failed to render {fname}: {e}")
