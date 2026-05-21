"""
main.py
-------
Entry point for bark-segmentation.

Full workflow:
    1. Export logs from Blender as .ply with assigned materials
    2. Place .ply files in data/raw/
    3. python main.py info          -> check PLY files and show statistics
    4. python main.py train         -> train PointNet++
    5. python main.py infer --input data/raw/new_log.ply -> segment

Commands:
    info        Analyze .ply files in data/raw and show statistics
    train       Train the model
    infer       Segment a new log
    visualize   Visualize point cloud with labels
    preprocess  (Optional) Pre-process PLY to .npy for faster training
"""

import argparse
import sys
from pathlib import Path


def cmd_info(args):
    """Analyzes available PLY files and shows statistics."""
    from config.config_loader import load_config
    from data.loader import scan_ply_folder, load_ply_labeled
    import numpy as np

    cfg      = load_config(args.config)
    raw_dir  = Path(cfg.paths.raw_data)
    ply_files = scan_ply_folder(raw_dir)

    if not ply_files:
        print(f"No .ply files found in: {raw_dir}")
        print("Export your logs from Blender and place them in that folder.")
        return

    print(f"\nPLY files found in {raw_dir}: {len(ply_files)}\n")
    print(f"  {'File':<35} {'Verts':>6} {'Bark':>8} {'Wood':>8} {'%Bark':>9} {'Labels':>7}")
    print("  " + "─" * 77)

    total_verts = 0
    total_bark  = 0
    unlabeled   = []

    for f in ply_files:
        try:
            _, labels, meta = load_ply_labeled(f, compute_normals=False)
            has_label = "YES" if meta["n_bark"] > 0 else "NO"
            pct = meta["bark_fraction"] * 100
            print(f"  {f.name:<35} {meta['n_vertices']:>6} "
                  f"{meta['n_bark']:>8} {meta['n_wood']:>8} {pct:>8.1f}% {has_label:>7}")
            total_verts += meta["n_vertices"]
            total_bark  += meta["n_bark"]
            if meta["n_bark"] == 0:
                unlabeled.append(f.name)
        except Exception as e:
            print(f"  {f.name:<35} [ERROR: {e}]")

    print("  " + "─" * 77)
    pct_total = total_bark / max(total_verts, 1) * 100
    print(f"  {'TOTAL':<35} {total_verts:>6} {total_bark:>8} {total_verts-total_bark:>8} {pct_total:>8.1f}%")

    if unlabeled:
        print(f"\n  WARNING: {len(unlabeled)} file(s) without labels:")
        for u in unlabeled:
            print(f"    - {u}")
        print("  Assign materials in Blender and re-export these logs.")

    labeled_count = len(ply_files) - len(unlabeled)
    print(f"\n  Ready to train: {labeled_count}/{len(ply_files)} logs")
    if labeled_count < 2:
        print("  At least 2 logs with labels are needed to train.")
    else:
        print("  Run: python main.py train")


def cmd_train(args):
    """Trains the PointNet++ model."""
    from config.config_loader import load_config
    from data.loader import scan_ply_folder, load_ply_labeled
    from training.trainer import train

    cfg = load_config(args.config)

    # check available labeled files
    raw_dir   = Path(cfg.paths.raw_data)
    ply_files = scan_ply_folder(raw_dir)
    labeled   = []
    for f in ply_files:
        try:
            _, _, meta = load_ply_labeled(f, compute_normals=False)
            if meta["n_bark"] > 0 or meta["n_wood"] > 0:
                labeled.append(f)
        except:
            pass

    if len(labeled) < 2:
        print(f"Only {len(labeled)} log(s) with labels. Need at least 2.")
        print("Run: python main.py info  to see current status.")
        return

    print(f"Starting training with {len(labeled)} logs with valid labels.")
    train(cfg, resume=args.resume)


def cmd_infer(args):
    """Segments a new log."""
    from config.config_loader import load_config
    from inference.predictor import BarkPredictor

    if not args.input:
        print("Specify the file with --input <path.ply>")
        return

    cfg        = load_config(args.config)
    model_path = args.model or cfg.inference.model_path

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Train first: python main.py train")
        return

    predictor = BarkPredictor.from_checkpoint(model_path)
    predictor.predict_ply(
        ply_path=args.input,
        save_ply=not args.no_ply,
        output_dir=cfg.inference.output_dir,
        visualize=args.visualize,
    )


def cmd_visualize(args):
    """Visualizes a point cloud with its labels."""
    from config.config_loader import load_config
    from data.loader import load_ply_labeled
    from utils.visualizer import visualize_segmentation, visualize_cloud

    if not args.file:
        print("Specify a file with --file <path.ply>")
        return

    cfg  = load_config(args.config)
    path = Path(args.file)
    if not path.exists():
        path = Path(cfg.paths.raw_data) / args.file

    cloud, labels, meta = load_ply_labeled(path, compute_normals=False)
    pts = cloud[:, :3]

    print(f"\n{path.name}")
    print(f"  Vertices: {meta['n_vertices']}")
    print(f"  Bark:     {meta['n_bark']} ({meta['bark_fraction']*100:.1f}%)")
    print(f"  Wood:     {meta['n_wood']}")
    print(f"  Dims(mm): {[f'{d:.0f}' for d in meta['dimensions_mm']]}")

    if meta["n_bark"] > 0:
        visualize_segmentation(pts, labels, title=path.stem)
    else:
        visualize_cloud(pts, title=f"{path.stem} (no labels)")


def cmd_preprocess(args):
    """Pre-processes PLY to .npy for faster training."""
    from config.config_loader import load_config
    from preprocessing.sampler import preprocess_dataset

    cfg = load_config(args.config)
    preprocess_dataset(cfg, overwrite=args.overwrite)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="bark-segmentation",
        description="3D residual bark segmentation on debarked logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config/default.yaml")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    sub.add_parser("info", help="Analyze available PLY files and show statistics")

    p_train = sub.add_parser("train", help="Train PointNet++ model")
    p_train.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    p_inf = sub.add_parser("infer", help="Segment a new log")
    p_inf.add_argument("--input",     help="Path to .ply file to segment")
    p_inf.add_argument("--model",     default=None)
    p_inf.add_argument("--visualize", action="store_true")
    p_inf.add_argument("--no-ply",    action="store_true")

    p_vis = sub.add_parser("visualize", help="Visualize point cloud with labels")
    p_vis.add_argument("--file", required=True)

    p_pre = sub.add_parser("preprocess", help="(Optional) PLY -> .npy")
    p_pre.add_argument("--overwrite", action="store_true")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "info":       cmd_info,
        "train":      cmd_train,
        "infer":      cmd_infer,
        "visualize":  cmd_visualize,
        "preprocess": cmd_preprocess,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
