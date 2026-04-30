"""
main.py
-------
Punto de entrada de bark-segmentation.

Flujo completo:
    1. Exporta troncos desde Blender como .ply con materiales asignados
    2. Coloca los .ply en data/raw/
    3. python main.py info          -> verifica los PLY y muestra estadisticas
    4. python main.py train         -> entrena PointNet++
    5. python main.py infer --input data/raw/tronco_nuevo.ply -> segmenta

Comandos:
    info        Analiza los .ply en data/raw y muestra estadisticas
    train       Entrena el modelo
    infer       Segmenta un tronco nuevo
    visualize   Visualiza nube de puntos con labels
    preprocess  (Opcional) Pre-procesa PLY a .npy para acelerar entrenamiento
"""

import argparse
import sys
from pathlib import Path


def cmd_info(args):
    """Analiza los PLY disponibles y muestra estadisticas."""
    from config.config_loader import load_config
    from data.loader import scan_ply_folder, load_ply_labeled
    import numpy as np

    cfg      = load_config(args.config)
    raw_dir  = Path(cfg.paths.raw_data)
    ply_files = scan_ply_folder(raw_dir)

    if not ply_files:
        print(f"No se encontraron archivos .ply en: {raw_dir}")
        print("Exporta tus troncos desde Blender y colocalos en esa carpeta.")
        return

    print(f"\nArchivos PLY encontrados en {raw_dir}: {len(ply_files)}\n")
    print(f"  {'Archivo':<35} {'Verts':>6} {'Corteza':>8} {'Madera':>8} {'%Corteza':>9} {'Labels':>7}")
    print("  " + "─" * 77)

    total_verts = 0
    total_bark  = 0
    unlabeled   = []

    for f in ply_files:
        try:
            _, labels, meta = load_ply_labeled(f, compute_normals=False)
            has_label = "SI" if meta["n_bark"] > 0 else "NO"
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
        print(f"\n  ATENCION: {len(unlabeled)} archivo(s) sin labels:")
        for u in unlabeled:
            print(f"    - {u}")
        print("  Asigna materiales en Blender y re-exporta estos troncos.")

    labeled_count = len(ply_files) - len(unlabeled)
    print(f"\n  Listos para entrenar: {labeled_count}/{len(ply_files)} troncos")
    if labeled_count < 2:
        print("  Se necesitan al menos 2 troncos con labels para entrenar.")
    else:
        print("  Ejecuta: python main.py train")


def cmd_train(args):
    """Entrena el modelo PointNet++."""
    from config.config_loader import load_config
    from data.loader import scan_ply_folder, load_ply_labeled
    from training.trainer import train

    cfg = load_config(args.config)

    # Verificar datos disponibles
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
        print(f"Solo {len(labeled)} tronco(s) con labels. Necesitas al menos 2.")
        print("Ejecuta: python main.py info  para ver el estado actual.")
        return

    print(f"Iniciando entrenamiento con {len(labeled)} troncos con labels validos.")
    train(cfg, resume=args.resume)


def cmd_infer(args):
    """Segmenta un tronco nuevo."""
    from config.config_loader import load_config
    from inference.predictor import BarkPredictor

    if not args.input:
        print("Especifica el archivo con --input <ruta.ply>")
        return

    cfg        = load_config(args.config)
    model_path = args.model or cfg.inference.model_path

    if not Path(model_path).exists():
        print(f"Modelo no encontrado: {model_path}")
        print("Entrena primero: python main.py train")
        return

    predictor = BarkPredictor.from_checkpoint(model_path)
    predictor.predict_ply(
        ply_path=args.input,
        save_ply=not args.no_ply,
        output_dir=cfg.inference.output_dir,
        visualize=args.visualize,
    )


def cmd_visualize(args):
    """Visualiza una nube de puntos con sus labels."""
    from config.config_loader import load_config
    from data.loader import load_ply_labeled
    from utils.visualizer import visualize_segmentation, visualize_cloud

    if not args.file:
        print("Especifica un archivo con --file <ruta.ply>")
        return

    cfg  = load_config(args.config)
    path = Path(args.file)
    if not path.exists():
        path = Path(cfg.paths.raw_data) / args.file

    cloud, labels, meta = load_ply_labeled(path, compute_normals=False)
    pts = cloud[:, :3]

    print(f"\n{path.name}")
    print(f"  Vertices: {meta['n_vertices']}")
    print(f"  Corteza:  {meta['n_bark']} ({meta['bark_fraction']*100:.1f}%)")
    print(f"  Madera:   {meta['n_wood']}")
    print(f"  Dims(mm): {[f'{d:.0f}' for d in meta['dimensions_mm']]}")

    if meta["n_bark"] > 0:
        visualize_segmentation(pts, labels, title=path.stem)
    else:
        visualize_cloud(pts, title=f"{path.stem} (sin labels)")


def cmd_preprocess(args):
    """Pre-procesa PLY a .npy para acelerar entrenamiento."""
    from config.config_loader import load_config
    from preprocessing.sampler import preprocess_dataset

    cfg = load_config(args.config)
    preprocess_dataset(cfg, overwrite=args.overwrite)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="bark-segmentation",
        description="Segmentacion 3D de corteza remanente en troncos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config/default.yaml")
    sub = parser.add_subparsers(dest="command", metavar="COMANDO")

    sub.add_parser("info", help="Analizar PLY disponibles y ver estadisticas")

    p_train = sub.add_parser("train", help="Entrenar modelo PointNet++")
    p_train.add_argument("--resume", action="store_true", help="Reanudar desde checkpoint")

    p_inf = sub.add_parser("infer", help="Segmentar tronco nuevo")
    p_inf.add_argument("--input",     help="Ruta al .ply a segmentar")
    p_inf.add_argument("--model",     default=None)
    p_inf.add_argument("--visualize", action="store_true")
    p_inf.add_argument("--no-ply",    action="store_true")

    p_vis = sub.add_parser("visualize", help="Visualizar nube con labels")
    p_vis.add_argument("--file", required=True)

    p_pre = sub.add_parser("preprocess", help="(Opcional) PLY -> .npy")
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
