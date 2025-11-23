import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional
import timm
import gigapath.slide_encoder
import pandas as pd
import torch
from huggingface_hub import login
from HistToSR_pipeline.run import run_HistToSR
from metrics.performance_eval import performance_summary


def _resolve_device(device_arg: str) -> int:
    normalized = str(device_arg).lower()
    if normalized.startswith("cuda:"):
        normalized = normalized.split(":", 1)[1]
    if normalized.isdigit():
        return int(normalized)
    raise ValueError(f"Unsupported device specification: {device_arg}")


def _assert_cuda(device_index: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but no GPU is available.")
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {device_index} requested but only "
            f"{torch.cuda.device_count()} device(s) detected."
        )


def _load_encoder(weight_path: Path, device_index: int) -> torch.nn.Module:
    model = torch.load(weight_path, map_location=torch.device(f"cuda:{device_index}"))
    return model.eval()


def _iter_target_folders(
    data_dir: Path, includes: Optional[Iterable[str]], blacklist: Iterable[str]
) -> Iterable[Path]:
    candidates: Iterable[Path]
    if includes:
        candidates = (data_dir / name for name in includes)
    else:
        candidates = (p for p in data_dir.iterdir() if p.is_dir())

    blacklist_set = {name.lower() for name in blacklist}
    for folder in candidates:
        if folder.is_dir() and folder.name.lower() not in blacklist_set:
            yield folder


def _summarize_metrics(result_dir: Path) -> Optional[pd.DataFrame]:
    csv_path = result_dir / "performance_metrics.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def run_pipeline(args: argparse.Namespace) -> None:

    from huggingface_hub import login
    import os
    login(os.getenv("HUGGINGFACE_HUB_TOKEN"))

    data_dir = Path(args.data_dir).resolve()
    tile_encoder_path = args.tile_encoder
    slide_encoder_path = args.slide_encoder

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    device_index = _resolve_device(args.device)
    _assert_cuda(device_index)

    logger = logging.getLogger("His2SR")
    logger.setLevel(logging.INFO if not args.verbose else logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)

    logger.info("Loading encoders once for all slides...")
    # tile_encoder = _load_encoder(tile_encoder_path, device_index)
    # slide_encoder = _load_encoder(slide_encoder_path, device_index)


    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False, checkpoint_path=tile_encoder_path)
    slide_encoder = gigapath.slide_encoder.create_model(slide_encoder_path, "gigapath_slide_enc12l768d", 1536)


    folders = list(
        _iter_target_folders(data_dir, args.folders, args.blacklist or [])
    )
    if not folders:
        logger.warning("No target folders found. Nothing to process.")
        return

    summary_rows: List[pd.Series] = []

    for folder in folders:
        logger.info("Processing folder %s", folder.name)
        run_HistToSR(
            str(folder),
            str(tile_encoder_path),
            str(slide_encoder_path),
            epochs=args.epochs,
            lr=args.lr,
            min_lr=args.min_lr,
            use_amp=args.amp,
            use_scheduler=not args.no_scheduler,
            tile_size=args.tile_size,
            stride=args.stride,
            device=device_index,
            tile_encoder=tile_encoder,
            slide_encoder=slide_encoder,
            loss_mode=args.loss_mode,
        )

        pred_path = folder / "result" / "His2SR" / "Hist2SRpred.pickle"
        gt_path = folder / "ground_truth" / "ground_truth.pickle"
        if pred_path.exists() and gt_path.exists():
            performance_summary(str(pred_path), str(gt_path), str(folder))

        metrics_df = _summarize_metrics(folder / "result" / "His2SR")
        metrics_df = metrics_df.set_index(metrics_df.columns[0])
        
        if metrics_df is not None:
            mean_values = metrics_df.mean()
            summary_rows.append(pd.Series(mean_values, name=folder.name))
            logger.info(
                "Metrics | RMSE: %.4f | SSIM: %.4f | PCC: %.4f",
                mean_values.get("RMSE", float("nan")),
                mean_values.get("SSIM", float("nan")),
                mean_values.get("PCC", float("nan")),
            )
        else:
            logger.warning("Metrics CSV not found for folder %s", folder.name)

    if summary_rows:
        summary_table = pd.DataFrame(summary_rows)
        logger.info("\n%s", summary_table.to_string(float_format="%.4f"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hist2SR predictions across slides.")
    parser.add_argument(
        "--data-dir",
        default="./Demo_data",
        help="Directory containing slide folders.",
    )
    parser.add_argument(
        "--tile-encoder",
        default="./Gigapath_encoders/pytorch_model_compelete.bin",
        help="Path to the pretrained tile encoder weights.",
    )
    parser.add_argument(
        "--slide-encoder",
        default="./Gigapath_encoders/slide_encoder_complete.pth",
        help="Path to the pretrained slide encoder weights.",
    )
    parser.add_argument(
        "--folders",
        nargs="*",
        help="Specific folders (names) to process. Defaults to all folders under data-dir.",
    )
    parser.add_argument(
        "--blacklist",
        nargs="*",
        default=[],
        help="Folder names to skip.",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=56)
    parser.add_argument(
        "--loss-mode",
        choices=["MSE", "MAE", "RMSE"],
        default="MSE",
        help="Loss function for predictor training.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device index.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision during training.",
    )
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Disable the cosine learning rate scheduler.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
