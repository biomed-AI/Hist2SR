# Hist2SR

Hist2SR predicts spatial transcriptomics profiles from histopathology slides.

## Requirements

- Python 3.9 or newer
- CUDA-enabled GPU (the default workflow targets `cuda:0`)
- PyTorch, torchvision, timm, pandas, numpy, einops, tqdm, scikit-image, opencv-python, matplotlib



## Data and Checkpoints

Each slide folder must contain:

- `HE_raw.jpg`
- `pixel-size-raw.txt`
- `radius-raw.txt`
- `locs-raw.tsv`
- `cnts.tsv`

The pipeline creates additional artifacts (`HE_scaled_pad{tile_size}.jpg`, `save_embedding/`, `save_predictor/`, `result/His2SR/`).

Pretrained encoders are expected under `Gigapath_encoders/`:

- `pytorch_model_compelete.bin` (tile encoder)
- `slide_encoder_complete.pth` (slide encoder)

Adjust paths as needed when running the CLI.

## Usage

Run the end-to-end pipeline with the new command-line interface:

```bash
python run.py \
  --data-dir ./Demo_data \
  --tile-encoder ./Gigapath_encoders/pytorch_model_compelete.bin \
  --slide-encoder ./Gigapath_encoders/slide_encoder_complete.pth \
  --device 0 \
  --epochs 400 \
  --lr 5e-4 \
  --min-lr 1e-5\
  --tile-size 224 \
  --stride 56
  --loss-mode MSE
```

### Key Options

- `--folders name1 name2` process only specific subdirectories.
- `--blacklist name` skip selected subdirectories.
- `--stride`, `--tile-size`, `--lr`, `--min-lr`, `--epochs` tune preprocessing and training.
- `--loss-mode {MSE,MAE,RMSE}` select the regression loss for predictor training.
- `--amp` enable automatic mixed precision.
- `--no-scheduler` disable cosine LR scheduling.
- `--verbose` increases log verbosity.

## Pipeline Steps

1. **Preparation** – rescale the H&E image, prepare spot coordinates, validate required files.
2. **Embedding Extraction** – reuse preloaded encoders to generate tile and slide embeddings (saved to `save_embedding`).
3. **Training** – train the predictor (KAN backbone) with cosine scheduling and optional AMP.
4. **Inference** – predict per-cell gene expression and reshape into heatmaps (`Hist2SRpred.pickle`).
5. **Visualization & Metrics** – render heatmaps and compute RMSE/SSIM/PCC (stored in `result/His2SR/performance_metrics.csv`).

## Outputs

- `save_embedding/Embed_training.pt` and `Embed_inference.pt`
- `save_predictor/predictor.pth`
- `result/His2SR/Hist2SRpred.pickle`
- `result/His2SR/performance_metrics.csv` with averaged metrics logged to console

