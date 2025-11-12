Convert All Image — UNet Segmentation (Clean Layout)

Overview
- Source under `src/convert_all_image` with modules for `data`, `models`, and `utils`.
- Config in `config/config.yaml` (lists of image and mask directories).
- Simple UNet and dataset for multi-class semantic segmentation.

Install
- Python 3.9+
- Optional: create a venv
  - `python -m venv .venv && .\.venv\Scripts\activate` (Windows)
  - `python -m venv .venv && source .venv/bin/activate` (Linux/macOS)
- `pip install -r requirements.txt`

Train
- Edit `config/config.yaml` to point to your image/mask dirs.
- Ensure Python sees the `src` folder (one-time per shell):
  - PowerShell: `$env:PYTHONPATH = "$(Get-Location)\src"`
  - bash/zsh: `export PYTHONPATH="$(pwd)/src"`
- Run:
  - `python -m convert_all_image.train --config config/config.yaml --epochs 30 --batch-size 8 --lr 1e-4 --num-classes 4 --img-size 256 256`

Infer
- Run with a trained model (or untrained for a sanity check):
  - `python -m convert_all_image.inference --image path/to/image.png --num-classes 4 --img-size 256 256 --weights path/to/model.pt`

Notes
- The dataset pairs images and masks by sorted filename order. Ensure your folders are aligned (e.g., same counts and relative ordering). For more complex naming schemes, adapt `SegmentationDataset` to map pairs by stem.
- Masks are treated as integer class indices; resizing uses nearest-neighbor to preserve labels.
