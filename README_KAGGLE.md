# SL-ACC on Kaggle

## 1) Enable runtime settings

- Enable GPU in Kaggle Notebook settings.
- Enable Internet if cloning from GitHub or downloading MNIST/CIFAR-100.

## 2) Attach HAM10000

- Attach HAM10000 as a Kaggle input dataset.
- Set Hydra overrides if paths differ from defaults:
  - `dataset.ham10000.image_dir=/kaggle/input/<dataset>/HAM10000_images`
  - `dataset.ham10000.metadata_csv=/kaggle/input/<dataset>/HAM10000_metadata.csv`

## 3) Clone and install

```bash
git clone https://github.com/<USER>/sl_acc_reproduction.git /kaggle/working/sl_acc_reproduction || true
cd /kaggle/working/sl_acc_reproduction
git pull
pip install -e .
pip install -r requirements-kaggle.txt
```

## 4) Run smoke

```bash
python -m experiments.train profile=kaggle_smoke
```

Quick tests:

```bash
bash scripts/run_tests_quick.sh
```

## 5) Run one shard

```bash
python -m experiments.train profile=kaggle_single_full dataset=ham10000 distribution=iid method=sl_acc seed=0
```

Run one matrix shard:

```bash
python -m experiments.train profile=kaggle_matrix_shard dataset=mnist distribution=iid method=sl_acc seed=0 client.num_clients=5
```

## 6) Resume

```bash
python -m experiments.train profile=kaggle_resume resume=true resume_path=/kaggle/working/outputs/checkpoints/latest.pt
```

## 7) Zip outputs

```bash
bash scripts/zip_outputs.sh
```

## Notes

- W&B is optional and disabled by default in Kaggle profiles.
- Local logs/checkpoints are always written to `/kaggle/working/outputs`.
- Optional W&B enable:
  - `tracking.wandb_enabled=true tracking.tracker=wandb`
  - Use Kaggle Secrets for `WANDB_API_KEY`.
  - If internet is disabled, set `WANDB_MODE=offline`.
