# ResNet teaser runs (Paper I, Sec. 6.1 and Fig. 3)

These scripts train a ResNet amplitude regressor on noise drawn from the paper's registry, then compare its error variance with the matched filter and with the η reference lines. There are two cells:

| cell | registry model | prior on A | σ_MF (exact) | design ratio r = √12 σ_MF / L |
|---|---|---|---|---|
| theorem cell | `T0_RED_REAL` (red + white, Gaussian: η = 1) | U[0, 5] | 0.8630 | 0.60 |
| mixture cell | `T1_PSRAND_WN` (spectral-tilt mixture + floor: η = 10.2, extended template) | U[0, 7] | 1.3285 | 0.66 |

The run list also contains an optional white-noise cell (`T0_WHITE`, σ_w = 9, r = 0.75). It is not used in the paper, and its checkpoints are not released.

## Files

| file | role |
|---|---|
| `make_teaser_dataset.py` | writes the TRAIN file (A·τ_ext + registry noise; 30 000 maps) and the noise-only EVAL file (6000 maps: a 2000-map PSD split + 4000 EVAL maps). Seeds are registry seed + 103000 (train) and + 213000 (eval). Needs numpy/scipy/h5py only. |
| `train_teaser_ddp.py`, `utils_teaser_ddp.py` | ResNet training: single device (`cuda` / `mps` / `cpu`), or multi-GPU DDP under `torchrun`. Losses `mse` and `bias_corrected_mse` (λ = 15, 8 bins over [0, L]), plus the experimental `ema_bias_corrected_mse`. Float32 data path with exact D4 augmentation. The checkpoint stores the normalization, dataset attributes and seeds. |
| `teaser_eval.py` | amplitude-grid evaluation of a checkpoint on the EVAL maps (common random numbers across the grid): bias g(A), variance V(A), the certified locally unbiased band, the shrinkage-corrected ratio R(A) = V / ([1 + g′]² σ_MF²), the paired matched filter and the η reference lines |
| `biasfixed_diagnostics.py` | the band and bias machinery that `teaser_eval.py` imports |
| `make_fig12_teaser.py` | Fig. 3 of the paper from the `*_eval.json` files (file name from an earlier figure numbering) |
| `test_teaser_stats.py` | self-test of the statistics path: a shrunk matched filter must give R ≡ 1 |
| `checkpoints/` | the four best-validation checkpoints behind Fig. 3 (bc15 and mse for each cell; ResNet weights plus metadata, 5.8 MB each) |
| `results/` | their `teaser_eval.py` summaries (41 amplitudes × 4000 EVAL maps), the inputs of Fig. 3 |
| `hpc/` | the Slurm jobs of the production runs (templates; set account, partition, GPUs) and the run list `teaser_runs.txt` |

The paper quotes the **best-validation** checkpoints (`*.pth.best.pth`). The final-epoch states are over-fitted and are not released.

**Why the checkpoints are included.** They are the networks the paper's numbers describe. A re-trained network reproduces those numbers only statistically, so the checkpoints are the only way to check the quoted R values, and the evaluation code, against exactly the networks in Fig. 3. Evaluation needs no GPU. In the release check, one checkpoint ran at about 35 ms per map-and-amplitude pair on two CPU threads, so the full 41 × 4000 grid takes of order an hour on a laptop CPU (use `--n-real` or `--n-grid` for a quicker look), and Apple-silicon `mps` is faster. Training one network costs roughly 300 times as much (700 epochs over 24 000 maps, forward and backward passes) and in practice needs a GPU.

## Reproducing

From this directory, with the repository installed (`pip install -e "..[torch]"`):

```bash
export TEASER_DATA=$PWD/teaser_data          # ~2.4 GB per model at full size

# 1. datasets (CPU, a few minutes per model)
python make_teaser_dataset.py --model T0_RED_REAL  --L 5 --n-train 30000 --n-eval 6000 --n-psd 2000 --outdir $TEASER_DATA
python make_teaser_dataset.py --model T1_PSRAND_WN --L 7 --n-train 30000 --n-eval 6000 --n-psd 2000 --outdir $TEASER_DATA

# 2a. evaluate a released checkpoint (no training needed)
python teaser_eval.py --checkpoint checkpoints/TEASER_T0_RED_REAL_L5_bc15.pth.best.pth \
    --eval-h5 $TEASER_DATA/TEASER_T0_RED_REAL_evalnoise6000_s990034.h5 --outdir eval_out_best

# 2b. or train (the paper: 700 epochs on 4 GPUs, about 1 h per run; Apple-silicon mps works, slowly)
python train_teaser_ddp.py --h5 $TEASER_DATA/TEASER_T0_RED_REAL_train30k_L5_s880034.h5 \
    --checkpoint checkpoints/my_run.pth --loss-function bias_corrected_mse \
    --epochs 700 --batch-size 50 --lr 0.01 --seed 40

# 3. the figure
python make_fig12_teaser.py eval_out_best/TEASER_T0_RED_REAL_L5_bc15.pth.best_eval.json ...
```

The datasets are regenerated bit for bit from the registry seeds, so a released checkpoint evaluated on a regenerated EVAL file reproduces the shipped `results/*_eval.json` up to the Monte Carlo error of the number of EVAL maps used. In the release check, the `T0_RED_REAL` bc15 checkpoint evaluated on only 200 regenerated maps gave R = 1.08 ± 0.09 over its certified band, compared with 1.107 ± 0.021 from 4000 maps in the paper. Training is stochastic (GPU nondeterminism, DDP), so re-trained networks reproduce the paper's numbers statistically, not bit for bit.

On a Slurm cluster, `hpc/teaser_gen.slurm`, `hpc/teaser_train.slurm` (array over `hpc/teaser_runs.txt`) and `hpc/teaser_eval.slurm` run the same three steps. Set the account, partition and GPU request for your site, and set `BTB_VENV_ACTIVATE` to your environment's activate script.

The ResNet architecture and training loop come from the author's earlier CNN-vs-matched-filter scripts. The changes for the teaser are the float32 data path (no 8-bit quantization), exact D4 augmentation instead of `RandomRotation`, bias-correction bins spanning [0, L], and checkpoint metadata. Each change is documented at the top of `train_teaser_ddp.py` and `utils_teaser_ddp.py`.
