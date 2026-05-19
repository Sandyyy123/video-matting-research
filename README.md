# Video Matting Experimentation Framework

A modular PyTorch research codebase for training and ablating video alpha matting models. Covers custom loss design, curriculum learning, hard-example mining, temporal consistency via recurrent heads, and structured ablation tooling.

This framework was developed to explore improvements over baselines like [RVM (Robust Video Matting)](https://github.com/PeterL1n/RobustVideoMatting) and FTP-VM, focusing on the training regime rather than backbone architecture.

---

## Key Modules

| Module | Description |
|---|---|
| `losses/matting_losses.py` | Four custom loss functions with Kendall uncertainty weighting |
| `training/curriculum.py` | Stage-based curriculum scheduler (synthetic to real) |
| `training/hard_example_sampler.py` | Per-sample loss tracking + FIFO replay buffer |
| `models/temporal_module.py` | ConvGRU-based recurrent head for frame-to-frame coherence |
| `experiments/ablation_runner.py` | Automated ablation orchestration with W&B logging |
| `configs/base_config.yaml` | Single-file training configuration |

---

## Loss Design (`losses/`)

### `TemporalConsistencyLoss`

Penalizes alpha matte flickering between adjacent frames. Given predicted alphas at times _t_ and _t+1_ and an optical flow field, the loss warps alpha_t into the coordinate frame of _t+1_ and computes an L1 difference:

```
L_temp = (1/N_valid) * sum_i [ |alpha_t_warped(i) - alpha_{t+1}(i)| * valid_mask(i) ]
```

Occlusion is handled by masking pixels where flow magnitude exceeds a threshold (default 20px). If no flow is provided, a naive frame-difference is computed instead.

### `BoundaryRefinementLoss`

Alpha boundaries are where most reconstruction error concentrates. This loss derives an edge mask from the ground-truth alpha using Sobel gradients, dilates it by a configurable radius, and applies a multiplicative up-weight (default 5x) to the per-pixel L1 loss in that region:

```
L_boundary = mean( |pred - gt| * (1 + (w_edge - 1) * edge_mask) )
```

### `TriMapGuidedLoss`

Combines L1 loss in the unknown (transition) region with binary cross-entropy in the known foreground/background regions, mirroring the decomposition used in Deep Image Matting (Xu et al., 2017):

```
L_trimap = w_unk * L1_unknown / N_unk   +   w_bce * BCE_known / N_known
```

### `MultiTaskMattingLoss`

Combines all three losses using Kendall et al. (2018) homoscedastic uncertainty weighting. Each task has a learnable log-variance parameter `s_i` (initialized to 0):

```
L_total = sum_i [ exp(-s_i) * L_i + s_i ]
```

The `s_i` parameters are learned end-to-end alongside network weights, automatically balancing loss magnitudes without manual tuning.

---

## Curriculum Learning (`training/curriculum.py`)

Training progresses through four difficulty stages:

| Stage | Epochs | Noise | Motion Blur | Synthetic : Real |
|---|---|---|---|---|
| `synthetic_clean` | 10 | 0.00 | No | 100% : 0% |
| `synthetic_noisy` | 15 | 0.05 | Yes | 100% : 0% |
| `real_easy` | 20 | 0.02 | No | 70% : 30% |
| `real_hard` | remaining | 0.05 | Yes | 60% : 40% |

`CurriculumScheduler` manages stage transitions, exposes per-epoch augmentation parameters to the data loader, and supports checkpoint save/restore of its state.

---

## Hard-Example Mining (`training/hard_example_sampler.py`)

`HardExampleSampler` maintains a rolling loss history per training sample over a configurable window (default 10 epochs). At the end of each epoch, samples whose mean loss exceeds `global_mean + threshold_std * global_std` are added to a FIFO replay buffer (capacity 5,000 by default). A configurable fraction of each batch (default 15%) is then replaced with hard examples drawn from the buffer.

This addresses the long-tail distribution of difficult foreground textures (hair, semi-transparent fabrics) without requiring manual dataset curation.

---

## Recurrent Temporal Head (`models/temporal_module.py`)

`RecurrentMattingHead` inserts temporal memory into any frame-level matting backbone without requiring optical flow at inference time.

```
Encoder features (B, C, H/s, W/s)
        |
  FeatureAdapter  ->  project to hidden_dim
        |
  ConvGRUCell x N  <-  hidden state from previous frame
        |
  MattingDecoder  ->  upsample to full resolution
        |
  alpha (B, 1, H, W)  +  new_hidden
```

The `ConvGRUCell` replaces standard GRU matrix multiplies with 2-D convolutions, preserving spatial structure in the hidden state. The module exposes a `process_sequence` method for efficient training over temporal clips, and full `state_dict` / `load_state_dict` support for inference resumption mid-video.

---

## Ablation Runner (`experiments/ablation_runner.py`)

`AblationRunner` accepts a base config dict and a list of `(name, override_dict)` conditions. For each condition it:

1. Deep-merges the override into a copy of the base config
2. Initializes a W&B run (if configured)
3. Calls the user-supplied `run_fn(config) -> metrics_dict`
4. Logs metrics to W&B and records wall-clock runtime
5. After all conditions complete, writes a Markdown summary table to `ablation_results/ablation_summary.md`

Example usage:

```python
conditions = [
    ("baseline",       {}),
    ("no_temporal",    {"losses": {"temporal_consistency": {"weight": 0.0}}}),
    ("no_boundary",    {"losses": {"boundary_refinement":  {"weight": 0.0}}}),
    ("high_temporal",  {"losses": {"temporal_consistency": {"weight": 1.0}}}),
]
runner = AblationRunner(base_config, conditions, run_fn=train_and_eval)
runner.run_all()
```

---

## Configuration (`configs/base_config.yaml`)

All hyperparameters are centralized in a single YAML file:

- Model architecture and temporal head dimensions
- Optimizer (AdamW), scheduler (cosine with warmup), gradient clipping
- Per-loss weights and edge dilation radius
- Curriculum stage definitions with per-stage augmentation params
- Hard-example buffer size and upsampling threshold
- Dataset paths and synthetic/real mixing ratios (default 60/40)
- W&B project and logging intervals

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python >= 3.9 and CUDA 11.7+. Optical flow warping in `TemporalConsistencyLoss` uses only PyTorch built-ins (`F.grid_sample`) and does not require a separate flow estimation library at training time.

---

## Dataset Layout

```
data/
  adobe_comp1k/          # Adobe Composition-1K (Xu et al., 2017)
  distinctions646/       # Distinctions-646 (Qiao et al., 2020)
  am2k/                  # AM-2K (Li et al., 2022)
  videomatte240k/        # VideoMatte240K (Lin et al., 2021)
  backgrounds/           # Background video clips for compositing
```

---

## Evaluation Metrics

The standard video matting metrics used throughout:

- **SAD** (Sum of Absolute Differences): primary ranking metric
- **MSE** (Mean Squared Error): alpha reconstruction quality
- **Grad** (Gradient error): sharpness of alpha boundary
- **Conn** (Connectivity error): structural coherence of alpha regions

---

## References

- Xu et al. (2017) "Deep Image Matting." CVPR. https://arxiv.org/abs/1703.03872
- Lin et al. (2021) "Robust High-Resolution Video Matting with Temporal Guidance." WACV. https://arxiv.org/abs/2108.11515
- Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to Weigh Losses in Deep Learning." CVPR. https://arxiv.org/abs/1705.07115
- Qiao et al. (2020) "Attention-Guided Hierarchical Structure Aggregation for Image Matting." CVPR.
- Li et al. (2022) "Bridging Composite and Real: Towards End-to-end Deep Image Matting." IJCV.

---

## License

MIT
