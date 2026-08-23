# Gaze2HOI: From Gaze to Grasp for Hand–Object Interaction Generation

Gaze2HOI generates grasp-based hand–object interaction motion from a **measured gaze
sequence** instead of a text prompt. Gaze specifies *where* on an object the interaction
should happen — the spatial intention that language often leaves implicit.

The model has three parts:

- **GeoGaze** — Geometry-Aware Gaze Encoding. Each gaze ray is mapped onto a fixed Basis
  Point Set (BPS) over the object as two frame-wise fields: *ray closeness* (local
  proximity to the ray) and *BPS-displacement alignment* (global gaze/geometry direction).
- **GazeFlow** — a cascaded gaze–interaction attention module. Gaze conditions the hand
  streams first, and the gaze-conditioned hand tokens then coordinate object motion
  (`Gaze -> Hand -> Object`).
- **HOT3D-Grasp** — a grasp-centered benchmark derived from egocentric HOT3D recordings
  with object-part annotations.

## Repository layout

```
gaze2hoi/          training and inference entry points
  train.py           diffusion training
  test.py            multi-seed inference, writes prediction pickles
lib/
  networks/gaze2hoi.py          denoiser: GazeFlow cascade + diffusion Transformer
  networks/diffusion.py         diffusion process
  utils/gaze2hoi_train_helpers.py   GeoGaze encoding, BPS, data conditioning
  utils/gaze2hoi_config.py      config resolution
  datasets/, models/, utils/    dataset, MANO, geometry helpers
configs/           Hydra configuration (configs/gaze2hoi/gaze2hoi.yaml is the main file)
constants/         per-dataset constants
preprocess/        HOT3D preprocessing
eval/              metric evaluator (ID, IV, Pen_1cm, Part Acc., PCP, G2C, GSR, ...)
scripts/           ablation training sweep and checkpoint scoring helpers
assets/            object-part labels, BPS basis, sampled MANO vertex indices
```

## Installation

```bash
conda create -n gaze2hoi python=3.10 -y
conda activate gaze2hoi
pip install -r requirements.txt
# pytorch3d: follow https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
```

## Data

Datasets and body models are **not redistributed here** and must be obtained from their
original providers under their own licenses:

| Resource | Where |
| --- | --- |
| HOT3D | https://github.com/facebookresearch/hot3d |
| MANO | https://mano.is.tue.mpg.de/ |

Expected layout:

```
data/hot3d/
  object_mesh/            per-object meshes
  dataset/
    obj.pkl               sampled object point clouds
    ori_dataset/gaze_train, ori_dataset/gaze_test
  text.json  text_length.json  text_count.json  balance_weights.pkl
```

Preprocessing:

```bash
python preprocess/preprocessing_hot3d.py
```

## Training

```bash
python gaze2hoi/train.py \
  gaze2hoi.exp.name=gaze2hoi_both \
  gaze2hoi.exp.iteration=100000 \
  gaze2hoi.exp.seed=0
```

Checkpoints land in `outputs/gaze2hoi/<name>/`, with a milestone every 10k iterations.

### Ablations

| Variant | Override |
| --- | --- |
| Both cues (default) | — |
| No gaze | `gaze2hoi.model.null_gaze_condition=true` |
| Ray closeness only | `gaze2hoi.model.gaze_condition_mode=gage_closeness_temporal` |
| Alignment only | `gaze2hoi.model.gaze_condition_mode=gage_alignment_temporal` |
| Direct (no cascade) | `gaze2hoi.model.gaze_token_fusion=token` |
| Parallel | `gaze2hoi.model.cross_attn_order=parallel` |
| Object -> Hand | `gaze2hoi.model.cross_attn_order=object_hand` |

The full sweep (7 variants x 3 seeds, sharded over GPUs):

```bash
for i in 0 1 2 3; do
  RUN_TAG=ablations GPU_ID=$i SHARD_ID=$i NUM_SHARDS=4 \
    bash scripts/train_ablations.sh &
done
bash scripts/training_status.sh    # progress and ETA
```

## Inference

```bash
python gaze2hoi/test.py \
  gaze2hoi.exp.weight_path=outputs/gaze2hoi/<name>/iteration_0100000.pth \
  gaze2hoi.exp.save_name=predictions/<name> \
  gaze2hoi.exp.num_test_seeds=10 \
  dataset.data_name=ori_dataset/gaze_test
```

Ablation flags recorded in the checkpoint (`gaze_condition_mode`, `cross_attn_order`,
`null_gaze_condition`, ...) are restored automatically, so inference matches training
without repeating the overrides.

## Evaluation

```bash
bash eval/run_eval_gpu.sh --device cuda:0 \
  --input predictions/<name>.pkl \
  --new-metric-csv-output results/<name>.csv \
  --new-metric-md-output  results/<name>.md
```

Reported metrics:

| Metric | Definition |
| --- | --- |
| ID | interpenetration depth (mm); per frame the deepest penetrating hand vertex, averaged over contact frames |
| IV | interpenetration volume (cm^3) on a 5 mm voxel grid |
| Pen_1cm | share of contact frames whose deepest penetration is <= 1 cm |
| Part Acc. | at the final valid frame, the dominant contacted part equals the target part |
| PCP | at the final valid frame, share of contacted points inside the target part |
| G2C | distance from the final contact to the gaze map (cm), contact-bearing samples only |
| ConPass | contact (CR > 0) held through the final five valid frames |
| GSR | ConPass **and** final-frame ID <= 1 cm **and** final object lift >= 5 mm |

Contact uses a 5 mm hand-to-surface threshold. `--gsr-lift-cm` changes the lift
requirement; `--gsr-contact-frames` changes the contact window.

Rerun visualization of the metric geometry (green lines = ID, red voxels = IV,
yellow points = contact):

```bash
bash eval/run_eval_gpu.sh --device cuda:0 --input predictions/<name>.pkl \
  --new-metric-visualize --sample-idx 0 1 2 --show-id-values \
  --rerun-save results/<name>.rrd --rerun-wait 0
```

## Attribution

This codebase started from **Text2HOI** (CVPR 2024),
https://github.com/JunukCha/Text2HOI, and keeps parts of its data pipeline, MANO
handling, and diffusion scaffolding. The BPS object encoding follows **DiffH2O**, the
hand-to-object displacement target follows **BimArt**, and Part Accuracy / Contact
Precision follow **Text2Grasp** / **NL2Contact**.

`assets/grab_bps_1024.pt` is the 1,024-point BPS basis used by DiffH2O/GRAB, and
`assets/part_fps_hand_index_100.npy` is the 100 sampled MANO vertex indices used by
BimArt. They are included so that results are reproducible; the underlying datasets and
the MANO model itself are not redistributed.

**Before publishing this repository, check the license of Text2HOI and of every asset
above, and add a `LICENSE` file that is compatible with them.** No license file is
included here because that choice is yours to make.
