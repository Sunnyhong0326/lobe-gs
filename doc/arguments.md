# Configuration and arguments

Run commands from the repository root inside the [Docker container](../README.md#installation).
Edit the YAML files or pass `key=value` overrides to `main.py`:

```bash
python main.py \
  pipeline.mode=train_full pipeline.source_path=/data/my_scene \
  pipeline.out_dir=/app/output pipeline.exp_name=custom \
  pipeline.pretrained_coarse_model_path=/app/output/custom/my_scene/coarse \
  pipeline.resolution=4 pipeline.num_blocks=8 \
  'pipeline.xyz_plane=[1,1,0]' \
  partition.partition_params.contracted_space=false
```

Use `true`/`false` for booleans, `null` for unset values, and quotes around lists.
Preview composed YAML with `python main.py --cfg job` without running training.
Shared `pipeline.*` values overwrite corresponding stage settings at runtime;
use them for paths, resolution, alignment, and appearance.

## Configuration files

| File | Override prefix | Contents |
| --- | --- | --- |
| [base.yaml](../config/base.yaml) | `pipeline.` | Paths, modes, shared settings |
| [coarse/base.yaml](../config/coarse/base.yaml) | `coarse.` | Coarse training |
| [partition/base.yaml](../config/partition/base.yaml) | `partition.` | Partitioning and camera selection |
| [fine/base.yaml](../config/fine/base.yaml) | `fine.` | Block training |
| [align/base.yaml](../config/align/base.yaml) | `align.` | Coordinate alignment |

Tables below list common arguments and YAML defaults. See these files for all settings.

## Pipeline

All arguments in this table use the `pipeline.` prefix.

| Argument | Default | Description |
| --- | --- | --- |
| `source_path` | `""` | Dataset root; set for your scene. |
| `train_dir` / `test_dir` | `train` / `test` | Split directories. Empty training directory uses the root; empty test directory disables separate test evaluation. |
| `images` | `images` | Image directory within each split. |
| `out_dir` / `exp_name` | `output` / `""` | Output root and optional experiment name. |
| `pretrained_coarse_model_path` | `""` | Coarse model directory for downstream stages; for a new run, point to the directory that coarse training will create. |
| `resolution` | `4` | `1`, `2`, `4`, `8`: image downsampling factor. `-1`: cap width at 1600 pixels. Other positive values: target width. |
| `xyz_plane` | `[1,1,0]` | Partition plane: XY (Z up); use `[1,0,1]` for XZ (Y up). |
| `num_blocks` | `8` | Target KD-tree block count, subject to splitting limits. |
| `camera_count` | `null` | Camera-count stopping target when `num_blocks` is nonpositive. |
| `aabb` | `[]` | Scene bounds: `[minx,miny,minz,maxx,maxy,maxz]`; empty infers bounds. Saved block JSON uses `[minx,maxx,miny,maxy,minz,maxz]`. |
| `aabb_pad` | `0` | Block/camera-selection padding; depends on scene scale and selection method. |
| `iterations` | `null` | Override both coarse and fine iteration counts; otherwise use stage defaults. |
| `use_pretrain` | `true` | Initialize downstream stages from the coarse model. |
| `appearance_enabled` | `false` | Enable the appearance module. |
| `align_enabled` | `false` | Read aligned `sparse_pca/0` reconstructions in downstream stages. |
| `eval_coarse` | `false` | Also evaluate the coarse model on the test split. |

For the example's standard train/test layout, results are under
`/app/output/custom/my_scene/{coarse,partition,fine}/`. Increase the downsampling
factor or block count if block training runs out of GPU memory.

### Modes

Set `pipeline.mode` (default: `full`). Fine training modes also merge blocks and
render/evaluate the test split when configured.

| Mode | Stages |
| --- | --- |
| `train_full` | Coarse training, partitioning, fine training |
| `full` | Alignment, then `train_full` |
| `align` / `coarse` / `partition` | Only the named stage |
| `fine` | Fine training using an existing partition |
| `partition_fine` | Partitioning and fine training using existing model inputs |
| `partition_benchmark` | Partitioning at `pipeline.partition_image_fractions` (default `[0.25,0.5,0.75]`) |

For prepared train/test data, use `train_full` with `align_enabled=false`.
The mode selects whether alignment runs; `align_enabled` selects whether its
outputs are read. If aligning, apply the same transform to both splits.

## Partitioning

| Argument | Default | Description |
| --- | --- | --- |
| `partition.model_params.proxy_load_metric` | `cam_count` | Load metric: `cam_count`, `visible_pts`, `contri_pts`, `block_pts`, `area`, or normalized mixtures such as `cam_count+visible_pts`. |
| `partition.partition_params.contracted_space` | `true` | Use contracted space; set `false` for world-space partitioning. |
| `partition.partition_params.camera_selection_method` | `rendered_depth` | `rendered_depth`, `mono_depth`, `point`, `ssim`, or `vast_bbox`. Rendered depth needs a trained Gaussian model; mono depth needs prepared depth inputs. |
| `partition.partition_params.camera_selection_threshold` | `0.15` | Selection threshold; interpretation depends on the method. |
| `partition.partition_params.depth_mode` | `ED` | Rendered-depth mode: `ED` or `D`. |
| `partition.partition_params.cutline_refinement_enabled` | `true` | Refine split positions to balance load. |

## Training

Use `coarse.optim_params.<argument>` or `fine.optim_params.<argument>`.

| Argument | Coarse default | Fine default | Description |
| --- | --- | --- | --- |
| `iterations` | `30000` | `30000` | Training length. |
| `position_lr_init` | `0.00016` | `0.000064` | Initial position learning rate. |
| `position_lr_final` | `0.0000016` | `0.00000064` | Final position learning rate. |
| `position_lr_max_steps` | `30000` | `30000` | Position schedule length; adjust separately from iterations. |
| `scaling_lr` | `0.005` | `0.005` | Scale learning rate. |
| `densification_interval` | `100` | `100` | Iterations between densification updates. |
| `densify_from_iter` / `densify_until_iter` | `500` / `15000` | `500` / `15000` | Densification window. |
| `densify_grad_threshold` | `0.0002` | `0.0002` | Densification gradient threshold. |
| `lambda_dssim` | `0.2` | `0.2` | SSIM loss weight. |


