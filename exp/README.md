# Experiment settings

Follow the [quickstart](../README.md#1-dataset) for downloading,
BigCity extraction, and launch commands. The Docker command there sets dataset
and output paths; outside Docker, set `LOBEGS_DATA_ROOT` and `LOBEGS_OUT_DIR`, or
edit their defaults in [datasets.py](datasets.py).

## Scene IDs

| ID | Scene | Original poses | Refined poses |
| --- | --- | --- | --- |
| 1 | Building | `mill19/building-pixsfm` | `mill19/building` |
| 2 | Rubble | `mill19/rubble-pixsfm` | `mill19/rubble` |
| 3 | Residence | `urban_scene_3d/residence-pixsfm` | `urban_scene_3d/residence` |
| 4 | Sci-Art | `urban_scene_3d/sci-art-pixsfm` | `urban_scene_3d/sci-art` |
| 5 | SmallCity | `mc_smallcity` | — |

Paths are relative to `original-poses/` or `refined-poses/`. ID 0 runs all scenes:
five base or four refined. Both launchers accept `--scene_id` as an alias.
BigCity has a separate launcher with one scene (ID 1).

## Options and outputs

| Launcher | Iteration override | Output under `LOBEGS_OUT_DIR` |
| --- | --- | --- |
| `base_exp.py` | `--iterations N` | `exp_base/<scene>/` |
| `refined_cam_bilgrid_exp.py` | `--iterations N` | `refined_cam_pose_bil_grid/<scene>/` |
| `bigcity_exp.py` | `--iteration N` or `--iterations N` | `exp_base/mc_bigcity/pose/` |

BigCity trains parents and children for 30,000 iterations each by default, with
camera targets of 2,000 and 1,000. Its iteration override applies to both stages.
`--dry-run` prints commands and writes stage YAMLs; use a scratch output root.
Resume with `--skip` followed by any of `parent-partition`, `parent-train`,
`child-partition`, `child-train`, `merge`, or `eval`. Downstream stages require
the skipped stages' saved artifacts.

Scene geometry and optimizer presets live in `datasets.py`. For custom inputs,
see [dataset preparation](../README.md#2-custom-dataset).
