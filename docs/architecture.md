# Architecture

MINT keeps the research pipeline modular while presenting one stable CLI.

```text
Approved videos
      |
      v
Ray data pipeline -----> LeRobot v3 dataset
      |                         |
      |                         v
Optional research backends   Training engine -----> Checkpoint
                                                   |
                                                   v
                                      Prediction engine
                                         |       |
                                         v       v
                                      NPZ data  Web viewer
```

## Boundaries

### Data plane

`ray_pipeline/` owns scheduling, actors, resumable manifests, monitoring, and
LeRobot export. `modules/` provides adapters for optional upstream research
models. `data_cleaning/` contains trajectory filters used before export.

Downloaded backends live under `third_party/` and are ignored by Git. This
keeps restricted licenses, model weights, and large upstream histories out of
the MINT source release.

### Training plane

`model_train/` is a configuration-driven PyTorch and Accelerate stack. A small
registry constructs the dataset, model, and loss objects. The public build
includes the primary LingBot-Map student and excludes cloud-vendor submission
logic, private dataset mixtures, credentials, and experimental model variants.

### Inference plane

`mint/inference/` reuses the exact training preprocessing and model constructor.
It supports bounded windowed inference and saves only prediction arrays. The
renderer decodes predicted camera and MANO values without loading annotations.

### Viewer boundary

The viewer receives one configured sample root at startup. It hashes relative
paths into opaque sample IDs and exposes no arbitrary filesystem route. Jobs
are generated server-side and artifact routes allow only a fixed filename set.
The default bind address is `127.0.0.1` and all responses include restrictive
browser security headers.

## Design principles

1. Portable paths are resolved from the repository, never a developer home.
2. Credentials arrive through the runtime environment, never tracked files.
3. Dataset and checkpoint licenses remain explicit installation boundaries.
4. Long-running work is resumable or bounded by an explicit frame limit.
5. Prediction visualization is independent of ground-truth data.

