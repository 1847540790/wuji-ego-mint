# Data pipeline

The data pipeline converts short egocentric videos into camera and hand
trajectories and exports LeRobot v3 datasets. It uses Ray for GPU-aware
scheduling and resumable manifests.

## Preflight

```bash
conda activate mint
python -m mint doctor --profile data --strict
```

Start with one approved clip and one GPU. Verify disk capacity for intermediate
frames, model predictions, and final Parquet/video output.

## Run

```bash
python -m mint pipeline \
  --input data/samples/epic-kitchens-01.mp4 \
  --output output/processed \
  --num-gpus 1
```

The underlying entry point also accepts a directory or a text file containing
one video path per line. Use only paths the current operator is authorized to
process.

## Resume

The pipeline writes a manifest under its timestamped output directory. Resume
the same output rather than copying partial files:

```bash
python -m mint pipeline --input data/samples --resume output/processed/<run-directory>
```

Use `--retry-failed` only after addressing the recorded error. Retrying a
missing model asset without fixing it creates noisy duplicate failures.

## Output contract

The public contract consists of:

- per-video prediction and cleaned trajectory files;
- sidecar metadata that uses paths relative to the run root;
- LeRobot v3 Parquet and video chunks;
- a resumable manifest with status and failure summaries.

No output should contain source-system mount names, operator usernames, cloud
credentials, or unrestricted source paths. Run the privacy audit before using
an output as a release artifact.
