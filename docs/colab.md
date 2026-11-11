# Colab Guide

`src/colab.sh` runs the segmentation and VLM rating steps on a Colab GPU, with inputs and results going through
Google Drive.

## Setup

Install Google Colab CLI:
```
uv sync --group colab
```

Sign in to Google:
```
uv run --group colab colab sessions
```

Sign in to Hugging Face, with an account that has access to SAM 3 (`up` copies the login to the VM):
```
.venv/bin/hf auth login
```

## Steps

Run the `data_prep` stage locally:
```
python src/main.py data_prep
```

`colab.sh` records `--step`, `--exp` and `--drive` in `.colab-run`: give them once, and the commands after use them
until you give others.

Run the pack step once, then upload the input archive to the Google Drive folder:
```
# Produce the input archive ./inputs_<city>.tar.gz and the code archive ./sim_code.tar.gz in the current directory
src/colab.sh pack --exp <experiment_config_name> --drive <drive_path>
```

Start a GPU session on the step and unpack the code, the inputs and any earlier results (from the directory you ran
pack in: it uploads the code archive from there). The first time, mounting Drive asks you to authorize it:
```
src/colab.sh up --step <seg|vlm> --gpu <gpu_type>
```

Start the full run, saving to Google Drive every 10 minutes:
```
src/colab.sh start
```

Watch it until it ends:
```
src/colab.sh watch
```

Save the results to Google Drive and release the GPU:
```
src/colab.sh down
```

Download the results archive from the Google Drive folder, then unpack it into the repo:
```
src/colab.sh fetch <path_to_results_archive>
```

> Delete the input archives from Drive: Google's terms limit how long that imagery can be stored.

## Commands

```
src/colab.sh <command> [--step <step>] [--exp <name>] [--gpu <type>] [--drive <folder>]
```

| Command | Description |
|---|---|
| `pack` | Makes the input archive and the code archive in the current directory; checks the archive's round trip; warns about uncommitted changes in `src/` or `configs/` |
| `up` | Starts a GPU session (L4 for the segmentation, G4 for the VLM, unless `--gpu` says otherwise), mounts Drive, unpacks the code, the inputs and any earlier results; installs `transformers`, `accelerate`, `peft`, `pyyaml`, `tqdm` and removes Colab's `torchao` (it stops `peft` loading JEV's adapter); reuses a running session |
| `start` | The full run in the background on the VM; results are saved to Drive every 10 minutes |
| `status` | The end of the log and the number of items done |
| `watch` | `status` every 5 minutes until the run ends |
| `save` | Saves the results to Drive now |
| `down` | Saves the results to Drive, then releases the GPU |
| `stop` | Stops the run and saves its results to Drive; `start` resumes it on the same session |
| `reset` | Stops the run and deletes its results on the VM and in Drive: only to start over |
| `fetch <archive>` | Unpacks a results archive downloaded from Drive (`.tar.gz` or `.tar`) into the repo, replacing the step's folder; refuses an archive whose name is not the step's and experiment's (`<step>_<run_id>`) |

Run them from any directory: `pack` writes both archives to it, `up` uploads the code archive from it, and `fetch`
reads a relative path from it.

## Arguments

| Argument | Description | Default |
|---|---|---|
| `--step` | `seg` runs `run_segmentation_models`, `vlm` runs `rate_views_with_vlm`. | **Required** except `pack` |
| `--exp` | The experiment: `configs/experiments/<name>.yaml`. | `murrayhill_qwen3` |
| `--gpu` | A GPU type the Colab CLI offers (`colab new --help` lists them; availability depends on your Colab subscription), and only `up` reads it. | `L4` for the `seg`, `G4` for the `vlm` |
| `--drive` | The Google Drive folder for the archives. | `MyDrive/sim` |
| `-h`, `--help` | Prints the usage. | - |

A missing or unknown step, an unknown command or option, an extra argument or an unknown experiment stops `colab.sh`
with the usage (or the steps or experiments there are), and nothing is recorded.

## Files

| File | Where | Made by |
|---|---|---|
| Input archive | `inputs_<city>.tar.gz` in the directory `pack` ran from, then the Drive folder | `pack` |
| Code archive | `sim_code.tar.gz` in the directory `pack` ran from (`up` uploads it from there), then `/sim_code.tar.gz` on the VM | `pack`, `up` |
| Results archive | `<Drive folder>/<step>_<run_id>.tar.gz` | `start` (every 10 minutes), `save`, `down`, `stop` |
| Run log | `/content/<step>.log` on the VM | `start` |

The input archive holds what both steps need, all from `data_prep`:
- The photos (`street-view-frames`) and the stitched views (`views-180`)
- `nodes.csv`, `views-180.csv`, `street-view-frames.csv`, `street-view-metadata.csv`
- `node-review-decisions.csv`, when you have made it, so excluded nodes are not rated

## Running again

One run per session: to rerun, `stop` (keeps the results) or `reset` (deletes them), then `start`.

Between the two `--step`: `down`, so the next `up` starts a session on that step's GPU. `up` reuses a running session
with the GPU it started on.

If a session dies: `up` unpacks the last results saved to Drive, and `start` skips what is done.

After a new review decision: `pack` again and upload the new input archive, so excluded nodes are not rated.
