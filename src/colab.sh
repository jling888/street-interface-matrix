#!/usr/bin/env bash
# Run a GPU step on Colab with the Colab CLI, inputs and results going through Google Drive.
#
# Usage: src/colab.sh <command> --step <step> [--exp <run_id>] [--gpu <type>] [--drive <folder>]
#
#   src/colab.sh pack     make the city's input archive (upload it to Drive yourself) and the code archive, in
#                         the current directory
#   src/colab.sh up       start a GPU session, mount Drive, unpack code + inputs (+ earlier results), install;
#                         run it where pack made the code archive
#   src/colab.sh start    full run in the background on the VM; results saved to Drive every 10 min
#   src/colab.sh status   tail of the log and the number of items done
#   src/colab.sh watch    status every 5 min until the run ends (one check at a time: a hung check is killed)
#   src/colab.sh save     save the results to Drive now
#   src/colab.sh down     save the results to Drive, then release the GPU
#   src/colab.sh stop     stop the run on the VM and save its results to Drive; `start` resumes it on the same session
#   src/colab.sh reset    stop the run on the VM and delete its results there and in Drive (only to start over)
#   src/colab.sh fetch F  unpack a results archive F (downloaded from Drive; .tar.gz or .tar) into the repo
#
#   --step <step>   the GPU step: seg (run_segmentation_models) or vlm (rate_views_with_vlm); every command
#                   but pack needs one
#   --exp <run_id>  the experiment, a file in configs/experiments/ (default murrayhill_qwen3_v2)
#   --gpu <type>    T4, L4, G4, H100 or A100, when `up` starts a session (default L4 for the segmentation, G4 for
#                   the VLM)
#   --drive <folder> the Google Drive folder for the archives (default MyDrive/sim)
#   -h, --help      this text
#
# --step, --exp and --drive are remembered (.colab-run in the repo): a command without one uses the last given, so
# give them once, e.g. on up. Every command prints the ones it uses.
#
# e.g. src/colab.sh up --step vlm --gpu H100 --exp murrayhill_jev
# Run from any directory (pack writes both archives there, up reads the code archive there). Details and troubleshooting: docs/colab.md
set -euo pipefail

usage() { sed -n '2,/^# Run from any directory/p' "$0"; }
CMD="" ARCHIVE="" STEP="" RUN_ID="" GPU="" DRIVE_FOLDER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --step) [ $# -ge 2 ] || { echo "--step needs a step"; exit 1; }; STEP="$2"; shift 2 ;;
    --exp) [ $# -ge 2 ] || { echo "--exp needs a run_id"; exit 1; }; RUN_ID="$2"; shift 2 ;;
    --gpu) [ $# -ge 2 ] || { echo "--gpu needs a GPU type"; exit 1; }; GPU="$2"; shift 2 ;;
    --drive) [ $# -ge 2 ] || { echo "--drive needs a folder"; exit 1; }; DRIVE_FOLDER="${2%/}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option $1"; echo; usage; exit 1 ;;
    *)
      if [ -z "$CMD" ]; then CMD="$1"
      elif [ "$CMD" = fetch ] && [ -z "$ARCHIVE" ]; then ARCHIVE="$1"
      else echo "unexpected argument $1"; echo; usage; exit 1
      fi
      shift ;;
  esac
done

case "$CMD" in
  pack|up|start|status|watch|save|down|stop|reset|fetch) ;;
  "") usage; exit 1 ;;
  *) echo "unknown command $CMD"; echo; usage; exit 1 ;;
esac
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"   # from the script's path: runs from any directory
CODE_ARCHIVE="$PWD/sim_code.tar.gz"                # made by pack, uploaded by up: in the directory they run in
# the flags not given come from the last command's, then the defaults
REMEMBERED="$REPO/.colab-run"
if [ -f "$REMEMBERED" ]; then
  while IFS="=" read -r key value; do
    case "$key" in
      step) STEP="${STEP:-$value}" ;;
      exp) RUN_ID="${RUN_ID:-$value}" ;;
      drive) DRIVE_FOLDER="${DRIVE_FOLDER:-$value}" ;;
    esac
  done < "$REMEMBERED"
fi
RUN_ID="${RUN_ID:-murrayhill_qwen3_v2}" DRIVE_FOLDER="${DRIVE_FOLDER:-MyDrive/sim}"

SESSION=sim
case "$STEP" in
  seg|vlm) ;;
  "") [ "$CMD" = pack ] || { echo "$CMD needs --step: seg or vlm"; exit 1; } ;;
  *) echo "--step $STEP: seg or vlm"; exit 1 ;;
esac
[[ "$DRIVE_FOLDER" =~ ^[A-Za-z0-9._/-]+$ ]] || {   # it goes unquoted into the commands on the VM
  echo "--drive $DRIVE_FOLDER: use letters, digits and . _ - / only (e.g. MyDrive/sim)"; exit 1
}
DRIVE="/content/drive/$DRIVE_FOLDER"               # the Drive folder on the VM; holds the archives
# the flags every later command of this run needs, for the hints and watch's own status calls
RUN_FLAGS="--step $STEP --exp $RUN_ID"
[ "$DRIVE_FOLDER" = MyDrive/sim ] || RUN_FLAGS="$RUN_FLAGS --drive $DRIVE_FOLDER"
colab() { uv run --project "$REPO" --group colab colab "$@"; }   # the Colab CLI of the project's colab group
[ -f "$REPO/configs/experiments/$RUN_ID.yaml" ] || {
  echo "no experiment $RUN_ID: one of $(cd "$REPO/configs/experiments" && ls *.yaml | sed 's/\.yaml$//' | paste -sd ' ' -)"
  exit 1
}
# the experiment's city, segmentation variant and inference config (its model setup), as the pipeline reads them
read -r CITY VARIANT SETUP < <(cd "$REPO" && uv run --project "$REPO" python -c '
import sys; sys.path.insert(0, "src")
from common.settings_and_paths import CONFIGS, load_config
cfg = load_config(CONFIGS / "experiments" / f"{sys.argv[1]}.yaml")
print(cfg["city"], cfg["segmentation"]["variant"], cfg["inference"])' "$RUN_ID")
[ -n "$SETUP" ] || { echo "could not read the experiment $RUN_ID (python src/main.py shows why)"; exit 1; }
# all checked: remember them for the next command
printf 'step=%s\nexp=%s\ndrive=%s\n' "$STEP" "$RUN_ID" "$DRIVE_FOLDER" > "$REMEMBERED"
USING="--exp $RUN_ID --drive $DRIVE_FOLDER"
[ "$CMD" = pack ] || USING="--step $STEP $USING"   # pack does not use the step
echo "using $USING"
RUN=/content/sim-run                               # where the code is unpacked on the VM; archives hold repo paths
DATA="data/$CITY" PREP="data/$CITY/data_prep" SEG="data/$CITY/inference/segmentation/$VARIANT"
RATINGS="$DATA/inference/vlm/$SETUP"               # the ratings, shared by the experiments with this setup
RAW_OUTPUTS="$SEG/raw-model-outputs"
# one input archive per city, for both steps: data_prep's photos and tables, and the review decisions when made
INPUTS="inputs_${CITY}.tar.gz"
INPUT_PATHS="$PREP/nodes.csv $PREP/views-180.csv $PREP/views-180 $PREP/downloads/street-view-frames
             $PREP/downloads/street-view-frames.csv $PREP/downloads/street-view-metadata.csv"
[ -f "$REPO/$PREP/node-review-decisions.csv" ] && INPUT_PATHS="$INPUT_PATHS $PREP/node-review-decisions.csv"
FOLDER="" PIPELINE_STEP=""                         # the step's results folder and pipeline step; pack has none
case "$STEP" in
  seg)                                          # the models' raw outputs out
    PIPELINE_STEP=run_segmentation_models
    GPU="${GPU:-L4}"                            # --gpu chooses another
    FOLDER=$RAW_OUTPUTS
    DONE_COUNT="find $RUN/$FOLDER/sam3 -name '*.npz' -not -name '*.tmp.npz' 2>/dev/null | wc -l"
    DONE_LABEL="detector passes saved (photos x passes)"
    ;;
  vlm)                                          # the ratings out; the ratings so far come back with the results archive
    PIPELINE_STEP=rate_views_with_vlm
    GPU="${GPU:-G4}"                            # RTX PRO 6000, 96 GB: Qwen3-VL-32B in bf16 needs ~67 GB
    FOLDER=$RATINGS
    DONE_COUNT="echo \$(( \$(wc -l < $RUN/$FOLDER/view-ratings.csv 2>/dev/null || echo 1) - 1 ))"
    DONE_LABEL="views rated (saved every 25)"
    ;;
esac
RESULTS="${STEP}_${RUN_ID}.tar.gz"                # results archive in the Drive folder (repo-relative paths inside)
LOG="/content/${STEP}.log"

# run a bash snippet on the VM through `colab exec`; its output is printed when it ends.
# `colab exec` allows 30 s of execution unless given --timeout: long steps pass their own.
remote() {
  local cmd="$1"; shift
  local literal
  literal="$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$cmd")"
  printf '%s\n' \
    "import subprocess" \
    "r = subprocess.run($literal, shell=True, executable='/bin/bash', capture_output=True, text=True, errors='replace')" \
    "print(r.stdout + r.stderr)" \
    "if r.returncode: raise SystemExit(f'remote command failed ({r.returncode})')" \
  | colab exec -s "$SESSION" "$@"
}

save_cmd="tar czf $DRIVE/$RESULTS.tmp -C $RUN $FOLDER && mv $DRIVE/$RESULTS.tmp $DRIVE/$RESULTS"
# the package from the unpacked code (no pip install -e: it would touch Colab's own torch)
py="cd $RUN && export PYTHONPATH=$RUN/src"

case "$CMD" in
  pack)
    out="$PWD/$INPUTS"                             # the input archive goes to the directory pack is run from
    cd "$REPO"
    for f in $INPUT_PATHS; do
      [ -e "$f" ] || { echo "no $f: run data_prep first"; exit 1; }
    done
    # no macOS tags: Linux tar warns per file
    COPYFILE_DISABLE=1 tar --no-xattrs -czf "$out" $INPUT_PATHS
    git diff --quiet HEAD -- src configs || echo "warning: uncommitted changes in src/ or configs/ are not in the code archive"
    git archive -o "$CODE_ARCHIVE" HEAD
    rm -rf /tmp/sim_check && mkdir /tmp/sim_check && tar xzf "$out" -C /tmp/sim_check
    ok=1; for f in $INPUT_PATHS; do diff -rq "$f" "/tmp/sim_check/$f" > /dev/null || ok=0; done
    [ "$ok" = 1 ] && echo "input archive verified identical to the originals"; rm -rf /tmp/sim_check
    ls -lh "$out" "$CODE_ARCHIVE"
    echo "now upload $out to the Drive folder ${DRIVE#/content/drive/} (drive.google.com or the Drive app)"
    ;;
  up)
    [ -f "$CODE_ARCHIVE" ] || { echo "no $CODE_ARCHIVE: run pack here first (src/colab.sh pack)"; exit 1; }
    if colab status -s "$SESSION" 2>&1 | grep -q "not found"; then   # (status exits 0 either way)
      case "$GPU" in
        T4|L4|G4|H100|A100) ;;
        *) echo "--gpu $GPU: the Colab CLI offers T4, L4, G4, H100 and A100"; exit 1 ;;
      esac
      if [ "$STEP" = vlm ]; then          # the VLM in bf16 needs ~67 GB
        case "$GPU" in
          T4|L4) echo "--gpu $GPU is too small for the VLM (~67 GB needed): use G4 or H100"; exit 1 ;;
          A100) echo "warning: Colab's A100 may have 40 GB, too little for the VLM's ~67 GB" ;;
        esac
      fi
      echo "starting session $SESSION on $GPU"
      colab new -s "$SESSION" --gpu "$GPU"
    else
      echo "session $SESSION is running: reusing it, with the GPU it started on"
      [ "$STEP" = vlm ] && echo "warning: the VLM needs ~67 GB; if this session is an L4 or T4 (e.g. from the segmentation), down first"
    fi
    colab upload -s "$SESSION" "$CODE_ARCHIVE" sim_code.tar.gz
    # the first connection to a new VM sometimes drops ("Connection was lost"): try the mount twice
    for try in 1 2; do
      # colab exec exits 0 even when the code fails: judge by the output
      remote "mountpoint -q /content/drive && echo is-mounted" --timeout 60 2>&1 | grep -q is-mounted && break
      colab drivemount -s "$SESSION" && break
      [ "$try" = 2 ] && { echo "Drive mount failed twice: src/colab.sh up $RUN_FLAGS again"; exit 1; }
      echo "Drive mount failed; trying again"; sleep 10
    done
    # SAM 3 is gated: the VM needs your Hugging Face login (`.venv/bin/hf auth login` once, here). The token
    # goes up as a file, so it never appears in a command or the CLI's history.
    token="${HF_HOME:-$HOME/.cache/huggingface}/token"
    [ -s "$token" ] || { echo "SAM 3 is gated: log in first with .venv/bin/hf auth login"; exit 1; }
    colab upload -s "$SESSION" "$token" hf_token
    remote "mkdir -p /root/.cache/huggingface && mv /hf_token /root/.cache/huggingface/token && chmod 600 /root/.cache/huggingface/token && echo 'Hugging Face login copied to the VM (SAM 3)'"
    remote "set -e
      test -f $DRIVE/$INPUTS || { echo 'missing $DRIVE/$INPUTS: upload it to Drive first (src/colab.sh pack $RUN_FLAGS)'; exit 1; }
      mkdir -p $RUN && cd $RUN && tar xzf /sim_code.tar.gz      # colab upload puts relative paths at /, not /content
      tar xzf $DRIVE/$INPUTS -C $RUN
      if [ -f $DRIVE/$RESULTS ]; then tar xzf $DRIVE/$RESULTS -C $RUN && echo 'resuming from $DRIVE/$RESULTS'; fi
      pip install -q -U 'transformers>=5.22.2' accelerate peft pyyaml tqdm
      pip uninstall -y -q torchao 2>/dev/null || true   # Colab's old torchao makes peft refuse to load JEV's adapter
      nvidia-smi --query-gpu=name,memory.total --format=csv" \
      --timeout 1800
    ;;
  start)
    # whole jobs detached from this command's output (else colab exec waits for the run to end and blocks
    # every later exec on the VM); at the lowest CPU and disk priority, else they starve the VM's Jupyter
    # server and later execs time out (ReadTimeout)
    remote "nohup nice -n 19 ionice -c 3 bash -c '$py && python src/main.py $PIPELINE_STEP configs/experiments/$RUN_ID.yaml' > $LOG 2>&1 < /dev/null &
      nohup nice -n 19 ionice -c 3 sh -c 'while sleep 600; do $save_cmd; done' > /dev/null 2>&1 < /dev/null &
      echo started"
    ;;
  status)
    remote "tail -c 1500 $LOG; echo; echo '$DONE_LABEL:' \$($DONE_COUNT)"      # quoted: labels hold ( )
    ;;
  watch)
    # progress every 5 min until the step finishes (the run itself does not need it: a full run finished with no check
    # for 7 hours); don't run other
    # colab commands on the VM at the same time
    mark="$PIPELINE_STEP done: "
    echo "$(date +%H:%M) checking now, then every 5 min (a check can take up to 3 min)"
    while true; do
      # a hard 3-minute limit from this side: the CLI can wait forever on a reply that never comes (seen when
      # two clients talk to the VM at once), which would freeze the loop; a check that hangs is killed, and retried.
      # Its whole process group: the CLI left alive would hold this capture's pipe open and freeze it just the same
      out="$(perl -e '$t = shift; $p = fork; if (!$p) { setpgrp(0, 0); exec @ARGV }
        $SIG{ALRM} = sub { kill "TERM", -$p; exit 142 }; alarm $t; waitpid $p, 0; exit $? >> 8' 180 "$0" status $RUN_FLAGS 2>&1 || true)"
      [ -n "$out" ] || out="(no answer within 3 minutes; trying again)"
      echo "$(date +%H:%M) ${out##*$'\n'}"
      case "$out" in
        *"$mark"*) echo "done: src/colab.sh down $RUN_FLAGS"; break ;;
        *"not found"*|*"appears to be lost"*) echo "session gone: src/colab.sh up, then start, with $RUN_FLAGS (resumes from the last save)"; exit 1 ;;
        # anything else (e.g. "Connection was lost") is one failed check: the run goes on, try again
      esac
      sleep 300
    done
    ;;
  save)
    remote "$save_cmd && ls -lh $DRIVE/$RESULTS" --timeout 900
    ;;
  down)
    remote "$save_cmd && ls -lh $DRIVE/$RESULTS" --timeout 900
    colab stop -s "$SESSION"
    echo "results are in Drive: ${DRIVE#/content/drive/}/$RESULTS -- download it, then: src/colab.sh fetch <file> --step $STEP --exp $RUN_ID"
    ;;
  stop)
    remote "pkill -f '[s]rc/main.py $PIPELINE_STEP' || true; pkill -f 'while [s]leep' || true
      $save_cmd && echo 'stopped; results kept on the VM and saved to Drive'" --timeout 900
    ;;
  reset)
    remote "pkill -f '[s]rc/main.py $PIPELINE_STEP' || true; pkill -f 'while [s]leep' || true
      rm -rf $RUN/$FOLDER $DRIVE/$RESULTS $DRIVE/$RESULTS.tmp $LOG
      echo 'stopped; results deleted on the VM and in Drive'" --timeout 300
    ;;
  fetch)
    [ -f "$ARCHIVE" ] || { echo "usage: src/colab.sh fetch <file> --step $STEP --exp $RUN_ID (the file: the downloaded $RESULTS, or ${RESULTS%.gz} if the browser unpacked it)"; exit 1; }
    # it replaces the step's folder: refuse another step's or experiment's archive (the flags may be remembered)
    case "${ARCHIVE##*/}" in
      "${STEP}_${RUN_ID}".*|"${STEP}_${RUN_ID} "*) ;;
      *) echo "${ARCHIVE##*/} is not $RESULTS: give the --step and --exp it was made with"; exit 1 ;;
    esac
    rm -rf "${REPO:?}/$FOLDER"                     # the archive holds the run's whole folder: no leftovers
    tar xf "$ARCHIVE" -C "$REPO"                   # tar detects whether it is gzipped
    echo "unpacked into $FOLDER; next: python src/main.py configs/experiments/$RUN_ID.yaml"
    ;;
  *)
    usage; exit 1
    ;;
esac
