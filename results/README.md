# Results

The final Street Interface Matrix (SIM) scores and figures of each experiment, copied from `outputs/<experiment>/`
(git-ignored, rewritten on every run) when they were final. The raw data behind them (Street View imagery, model
outputs, downloaded public data) is not in the repository.

| Folder | Experiment | Metric |
|---|---|---|
| `murrayhill_qwen3_v1/` | `configs/experiments/murrayhill_qwen3_v1.yaml` | SIM v1, the previous paper's: Imageability / Identity / Dependence, from ten VLM ratings |
| `murrayhill_qwen3_v2/` | `configs/experiments/murrayhill_qwen3_v2.yaml` | SIM v2: Identity / Dependence / Comfort, from 29 of the 31 VLM ratings and city records |

Both: Murray Hill, Manhattan; 716 usable nodes, two 180-degree views each (1,432 views); rated with
Qwen3-VL-32B-Instruct.

Each folder holds:
- `scores-per-usable-view.csv`: one row per usable view; the rated terms on 0-1, the records the indicators read,
  each indicator (`index_<name>`, 0-1), each pillar (`<pillar>_raw` on 0-1, `<pillar>` on 1-7) and `M`, the SIM
  score: the geometric mean of the pillars, on 1-7. Join views to nodes on `node_id`.
- `figures/`: the figures made by pipeline step `make_figures`.

## Regenerating

```
uv run python src/main.py configs/experiments/murrayhill_qwen3_v2.yaml      # every step (docs/pipeline.md)
uv run python src/main.py configs/experiments/murrayhill_qwen3_v1.yaml metric
uv run python src/main.py configs/experiments/murrayhill_qwen3_v1.yaml analysis
```

then copy `outputs/<experiment>/metric/street_interface_matrix/scores-per-usable-view.csv` and
`outputs/<experiment>/analysis/figures/` here. `outputs/<experiment>/metric/street_interface_matrix/column-dictionary.md`
describes every column.

## Sources

- Street View imagery: Google (not redistributed).
- Pedestrian volume (figures only): NYCWalks, Sevtsuk et al., *Nature Cities* 2025, MIT City Form Lab.
- City records and building footprints: NYC Open Data.
- Street network: © OpenStreetMap contributors, ODbL.
