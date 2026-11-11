# Street Interface Matrix

Code for the Street Interface Matrix (SIM) study of Murray Hill, Manhattan: Madison to 1st Avenue, East 34th to 42nd
Street. How the study works and why: [`experiment-notes.md`](docs/experiment-notes.md).

## Setup

Needs Python 3.12+ and [uv](https://docs.astral.sh/uv/); then follow the setup in [`pipeline.md`](docs/pipeline.md).

## Steps

Run the steps as [`pipeline.md`](docs/pipeline.md) describes, and the GPU steps on Colab as
[`colab.md`](docs/colab.md) describes.

Run the tests:
```
uv run pytest
```

Every result per node is in `outputs/<run_id>/analysis/all-results-per-node.csv` and `.geojson`.

## Data

| Source | Terms |
|---|---|
| Street View | Downloaded with your own key, not redistributed (Google allows caching for 30 days); the views and label maps made from it are git-ignored |
| NYCWalks | Modelled weekday pedestrian volumes per sidewalk segment, 2018-2019 (Sevtsuk et al., *Nature Cities* 2025; MIT City Form Lab); not redistributed, since no licence is stated |
| New York's open data | Its sources are constants in `src/public_data/new_york/`, with their providers and terms |

Each step fetches its sources the first time it runs and logs them in `data/<city>/downloaded-sources.csv`.

## Layout

| Folder | Contents |
|---|---|
| `src/` | The code, one folder per stage (`data_prep`, `inference`, `public_data`, `metric`, `analysis`) and `common/` |
| `configs/` | `experiments/` (one file per run) and one folder of configs per stage |
| `data/<city>/` | What depends only on the city, shared by every experiment |
| `outputs/<run_id>/` | The experiment's metric and analysis |
| `tests/` | pytest |
| `docs/` | Running the repo: [`pipeline.md`](docs/pipeline.md), [`colab.md`](docs/colab.md); the experiment's record: [`experiment-notes.md`](docs/experiment-notes.md) (why it works as it does, what was checked, the literature behind the rated terms) |

## License

MIT: see [`LICENSE`](LICENSE).
