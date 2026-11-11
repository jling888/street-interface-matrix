# Experiment Notes

Why the Murray Hill experiment's choices were made, and what was checked. Numbers are from Murray Hill unless stated.

## Models

| Role | Model | Why |
|---|---|---|
| Street segmentation | Mask2Former Swin-L, Mapillary Vistas | The base label map; current street view practice |
| Facade detail | Mask2Former Swin-L, ADE20K | Windows, doors, columns, stoops, trees, which Mapillary lacks |
| Text-prompted classes | SAM 3 | A mask for every match of a prompt (signboards, glazing, doors, stoops, benches, planters, hedges); replaced Grounding DINO + SAM 2.1 |
| VLM ratings | Qwen3-VL-32B-Instruct, bf16 | This paper's ratings; unquantised on one 80 GB card |
| VLM ratings, later paper | JEV-27B-VL | Ahead in the bake-off; kept as `murrayhill_jev`, to be rated with the current prompts |

| Not used | Why |
|---|---|
| Mask2Former, Cityscapes | 19 classes, a subset of Mapillary's; calls sidewalk road (84% on the tunnel approach); no curb class |
| OneFormer Swin-L, ADE20K | About +1 mIoU, not worth a second code path |
| MM-Grounding-DINO, LLMDet | Boxes missed and shifted; Grounding DINO base framed each window and door |
| Qwen2-VL-7B, 4-bit (earlier, 12 GB card) | Superseded; greenery vs GVI +0.73, enclosure vs VEI +0.48 |

### VLM bake-off (2026-10-23)

Nine earlier terms; Spearman ρ of each expected rung with its segmentation counterpart over the view's arc. Selection
on 100 views; the held-out check on the other 1,286 views (691 nodes), intervals resampling the 29 block faces.

| Term | JEV | Qwen3-VL-32B | JEV − Qwen3 [95%] |
|---|---|---|---|
| v_nat | 0.92 | 0.88 | +0.04 [+0.02, +0.06] |
| gvi_eye | 0.71 | 0.64 | +0.06 [+0.04, +0.11] |
| gmi | 0.56 | 0.55 | +0.01 [−0.03, +0.05] |
| v_built | 0.73 | 0.54 | +0.19 [+0.14, +0.26] |
| enclosure | 0.46 | 0.20 | +0.26 [+0.19, +0.34] |
| v_sign | 0.35 | 0.27 | +0.08 [+0.03, +0.13] |
| v_pave | 0.29 | 0.19 | +0.11 [+0.06, +0.15] |
| gfapi | 0.46 | 0.46 | −0.01 [−0.06, +0.05] |
| ias | 0.29 | 0.31 | −0.02 [−0.08, +0.05] |
| **Mean** | **0.531** | 0.450 | **+0.081 [+0.065, +0.099]** |

- JEV gains where the others barely read the photo (sidewalk, enclosure, signage); level on three terms, behind on none.
- The index barely depends on the model: node-mean ρ JEV vs Qwen3 0.80-0.92 per dimension, 0.87 for M.
- 5 rungs vs 7: no difference (0.423 vs 0.425), so 1-7 is kept.
- Both models are deterministic across reruns.
- Agreement with segmentation is not ground truth; human ratings on 50-100 views would be the decisive check.

## Steps

### Data preparation

| Step | Decision | Why |
|---|---|---|
| sample_street_nodes | Every named street in the polygon, a node every 20 m; ids `<street>_<seq>` | Regular, readable, stable while sampling does not change |
| | OSM as of `osm_date`, by bounding box, cached | Reproducible; Overpass times out on dated polygon queries |
| | No Street View snapping | It would tie the frame to Google's current panoramas |
| download_street_view | Metadata first; panoramas over 15 m from the node rejected; requests by `pano_id`; user uploads not downloaded | Metadata is free; a coordinate request can snap headings to different panoramas |
| stitch_180_views | Four 90-degree frames reprojected into one 1280 px strip per walking direction, along the node's own heading | The frames' own resolution (~7.1 px/degree); curved streets render along the street |

### Inference

| Step | Decision | Why |
|---|---|---|
| run_segmentation_models | Three models, raw outputs saved (~60 KB per photo) | One model gave wobbly boundaries and missed classes; fusion can then change on CPU |
| fuse_label_maps | Classes, priorities and `may_replace` in `segmentation_classes.yaml`, not code | A window can split a facade, never a tree |
| rate_views_with_vlm | One term per prompt | A twelve-field schema collapsed (ρ 0.40 vs 0.79 alone) |
| | All seven rungs named | With only the ends named, 4 was never chosen |
| | Logits at a forced `{"field": ` prefix | Same as greedy decoding, keeps the distribution, deterministic |
| | No place name; images unedited, uncapped | A city name sets a prior; painting out the camera mast shifted unrelated terms |
| calculate_segmentation_pixel_shares | Term shares per one-degree bearing, whole photo and eye band (0-15° below the horizon), solid-angle weighted | Ratings compare to their own arc; corners would count 2.7x |
| | View indices GVI, GVI_eye_seg, SVF_band, BVF_band, VEI, PVI_band; no road view | The road share mostly measures where the car is |
| | Standard errors clustered on `face_id` | Moran's I about 0.8 for GVI and VEI |

| Segmentation finding | Fix |
|---|---|
| Bare trees: crown area labelled vegetation | Trees count whole, whatever the season (most panoramas are from April) |
| Scaffold fired on plain ground floors, erased windows | Scaffold and sidewalk-shed classes dropped |
| Hedge detections were tree crowns | Hedge needs the street-level gate |
| Shopfront glass as awning, crowns as green wall | Glazing, door and signboard above awning; green_wall dropped (0/3 correct) |
| Glass curtain walls | Stay building: only see-through windows are glazing |
| Lawns | `ground_cover`, out of V_nat as the paper defines it; GVI keeps them |

Known limits: few stoops and no class for ledges or low walls (IAS comes from the VLM); near pavement is sometimes
labelled sidewalk, or sidewalk roadway.

### Public data

| Step | Decision | Why |
|---|---|---|
| measure_street_canyons | W measured facade to facade, by nine parallel rays across one node spacing | Typology-assigned W made the H/W contrast definitional; adjacent nodes tile the street |
| | An empty side is re-probed; still empty is `open_one_side` | Genuinely open, unless it points down a cross street |
| | Corner nodes inherit H/W from the same street within 28 m | A corner has no facing facade |
| measure_space_syntax | Angular segment analysis at metric radii on street centrelines | OSM's walk network has two sidewalks per street: crossing counts as two turns |
| | Checked against depthmapX 0.9.1 on 1,858 segments | Integration rank agreement 1.00; choice 0.99 (values about half depthmapX's) |

### Metric

| Choice | Value |
|---|---|
| Form | Human Development Index: M = geometric mean of three dimensions (Identity, Dependence, Comfort), each 1 + 6 x the mean of its indicators |
| Indicators | A rating scaled between rungs 1 and 7, a record between fixed goalposts (ln(1 + x) where `log: true`), or half each; 29 of 31 terms |
| Ratings | Expected value over the seven rungs, on [0, 1] by (EV − 1) / 6 |
| Weights | Equal; PCA weights and weight and exponent robustness are diagnostics |
| Checks, not in M | enclosure, v_built, nat_built (validate_vlm_ratings) |

| Dimension design | Dimensions | Indicators | Terms in M | Records in M | Mean \|ρ\| between dimensions | α per dimension |
|---|---|---|---|---|---|---|
| previous (paper_revised) | Imageability / Identity / Dependence | 9 | 9 | 0 | 0.25 | 0.90 / 0.28 / 0.38 |
| image_form_stay | Image / Form / Stay | 22 | 8 | 14 | 0.18 | 0.19 / 0.56 / 0.32 |
| protection_comfort_enjoyment | Protection / Comfort / Enjoyment | 22 | 8 | 14 | 0.16 | 0.07 / 0.15 / 0.56 |
| identity_dependence_nature | Identity / Dependence / Nature | 21 | 29 | 16 | 0.15 | 0.64 / 0.40 / 0.85 |
| identity_dependence_comfort_first | Identity / Dependence / Comfort | 21 | 29 | 16 | 0.18 | 0.60 / 0.25 / 0.74 |
| **identity_dependence_comfort (in use)** | Identity / Dependence / Comfort | 21 | 29 | 16 | 0.21 | 0.64 / 0.20 / 0.81 |

### Analysis

| Choice | Why |
|---|---|
| Network controls read as a block | Largest VIF 37.5 (choice at 800 m) |
| Conley errors and residual Moran's I on network distance | A node's two views are at distance 0 |
| Lingering = footfall × (M / 7)^s at s = 0.5, 1, 2, an output only | No lingering data exists to estimate s |

## Literature behind the rated terms

| Source | What the study takes | Limit |
|---|---|---|
| Lynch 1960 | Local identity (signs, storefronts, trees) is the part of imageability one photo shows | Legibility, districts, serial vision need a mental map or a sequence |
| Ewing & Handy 2009 | Imageability, enclosure, human scale, transparency, complexity; item wording | Expert ICC 0.49-0.58; enclosure measures poorly in the field |
| Ewing & Clemente 2013, Ewing 2016 (NYC) | Windows, active frontage, street furniture tied to pedestrian counts | Footfall, not staying |
| Harvey 2015 (NYC) | Tree canopy, buildings per length, H/W predict perceived safety | Safety only |
| Gehl, 12 criteria | Seating, path, active edges, details, greenery are visible | Sound, air, climate are not |
| Mehta 2006 (Boston) | Seating, community places, personalised storefronts, sidewalk width predict staying | Cross-sectional, not Manhattan |
| Audits (MAPS, PEDS, Healthy Streets, Active Design) | Concrete item wording | Appearance items the least reliable |

| Feature | Evidence on staying |
|---|---|
| Seating, with food or community uses | Strongest (Mehta, Whyte) |
| Shade, microclimate | Good, seasonal |
| Active frontage, enclosure | Only via footfall or collinear studies |
| Lighting, cleanliness, noise, buffers, art, clutter | Little or none |

| Rule | Basis |
|---|---|
| Ratings describe visible setting attributes, not place identity or dependence | Both are person-place bonds measured by surveys |
| Dependence: activity-supporting features; Identity: historic and symbolic cues | The literature's mapping; Identity has the weakest warrant |
| No linear sign for enclosure | The H/W response is non-monotonic |
| The index is formative, claimed on conceptual grounds; low α is no objection | Grouping by correlations is exploration, kept apart from the outcome |
| Terms, prompts, filters and thresholds fixed before outcomes | Choosing them after can yield almost any result |
| VLMs rate visible physical properties; not appraisals, small counts or absent objects | They lean positive, compress the range, and vary by model |

| Rejected terms | Why |
|---|---|
| Legibility, linkage, coherence | ICC 0.34-0.38; whole-city qualities |
| Tidiness, litter, graffiti, maintenance | Time-varying, unreliable, read as wealth |
| Liveliness, safety, wealth, beauty | Model priors; can draw on people in frame |
| Pedestrians, vendors, occupied seats | People are not detected |
| Sky openness, H/W, height | Footprints and segmentation measure them: validators |
| Sidewalk sheds, transit stops | Records measure them |
| Traffic buffer | No staying evidence; the first to add if wanted |
| Public art, water, plazas, lighting | Too rare or unreliable |
| Noise, air, climate, density, night use | Not visible |

Do not cite without a DOI or PDF (not found, 2026-11-02): Gong et al. 2025, Ma, Yu & Zhao 2024, Xia, Gong & Dong 2023,
Zhang, Liu & Chen 2025, Oke & Voogt 2023, Tschumi 2024. Unverified: Gehl's "seven times more stops", the Punter
sense-of-place attribution, Harvey's "appeal". Koolhaas and Hays do not support enclosure as an identity anchor.
No VLM has been validated on these terms against humans in Manhattan.
