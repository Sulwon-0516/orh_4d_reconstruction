# Input data contract

## ⚠️ The published dataset does not match this contract yet

The HuggingFace dataset `Sulwon/anonymous_dataset_lih_orh_0920`
(revision `535c72f84dc124cfed76fb1be2c17e136fb331c3`) was inspected on 2026-09-20. **Its layout and
the layout this pipeline was developed against disagree in three ways that block a stranger from
running it.** Stated plainly rather than papered over:

| | Published on HuggingFace | What this pipeline needs |
|---|---|---|
| clip id | `C001` … `C017`, one `data/C0NN.tar` each (~4.3 GB) | ids are free-form; fine |
| contents | `C001/videos/<serial>.mp4`, `video_manifest.json`, `cameras.json`, `cam_param/{intrinsics,extrinsics}.json`, `timestamps/<serial>.json`; the JPEG tars contain `manifest.json` + `rgb/<serial>/<frame>.jpg` | `manifest.json` with the schema below, and extracted per-frame images |
| **foreground masks** | **not present in the published layout** | **required**, but see the box below — they affect DA3 grouping only |
| path style | unknown until extracted | must resolve on the local machine |

Three concrete consequences:

> ### What masks actually do — they do NOT make the output foreground-only
>
> Verified in the code, and worth stating plainly because the name misleads:
> - AmbiSuR's camera loader keeps the alpha channel separately and retains the **unmasked** RGB
>   (`third_party/AmbiSuR/scene/cameras.py:32,87`), and training's image loss uses that full RGB
>   (`train.py:174`).
> - The exporter's validity test is rendered **opacity and depth**, not the mask
>   (`orhsurf/_vendor/export_surface.py:138`).
>
> So the only thing a mask changes is the **camera ordering for the DA3 prior**, via the visual
> hull in `orhsurf/stages/da3_prior.py::visual_hull_center`. The exported cloud is the **whole
> scene**, not the subject. Making it foreground-only would be a new feature with its own A/B
> check, not a configuration change.

1. **Masks are the blocker.** Nothing in the published tree corresponds to our
   `masks_all/<serial>/<frame>.png`. Without them `orhsurf run` stops in `prep` with an explicit
   error. Either the masks must be published alongside the clips, or a mask-generation step must be
   added to the package. **This needs a decision — the package cannot invent it.**
2. **Our own `manifest.json` carries `mask_path: None`.** On the development machine the masks are
   in a *second* manifest, `manifest_fg.json`. Any published manifest must carry `mask_path` per
   frame, or the package must be told where masks live.
3. **Our manifests store absolute paths** (`/…/frames/<serial>/00040.png`). These do not survive a
   move to another machine. A published manifest must use paths relative to the clip directory, or
   the extraction step must rewrite them.

The dataset is also **still uploading**: `hevc_upload_status.json` reports `completed_clips: 4` of
`total_clips: 100`, with 17 JPEG archives retained. Total published size today ≈ **77 GiB**.

---

## Directory layout the pipeline expects

```
<ORHSURF_DATA_ROOT>/
  <clip-id>/
    manifest.json                     # the schema below
    frames/<serial>/<index:05d>.png   # undistorted-input frames, referenced by manifest
    masks/<serial>/<index:05d>.png    # RGBA; ALPHA is the foreground mask
```

Only `manifest.json` is read directly; `frame_path` and `mask_path` inside it locate everything
else, so any layout works as long as those resolve.

## `manifest.json` schema

Required top level:

| key | type | meaning |
|---|---|---|
| `sequence_id` | str | clip id |
| `valid_serials` | list[str] | cameras to use. The reference clip has 47 (of 48). |
| `calibrated_serials` | list[str] | **also required by the loader** (`_vendor/colmap_dataset.py:53`) |
| `conventions` | dict | **also required by the loader** |
| `timestamps` | list | one entry per frame; its length is the frame count |
| `cameras` | dict[serial → camera] | below |

Each camera:

| key | type | meaning |
|---|---|---|
| `width`, `height` | int | **must be identical across all cameras** — DA3 batches them together and centre-crops to the smallest, which would silently corrupt the prior |
| `K_original` | 3×3 | intrinsics of the raw distorted frame |
| `dist_params` | list[5] | OpenCV radtan `(k1,k2,p1,p2,k3)`; **`k3` must be 0** (asserted) |
| `K_undistort` | 3×3 | intrinsics after `cv2.undistort`; principal point need not be centred |
| `T_cam_from_world` | 4×4 | **world→camera**, metres. Bottom row `[0,0,0,1]`; rotation orthonormal. Both asserted. |
| `T_world_from_camera` | 4×4 | **also required by the loader**; must be the inverse of the above |
| `camera_center_world` | list[3] | camera centre in world coordinates |
| `valid` | bool | false ⇒ dropped |
| `frames` | list | per-frame entries |

Each `frames[i]`:

| key | meaning |
|---|---|
| `index` | must equal `i` (asserted) |
| `frame_path` | path to the RGB frame. **Consumed as given** — a relative path is NOT resolved against the manifest directory, so use absolute paths or run from the right cwd. |
| `mask_path` | RGBA PNG whose **ALPHA channel** is the foreground mask. Required. |

## Conventions, stated because getting them wrong fails silently

- **Extrinsics are world→camera.** `X_cam = T_cam_from_world @ X_world`. The inverse convention
  produces a reconstruction that looks plausible from one view and is wrong everywhere else.
- **Units are metres.** `multi_view_max_dis = 1.5` is 1.5 m; the isolation gate's 5 mm is 5 mm. A
  clip in millimetres would pass every assertion and produce nonsense.
- **Masks:** ALPHA channel of an RGBA PNG, thresholded at 128 after undistortion.
- An **all-zero mask is legal** — with 47 cameras ringing a room the subject is genuinely outside
  some of them (measured: 4/47 at frame 40 of the reference clip). At least 8 non-empty masks are
  required, because that is the visual hull's own `min_vis` gate.

## Two different undistorted pinholes — do not mix them

| | used for | principal point | size |
|---|---|---|---|
| `cv2.undistort(K_original, dist, None, K_undistort)` | **DA3 input** | wherever calibration put it | full, identical across views |
| COLMAP `image_undistorter`, then re-centred | **AmbiSuR scene** | forced to exactly (W/2, H/2) | cropped, differs per view |

They share the same camera **pose**, which is what makes the rewarp between them a pure,
depth-independent homography (`orhsurf/stages/da3_prior.py::rewarp`). AmbiSuR renders a symmetric
frustum and hardcodes the principal point to the image centre, which is why the scene must be
re-centred at all.

## Output

```
<out>/<clip-id>/<frame:05d>/
    surface.npz           xyz (N,3) f4 · normal (N,3) f4 · rgb (N,3) u1
                          confidence (N,) f4 · observed (N,) bool · support (N,) i16
    surface_export.json   point counts and every filter parameter actually used
    provenance.json       full recipe, stage timings, DA3 metadata
    metadata.json
    _DONE.json            written LAST. Its absence means the frame is not finished.
    surface.ply           only with --write-ply
```

World frame, metres, same convention as the input. `support` is the number of cameras that agreed
on a point; `confidence` is that rescaled to [0,1].
