# Project instructions

- For converted video inputs, use all-foreground RGBA masks (alpha 255 everywhere) by default when `--masks` is omitted. The user explicitly withdrew the earlier prohibition on all-white masks.
- Explicit `--masks` inputs override this default. Do not fill gaps in user-supplied masks silently.
- Record the mask policy in the prepared manifest. All-foreground masks can change the visual-hull centre and DA3 grouping; do not claim equivalence to subject-mask results.
- Keep shared-storage reads scoped to known task paths. Do not run broad recursive disk scans or repeated usage checks.
