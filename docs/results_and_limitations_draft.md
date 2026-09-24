# Results and Limitations (draft)

We evaluate the CUSUM/jump-latch spoofing detector and the trust-weighted
sensor-fusion layer across an expanded matrix of 6 velocity profiles
(gentle, moderate, aggressive, circular, figure-eight, sustained-cruise) x
4 attack types (slow gradual drift, fast gradual drift, sudden jump,
intermittent on/off), 24 SITL trials in total. Detector numbers below are
unchanged from the original 24-trial run; fusion numbers reflect a
subsequent fix to the trust-weighting logic (a bounded-offset recovery
window layered on top of the pre-existing decay latch and floor) that was
re-run on all 24 trials and is the version reported here throughout.

## Detection performance by attack type

| Attack type | Precision | Recall | F1 | Mean FP count |
|---|---|---|---|---|
| sudden_jump | 0.972 | **1.000** | **0.985** | 16.3 |
| fast_drift | 0.777 | 0.920 | 0.805 | 162.5 |
| intermittent | 0.636 | 0.939 | 0.754 | 245.0 |
| slow_drift | 0.810 | 0.684 | 0.726 | 15.2 |

`sudden_jump` is reliably detected everywhere (recall = 1.0 in every
trial). `intermittent` drives the lowest precision (0.636) — every one of
its 6 trials produces false positives, unlike `slow_drift`/`sudden_jump`
where most trials produce none. `slow_drift` drives the lowest recall
(0.684), and that average hides a sharp split described below.

## Detection performance by velocity profile

| Velocity profile | Precision | Recall | F1 |
|---|---|---|---|
| gentle | 0.923 | 0.912 | 0.907 |
| sustained_cruise | 0.883 | 0.985 | 0.916 |
| aggressive | 0.820 | 0.958 | 0.874 |
| circular | 0.917 | 0.709 | 0.772 |
| moderate | 0.720 | 1.000 | 0.829 |
| figure_eight | **0.528** | 0.750 | **0.607** |

figure_eight is the clear outlier on F1 (0.607), driven by one complete
detector miss.

## Limitation 1: detector sensitivity to continuous-heading maneuvering

The detector's F1 collapses specifically on figure_eight (0.607, lowest of
any profile), and the mechanism is visible directly in the per-sample
detector output rather than inferred from summary statistics. For
`figure_eight_slow_drift`, precision/recall/F1 are all exactly 0.000 — a
complete non-detection across the full 130 s flight — and reading
`detector_output_figure_eight_slow_drift.csv` directly confirms this is a
genuine miss, not a labeling artifact: the CUSUM statistic never
approaches its own calibrated alarm threshold (`cusum_stat` peaks at
0.1199 against a calibrated `h` of 0.1510) for the full 896-sample attack
window. The pattern generalizes: `circular` and `figure_eight` are the two
profiles with continuous heading rotation, and both show a recall
collapse specifically on `slow_drift` (0.461 and 0.000 respectively)
while their `fast_drift` recall stays much higher (0.571 and 1.000) — the
same profiles catch the 4x-steeper 1.2 m/s drift rate fine but miss the
0.3 m/s one. Straight-line or single-direction profiles (gentle, moderate,
aggressive, sustained_cruise) hold slow_drift recall at 0.73-1.0 by
contrast. This isolates the failure to a drift-rate-vs-maneuvering-noise
interaction: continuous heading change injects noise into the slope-gap
signal the CUSUM statistic tracks, and at the slowest drift rate that
noise is large enough relative to the attack's own per-sample signal that
the CUSUM accumulator cannot sustain the climb to its threshold — a
genuine new limitation the original, non-continuously-turning validation
profiles never exposed.

## Limitation 2: the fusion recovery window trades an intermittent-attack win for a sudden-jump regression

The fix under evaluation adds a bounded-offset recovery window that lets
`trust_weight` climb back above its 0.15 floor when the detector's
jump-delta signal looks like a bounded, ended excursion rather than
ongoing gradual drift — aimed at the previous version's inability to
recognize a genuine GPS recovery during `intermittent`'s off-blocks. It
works as intended on that attack type: mean trust-weighted error across
the 6 `intermittent` trials drops from 50.8 m to 8.6 m (an 83% reduction),
and two trials flip from losing to GPS-only to beating it
(`aggressive_intermittent`: 30.76 m -> 2.17 m against a 5.29 m GPS-only
baseline; `figure_eight_intermittent`: 89.10 m -> 9.35 m against 9.96 m).
`sustained_cruise_intermittent` improves from 138.22 m to 12.92 m and
`circular_intermittent` from 29.14 m to 9.81 m, though neither quite
overtakes GPS-only (11.19 m and 7.22 m respectively).

That same recovery logic cannot distinguish "a bounded excursion that has
ended" from "a bounded excursion that is a permanent, ongoing sudden
jump," and on `sudden_jump` this produces one clear regression. Reading
`fusion_output_moderate_sudden_jump.csv` directly: at t=32.1s the GPS
jumps and is spoofed for the rest of the flight, trust correctly latches
to the 0.15 floor by t=35.2s, but at t=38.76s the recovery window opens
and trust climbs to 0.92 for roughly 3.4 s before dropping back to the
floor at t=42.3s — while `is_spoofed` is 1 throughout, i.e. GPS is still
actively spoofed the entire time the window is open. Because 0.92 exceeds
the fusion module's own `REANCHOR_TRUST_THRESHOLD` (0.90), this briefly
re-triggers the DR-reanchor-to-GPS logic that the design otherwise reserves
for genuine trust recovery, reanchoring dead-reckoning to the
still-spoofed GPS position and carrying that corrupted reference forward
for the rest of the flight. The result: `moderate_sudden_jump`'s
trust-weighted mean error regresses from 2.52 m (previous version, already
beating GPS-only's 3.00 m) to 21.08 m post-fix (trust_weighted_error_max
peaks at 93.98 m), a case that now loses to GPS-only where it previously
won. The same >0.90 spike occurs in `figure_eight_sudden_jump` (peak trust
0.92) and `circular_sudden_jump` (peak trust 1.0) post-attack, but there
the net effect is a large improvement rather than a regression (64.51 m ->
38.74 m and 37.11 m -> 31.82 m) — the reanchor-to-spoofed-position error
this introduces happens to land closer to the true trajectory than the
pure-floor DR drift it replaces in those two cases, but there is no
guarantee of that in general. Net across all 6 `sudden_jump` trials the
mean trust-weighted error moves only slightly (35.46 m -> 33.37 m) because
one regression and two improvements largely cancel, and the aggregate
count of trials where trust-weighted fusion beats GPS-only on mean error
moves from 8/24 to 9/24 overall (one `sudden_jump` trial flips from win to
loss, two `intermittent` trials flip from loss to win). The recovery
window is a net improvement on the attack type it targeted, but it shares
a code path (the 0.90 reanchor threshold) with a case it was not designed
for, and that interaction is not yet resolved.
