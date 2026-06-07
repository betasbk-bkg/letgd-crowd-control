# LETG-d: Baseline-Free Real-Time Detection of Coordinated Manipulation in Crowd-Sourced Continuous Control

Code and data accompanying the paper.
Author: BongKeun Song (FAU Erlangen-Nürnberg, Department of Chemical and Biological Engineering).

## Overview

A baseline-free detector that monitors the consensus strength gamma of a
crowd-sourced continuous control (CSCC) system and flags coordinated
manipulation (troll/bot influx) in real time using an EWMA baseline + CUSUM +
suspicion-freeze scheme. This archive reproduces the paper's empirical results.

## Requirements

- Python 3.9+
- numpy
- matplotlib (only for figures)

```
pip install numpy matplotlib
```

## Files

| File | Purpose |
| :--- | :--- |
| `simulation_main.py` | CSCC engine: 8-direction voting, vector aggregation, four trajectories (circle, square, lemniscate, zigzag), honest/troll vote models. Core constants: MSPD=5.0, SMOOTH=0.2, DELAY_F=26, VOTE_INT=18, FRAMES=3900. |
| `letgd_all_in_one.py` | Generates the core result files (ROC, robustness, attack schedules). Defines the adaptive detector (EWMA+CUSUM+freeze) and a fixed-threshold baseline. |
| `letgd_systematic.py` | Systematic troll characterization: TPR/FPR/latency over attack ratio x concentration x behavior, signal measurement (Delta-gamma, sigma), and CUSUM trajectories. |

## Result files (JSON)

- `letgd_roc.json` — adaptive vs fixed ROC (mean +/- std over repetitions)
- `letgd_robustness.json` — TPR/FPR/latency over attack ratio x crowd size
- `letgd_schedules.json` — detection across step/ramp/burst/burst-weak/periodic schedules
- `sys_detection.json` — detection over attack ratio x concentration x troll behavior
- `sys_signal.json` — Delta-gamma, baseline sigma, effective sigma, mu0
- `sys_cusum.json` — CUSUM accumulation trajectories

## Reproduction

```
python letgd_all_in_one.py      # -> letgd_roc.json, letgd_robustness.json, letgd_schedules.json
python letgd_systematic.py      # -> sys_detection.json, sys_signal.json, sys_cusum.json
```

Each script imports `simulation_main.py`, which must be in the same folder.
Random seeds are fixed; reported values are means with standard deviations over
repeated independent pools.

## Notes

- All results are simulation-based. No human-subject data are involved.
- The companion theoretical paper derives the consensus-strength signal and the
  detectability phase transition from first principles using these same outputs.

## License

Released under CC-BY-4.0 (see LICENSE). Please cite the accompanying paper and
this archive (see CITATION.cff).
