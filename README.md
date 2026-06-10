# LETG-d: Baseline-Free Real-Time Detection of Coordinated Manipulation in Crowd-Sourced Continuous Control

Code and data archive accompanying the manuscript **"Baseline-Free Real-Time Detection of Coordinated Manipulation in Crowd-Sourced Continuous Control."**

Author: **BongKeun Song**  
Affiliation: Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)

## Overview

This repository contains the simulation code, analysis scripts, and aggregated result data used for the LETG-d manuscript. The study evaluates a baseline-free real-time detector for coordinated manipulation in crowd-sourced continuous control (CSCC). The detector monitors the consensus strength, gamma, of an aggregate vote vector and combines:

- an exponentially weighted moving-average (EWMA) baseline,
- cumulative-sum (CUSUM) change detection,
- and a suspicion-triggered baseline freeze.

The repository supports the reported simulation-based results, including ROC comparison, robustness across attack ratios and crowd sizes, time-varying attack schedules, detector ablation, systematic attack-composition diagnostics, and the representative detector trace shown as Figure 4 in the manuscript.

No human-subject data are included. All results are generated from simulated CSCC traces.

## Requirements

- Python 3.9+
- NumPy
- SciPy, if required by the local `simulation_main.py` environment
- Matplotlib, for Figure 4 or other plotting scripts

Install the common requirements with:

```bash
pip install numpy scipy matplotlib
```

## Repository contents

| File | Purpose | Manuscript linkage |
|---|---|---|
| `simulation_main.py` | Core CSCC simulation engine: trajectory classes, eight-direction voting, vector aggregation, participant vote model, and simulation constants. | Methods / Table I |
| `letgd_all_in_one.py` | Main script for ROC, robustness, and time-varying schedule experiments. | Fig. 1, Table II, Fig. 2, Table III, Fig. 3 |
| `letgd_roc.json` | Aggregated adaptive-vs-fixed detector operating curve results. | Fig. 1 and threshold-sensitivity text |
| `letgd_robustness.json` | Aggregated robustness results over adversarial fraction and crowd size. | Table II and Fig. 2 |
| `letgd_schedules.json` | Aggregated time-varying attack schedule results. | Table III and Fig. 3 |
| `letgd_ablation_v2_slowramp.py` | Revised ablation script emphasizing the slow-ramp setting and the effect of the baseline-freeze mechanism. | Table IV |
| `letgd_ablation.json` | Aggregated detector-component ablation results. | Table IV |
| `letgd_systematic.py` | Systematic attack-composition and signal-diagnostic experiment: attack ratio, coherence, behavior mode, gamma-drop diagnostics, and CUSUM diagnostics. | Attack-composition and mechanism discussion |
| `sys_detection.json` | Detection results from the systematic attack-composition analysis. | Effect of attack composition |
| `sys_signal.json` | Signal-level diagnostics, including baseline gamma, fluctuation, and attack-induced gamma drop. | Low-fraction detectability discussion |
| `sys_cusum.json` | CUSUM accumulation diagnostics after attack onset. | Mechanism discussion |
| `make_fig4_trace.py` | Script used to generate the representative detector-dynamics trace. | Fig. 4 |
| `fig4_trace_data.json` | Underlying data for the representative Figure 4 detector trace. | Fig. 4 |
| `LETGd_IEEEAccess.docx` | Manuscript file included for reference. | Main manuscript |
| `CITATION.cff` | Citation metadata for the repository. | Repository citation |
| `LICENSE` | License file. | Reuse terms |

## Reproduction guide

Run all commands from the repository root, with `simulation_main.py` in the same directory as the analysis scripts.

### 1. Core ROC, robustness, and schedule experiments

```bash
python letgd_all_in_one.py
```

This regenerates:

- `letgd_roc.json`
- `letgd_robustness.json`
- `letgd_schedules.json`

These files support the adaptive-vs-fixed ROC comparison, the robustness grid, and the step/ramp/burst/periodic schedule analysis.

### 2. Detector ablation

```bash
python letgd_ablation_v2_slowramp.py
```

This regenerates:

- `letgd_ablation.json`

The ablation compares the fixed threshold, EWMA-only detector, EWMA+CUSUM without freeze, and the full EWMA+CUSUM+freeze detector. The slow-ramp case is included to test whether the freeze prevents a slowly drifting attack from being absorbed into the adaptive baseline.

### 3. Systematic attack-composition and signal diagnostics

```bash
python letgd_systematic.py
```

This regenerates:

- `sys_detection.json`
- `sys_signal.json`
- `sys_cusum.json`

These outputs support the discussion of attack composition, directional coherence, low-fraction detectability, and gamma-drop signal strength.

### 4. Representative Figure 4 detector trace

```bash
python make_fig4_trace.py
```

This regenerates or updates:

- `fig4_trace_data.json`
- the representative detector-trace figure, depending on the local plotting settings in the script

Figure 4 is a single measured simulation run used to illustrate detector dynamics. It is not an additional aggregate performance statistic.

## Reporting notes

- Manuscript values are rounded summaries of the full-precision JSON outputs.
- Undefined latency values are represented as `null` in standards-compliant JSON files.
- The main experiments use independently regenerated simulation pools, as described in the manuscript Methods section.
- The core pool design is 25 traces per trajectory across four trajectories, giving 100 attack traces and 100 non-attack traces per repetition.
- The main reported summaries use five independent pool repetitions.
- The non-attack condition retains a 5% background noisy/adversarial component; the detector is not tested against a perfectly clean baseline.
- Attack onset for step-like attack traces is sampled within the middle portion of the simulation to provide both pre-attack baseline observations and post-onset detection time.
- All analyses are simulation-based. Transfer to real crowds requires separate human-participant validation.

## Relationship to the manuscript

The repository is intended to support the empirical results reported in the manuscript. The main mapping is:

- Fig. 1 and threshold-sensitivity text: `letgd_roc.json`
- Table II and Fig. 2: `letgd_robustness.json`
- Table III and Fig. 3: `letgd_schedules.json`
- Table IV: `letgd_ablation.json`
- Attack-composition and signal-diagnostic discussion: `sys_detection.json`, `sys_signal.json`, and `sys_cusum.json`
- Fig. 4: `fig4_trace_data.json` and `make_fig4_trace.py`

## License and citation

This repository is released for academic reproducibility under the license stated in the repository and associated archive record. Please cite the accompanying manuscript and the archived DOI when using this code or data.
