# LETG-d: Baseline-Free Real-Time Detection of Coordinated Manipulation in Crowd-Sourced Continuous Control

Code and data archive accompanying the manuscript **"Baseline-Free Real-Time Detection of Coordinated Manipulation in Crowd-Sourced Continuous Control."**

Author: **BongKeun Song**  
Affiliation: Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)

## Overview

This repository contains the simulation code, analysis scripts, and aggregated result data used for the LETG-d manuscript. The study evaluates a baseline-free real-time detector for coordinated manipulation in crowd-sourced continuous control (CSCC). The detector monitors the consensus strength, gamma, of an aggregate vote vector and combines:

- an exponentially weighted moving-average (EWMA) baseline,
- cumulative-sum (CUSUM) change detection,
- and a suspicion-triggered baseline freeze.

The repository supports the reported simulation-based results, including the adaptive-vs-fixed ROC comparison, robustness across attack ratios and crowd sizes, time-varying attack schedules, detector ablation, systematic attack-composition diagnostics, the parameter-sensitivity analysis, and the representative detector trace.

No human-subject data are included. All results are generated from simulated CSCC traces.

## Requirements

- Python 3.9+
- NumPy
- Matplotlib, for the representative-trace or other plotting scripts

Install the common requirements with:

```bash
pip install -r requirements.txt
```

## Repository contents

| File | Purpose | Manuscript linkage |
|---|---|---|
| `simulation_main.py` | Core CSCC simulation engine: trajectory classes, eight-direction voting, vector aggregation, participant vote model, and simulation constants. | Methods / Table I |
| `letgd_all_in_one.py` | Main script for ROC, robustness, and time-varying schedule experiments. | Fig. 2, Table II, Fig. 3, Table III, Fig. 4 |
| `letgd_roc.json` | Aggregated adaptive-vs-fixed detector operating-curve results. | Fig. 2 |
| `letgd_robustness.json` | Aggregated robustness results over adversarial fraction and crowd size. | Table II and Fig. 3 |
| `letgd_schedules.json` | Aggregated time-varying attack schedule results. | Table III and Fig. 4 |
| `letgd_ablation_v2_slowramp.py` | Revised ablation script emphasizing the slow-ramp setting and the effect of the baseline-freeze mechanism. | Table IV |
| `letgd_ablation.json` | Aggregated detector-component ablation results. | Table IV |
| `letgd_r13_r24_sensitivity.py` | Parameter-sensitivity experiment (revision): 3x3x3 grid over EWMA decay, CUSUM slack, and warm-up window, with a mini h-sweep, at the 20% and 10% conditions; also reports the pre-onset alarm rate at the default operating point. | Table V, Section V.F, and the pre-onset-alarm rule |
| `letgd_sensitivity.json` | Aggregated parameter-sensitivity and pre-onset-alarm results. | Table V and pre-onset-alarm rate (1.6%) |
| `letgd_systematic.py` | Systematic attack-composition and signal-diagnostic experiment: attack ratio, coherence, behavior mode, gamma-drop diagnostics, and CUSUM diagnostics. | Attack-composition and mechanism discussion |
| `sys_detection.json` | Detection results from the systematic attack-composition analysis. | Effect of attack composition |
| `sys_signal.json` | Signal-level diagnostics, including baseline gamma, fluctuation, and attack-induced gamma drop. | Low-fraction detectability discussion |
| `sys_cusum.json` | CUSUM accumulation diagnostics after attack onset. | Mechanism discussion |
| `make_fig4_trace.py` | Script used to generate the representative detector-dynamics trace. | Fig. 5 |
| `fig4_trace_data.json` | Underlying data for the representative detector trace. | Fig. 5 |
| `requirements.txt` | Python dependencies. | Reproducibility |
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

### 3. Parameter sensitivity and pre-onset alarm rate

```bash
python letgd_r13_r24_sensitivity.py
```

This regenerates:

- `letgd_sensitivity.json`

The script evaluates a 3x3x3 grid over EWMA decay (0.10, 0.15, 0.25), CUSUM slack (0.25, 0.50, 0.75), and warm-up window (10, 15, 25), each with a mini threshold sweep (h in 10, 14, 20), on the 20% ROC condition and the 10% boundary condition. At the default operating point (lam=0.15, slack=0.5, W=15, h=14) it also reports the pre-onset alarm rate. Pools (gamma traces) are generated once per condition and cached, and every parameter combination is evaluated on the same cached traces.

### 4. Systematic attack-composition and signal diagnostics

```bash
python letgd_systematic.py
```

This regenerates:

- `sys_detection.json`
- `sys_signal.json`
- `sys_cusum.json`

These outputs support the discussion of attack composition, directional coherence, low-fraction detectability, and gamma-drop signal strength.

### 5. Representative detector trace

```bash
python make_fig4_trace.py
```

This regenerates or updates:

- `fig4_trace_data.json`
- the representative detector-trace figure, depending on the local plotting settings in the script

The representative trace is a single measured simulation run used to illustrate detector dynamics. It is not an additional aggregate performance statistic.

## Notes on the parameter-sensitivity analysis

- The sensitivity analysis is exploratory. The default configuration is retained for the primary results and is not presented as globally optimal.
- The scoring rule matches the main experiments: a trace is a true positive only if the first alarm occurs at or after the programmed onset; a first alarm before onset is scored as a pre-onset false alarm (a missed post-onset detection), and the false-positive rate is measured on the separate non-attack pool.
- At the default operating point, the pre-onset first-alarm rate is 1.6% (8 of 500 attack traces across the two audited conditions), consistent with the value reported in the manuscript.

## Reporting notes

- Manuscript values are rounded summaries of the full-precision JSON outputs.
- Undefined latency values are represented as `null` in standards-compliant JSON files.
- The main experiments use independently regenerated simulation pools, as described in the manuscript Methods and Experimental Setup sections.
- The core pool design is 25 traces per trajectory across four trajectories, giving 100 attack traces and 100 non-attack traces per repetition.
- The main reported summaries use five independent pool repetitions.
- The non-attack condition retains a 5% background noisy/adversarial component; the detector is not tested against a perfectly clean baseline.
- Attack onset for step-like attack traces is sampled within the middle portion of the simulation to provide both pre-attack baseline observations and post-onset detection time.
- In the time-varying schedule experiment, the abrupt-step onset tick is sampled once per trace when the schedule is constructed, so the step is a single permanent transition.
- For the time-varying schedules, detection latency is measured from the first voting tick at which the adversarial fraction exceeds 10%, i.e. twice the 5% background level.
- One voting tick corresponds to 0.30 s at the 60-Hz simulation rate (18 frames per vote).
- Reported schedule detection rates are session-level: a trace counts as detected if its first alarm occurs at or after the onset, which for the periodic schedule may fall in a later elevated interval.
- All analyses are simulation-based. Transfer to real crowds requires separate human-participant validation.

## Relationship to the manuscript

The repository is intended to support the empirical results reported in the manuscript. The main mapping is:

- Fig. 2: `letgd_roc.json`
- Table II and Fig. 3: `letgd_robustness.json`
- Table III and Fig. 4: `letgd_schedules.json`
- Table IV: `letgd_ablation.json`
- Table V and Section V.F (parameter sensitivity), pre-onset-alarm rate: `letgd_sensitivity.json`
- Attack-composition and signal-diagnostic discussion: `sys_detection.json`, `sys_signal.json`, and `sys_cusum.json`
- Fig. 5 (representative trace): `fig4_trace_data.json` and `make_fig4_trace.py`

Figure 1 in the manuscript is the detector workflow and state diagram; it is a schematic and has no separate data file.

## Version history

### v1.3.1

- Fixed the abrupt-step schedule factory in `letgd_all_in_one.py`. In v1.3.0 the step onset tick was re-sampled on every voting tick, so the elevated level appeared intermittently before becoming persistent rather than as a single abrupt transition. The onset is now drawn once per trace, as intended.
- Regenerated `letgd_schedules.json` with the corrected factory. Only the `step` condition changed (TPR 0.982 to 0.998; median latency 2.0 to 1.0 voting ticks). The `ramp`, `burst`, `burst-weak`, and `periodic` conditions are numerically unchanged.
- Updated the schedule figure that reports these values.
- No other experiment, script, or result file is affected: the ROC, robustness, ablation, sensitivity, systematic-composition, and representative-trace outputs are unchanged.

### v1.3.0

- Initial archived release accompanying the manuscript revision.

## License and citation

This repository is released for academic reproducibility under the license stated in the repository and associated archive record. Please cite the accompanying manuscript and the archived DOI when using this code or data.
