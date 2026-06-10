
## Ablation and Figure 4 (added)

### Ablation study (TABLE IV)
- `letgd_ablation_v2_slowramp.py` — runs the four-variant ablation
  (fixed threshold / EWMA only / EWMA+CUSUM without freeze / full detector)
  on three attack conditions: abrupt step, fast ramp, and slow ramp.
  Requires only `simulation_main.py`. Run: `python letgd_ablation_v2_slowramp.py`
- `letgd_ablation.json` — output used for TABLE IV. The slow-ramp condition
  shows where the suspicion-freeze contributes (TPR 0.85->0.96, FPR 0.09->0.03).

### Figure 4 (representative detector trace)
- `make_fig4_trace.py` — generates the single measured ramp-attack trace shown
  in Figure 4 (consensus strength, EWMA baseline, CUSUM, freeze and alarm ticks),
  directly from the simulation engine. Run: `python make_fig4_trace.py`
- `fig4_trace_data.json` — the full time series for that trace (gamma, EWMA,
  CUSUM/h, adversarial fraction, plus the seed, freeze tick and alarm tick),
  provided so Figure 4 is fully reproducible rather than illustrative.

All detector parameters (warmup W=15, EWMA decay lambda=0.15, CUSUM slack
delta=0.5, threshold h swept) match the values reported in the paper.
