"""
================================================================================
LETG-d REVISION SUPPORT — R1-3 parameter sensitivity + R2-4 pre-onset alarm rate
================================================================================
Run:   python letgd_r13_r24_sensitivity.py
Needs: simulation_main.py in the same folder.
Out:   letgd_sensitivity.json

WHAT THIS DOES
  Pools (gamma traces) are generated ONCE per (condition, repetition) and cached;
  every detector variant is then evaluated on the SAME cached traces, so the
  27-combo sensitivity costs barely more than one ROC run.

  Part A (R2-4): at the default operating point (lam=0.15, slack=0.5, W=15, h=14),
    pre-onset alarm rate = attack traces whose FIRST alarm precedes onset / all
    attack traces. Also re-derives TPR/FPR as a consistency anchor against the
    published 0.98/0.03.

  Part B (R1-3): 3x3x3 grid lam{0.10,0.15,0.25} x slack{0.25,0.50,0.75} x
    W{10,15,25}, h held at 14, on the ROC condition (20%, N=120) and the
    boundary condition (10%, N=120). Because slack rescales the CUSUM increment,
    each combo also reports a mini h-sweep {10,14,20} so the operating REGION
    (not a single h) is visible. Reports TPR / clean FPR / pre-onset rate /
    conditional latency (median-per-rep, mean across reps — same as paper).

Scoring rule (identical to letgd_all_in_one.py score()):
  TP  iff first alarm >= onset tick; pre-onset first alarm => miss (not FP);
  FPR from the separate non-attack pool; latency median over detected traces.
================================================================================
"""
import numpy as np, json, time
import simulation_main as S

# ---------------- KNOBS ----------------
N_PER_TRAJ = 25          # per pool per trajectory (paper value)
REPS       = 5           # paper value
BASE_TROLL = 0.05
CONDITIONS = [(0.20, 120), (0.10, 120)]   # (attack fraction, N): main + boundary
LAMS   = [0.10, 0.15, 0.25]
SLACKS = [0.25, 0.50, 0.75]
WARMS  = [10, 15, 25]
HS     = [10, 14, 20]
DEFAULT = dict(lam=0.15, slack=0.50, warmup=15, h=14)
# ---------------------------------------

TRAJS = [S.Circle(R=10), S.Square(), S.Lemniscate(), S.Zigzag()]

# --- trace generator: copied 1:1 from letgd_all_in_one.py (do not edit) ---
def trace_with_attack(traj, N=120, base_troll=0.05, attack_troll=0.30,
                      t_attack_frame=None, seed=0):
    rng = np.random.default_rng(seed)
    pos = traj.start(); vel = np.zeros(2); ph = [pos.copy()]
    pa = 0.; cd = np.array([1., 0.]); cg = 0.5
    gammas = []; t_attack_tick = None
    for f in range(S.FRAMES):
        if f % S.VOTE_INT == 0:
            troll = attack_troll if (t_attack_frame is not None and f >= t_attack_frame) else base_troll
            if t_attack_frame is not None and f >= t_attack_frame and t_attack_tick is None:
                t_attack_tick = len(gammas)
            di = max(0, len(ph)-1-S.DELAY_F); dp = ph[di]
            _, arc = traj.closest(dp); la = traj.at(arc + S.LOOK)
            idd = la - dp; nrm = np.linalg.norm(idd)
            if nrm > 1e-10: idd /= nrm
            ia = np.degrees(np.arctan2(idd[1], idd[0]))
            votes = S.gen_votes(ia, pa, troll, N, rng); pa = ia
            bl = S.DIRS[votes].mean(axis=0); cg = float(np.linalg.norm(bl))
            cd = bl/cg if cg > 1e-10 else np.array([1., 0.])
            gammas.append(cg)
        tv = cd * S.MSPD; vel += S.SMOOTH*(tv - vel); pos = pos + vel*S.DT; ph.append(pos.copy())
    return np.array(gammas), t_attack_tick

# --- parameterized detector: detect_adaptive with lam/slack/warmup exposed ---
def detect_adaptive_p(gamma, h, warmup, lam, slack):
    if len(gamma) < warmup + 5: return None
    mu = gamma[:warmup].mean()
    var = max(gamma[:warmup].var(), 1e-4)
    cusum = 0.0
    for i in range(warmup, len(gamma)):
        sd = np.sqrt(var) + 1e-6
        dev = (mu - gamma[i]) / sd
        cusum = max(0.0, cusum + dev - slack)
        if cusum > h: return i
        if cusum < h * 0.5:
            mu = (1-lam)*mu + lam*gamma[i]
            var = (1-lam)*var + lam*(gamma[i]-mu)**2
    return None

# --- pool cache: same seed pattern as letgd_all_in_one.build_pool ---
def build_pools(attack_troll, N):
    """Returns list over reps of (pos=[(gamma, ta)...], neg=[gamma...])."""
    reps = []
    for r in range(REPS):
        seed_off = r*1000+1
        rng = np.random.default_rng(seed_off)
        pos, neg = [], []
        for i in range(N_PER_TRAJ * len(TRAJS)):
            traj = TRAJS[i % len(TRAJS)]
            t = int(rng.uniform(0.30, 0.70)*S.FRAMES); t = (t//S.VOTE_INT)*S.VOTE_INT
            g, ta = trace_with_attack(traj, N=N, base_troll=BASE_TROLL,
                                      attack_troll=attack_troll, t_attack_frame=t,
                                      seed=seed_off + i*7 + 1)
            if ta is not None: pos.append((g, ta))
        for i in range(N_PER_TRAJ * len(TRAJS)):
            traj = TRAJS[i % len(TRAJS)]
            g, _ = trace_with_attack(traj, N=N, base_troll=BASE_TROLL,
                                     attack_troll=attack_troll, t_attack_frame=None,
                                     seed=seed_off + i*11 + 5000)
            neg.append(g)
        reps.append((pos, neg))
    return reps

def evaluate(pools, h, warmup, lam, slack):
    ts, fs, ls, pre = [], [], [], []
    for pos, neg in pools:
        tp = 0; lat = []; pre_ct = 0
        for g, ta in pos:
            d = detect_adaptive_p(g, h, warmup, lam, slack)
            if d is not None and d >= ta:
                tp += 1; lat.append(d - ta)
            elif d is not None and d < ta:
                pre_ct += 1                      # pre-onset first alarm => miss
        fp = sum(1 for g in neg
                 if detect_adaptive_p(g, h, warmup, lam, slack) is not None)
        ts.append(tp/len(pos)); fs.append(fp/len(neg))
        pre.append(pre_ct/len(pos))
        if lat: ls.append(float(np.median(lat)))
    return dict(tpr=round(float(np.mean(ts)),4), tpr_sd=round(float(np.std(ts)),4),
                fpr=round(float(np.mean(fs)),4),
                pre_onset=round(float(np.mean(pre)),4),
                lat=round(float(np.mean(ls)),2) if ls else None)

if __name__ == '__main__':
    out = {}
    for atk, N in CONDITIONS:
        key = f"atk{int(atk*100)}_N{N}"
        print(f"\n=== condition {key}: building {REPS} pool reps "
              f"({N_PER_TRAJ*4} attack + {N_PER_TRAJ*4} clean each) ===")
        t0 = time.time()
        pools = build_pools(atk, N)
        print(f"    pools built in {time.time()-t0:.0f}s")

        # Part A — default operating point + pre-onset rate (anchor check)
        m = evaluate(pools, DEFAULT['h'], DEFAULT['warmup'], DEFAULT['lam'], DEFAULT['slack'])
        out[key] = {'default': m, 'sensitivity': {}}
        print(f"    DEFAULT h=14 lam=.15 slack=.5 W=15 -> "
              f"TPR={m['tpr']:.2f} FPR={m['fpr']:.2f} pre_onset={m['pre_onset']:.3f} lat={m['lat']}")

        # Part B — 3x3x3 grid x mini h-sweep on cached pools
        for lam in LAMS:
            for sl in SLACKS:
                for W in WARMS:
                    for h in HS:
                        m = evaluate(pools, h, W, lam, sl)
                        out[key]['sensitivity'][f"lam{lam}_sl{sl}_W{W}_h{h}"] = m
        print(f"    sensitivity grid done ({len(LAMS)*len(SLACKS)*len(WARMS)*len(HS)} points)")

    json.dump(out, open('letgd_sensitivity.json','w'), indent=1)
    print("\nsaved letgd_sensitivity.json")
    print("Read-out: (1) default row pre_onset  (2) does the h=14 column keep "
          "TPR>=0.95 / FPR<=0.06 across the grid at atk20?  (3) at atk10, does the "
          "floor stay a floor for all combos (no combo rescues it)?")
