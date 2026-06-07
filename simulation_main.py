"""
========================================================================
Simulation Code for:
"Speed Over Strategy: Why Agent Velocity Dominates Aggregation Method
 in Crowd-Sourced Continuous Control"

[Author information removed for double-blind review]

Experiments:
  E1: Speed × Method ANOVA (circle, square trajectories)
  E2: Speed Sweep across troll ratios
  E3: Crowd-Size Sweep (ceiling effect, all 4 trajectories)
  E4: Model Fitting (multiplicative model)
  E5: Out-of-sample generalization

Usage: python3 simulation_main.py
Output: results.json
========================================================================
"""
import numpy as np
import time
import json
import sys
from scipy.optimize import curve_fit
from scipy import stats

# ====================================================================
# SECTION 0: SIMULATION CONSTANTS
# ====================================================================
DT        = 1/60       # 60 fps
MSPD      = 5.0        # max speed (m/s)
SMOOTH    = 0.2        # exponential smoothing factor (alpha)
WIN       = 0.3        # vote aggregation window (seconds)
DELAY_F   = 26         # input delay frames: 26 × (1/60s) ≈ 433 ms (realistic streaming latency)
DUR       = 65.0       # simulation duration (sec)
FRAMES    = int(DUR / DT)  # total frames = 3900
LOOK      = 2.0        # trajectory look-ahead distance (m)
VOTE_INT  = int(WIN / DT)  # 18 frames between votes

S2 = np.sqrt(2) / 2
DIRS = np.array([
    [1, 0], [S2, S2], [0, 1], [-S2, S2],
    [-1, 0], [-S2, -S2], [0, -1], [S2, -S2]
])
DIR_ANGLES = np.degrees(np.arctan2(DIRS[:, 1], DIRS[:, 0])) % 360

def angle_to_dir(angles):
    """Vectorized: continuous angles → nearest 8-direction index"""
    a = angles % 360
    diffs = np.abs(DIR_ANGLES[None, :] - a[:, None])
    diffs = np.minimum(diffs, 360 - diffs)
    return np.argmin(diffs, axis=1)


# ====================================================================
# SECTION 1: TRAJECTORIES (4 types, identical to ceiling_verify.py)
# ====================================================================
class Circle:
    def __init__(self, R=10):
        self.R = R
        self.circ = 2 * np.pi * R
        self.name = 'circle'

    def closest(self, p):
        t = np.arctan2(p[1], p[0])
        cp = self.R * np.array([np.cos(t), np.sin(t)])
        return cp, (t % (2 * np.pi)) * self.R

    def at(self, arc):
        t = arc / self.R
        return self.R * np.array([np.cos(t), np.sin(t)])

    def start(self):
        return np.array([self.R, 0.])


class Square:
    def __init__(self, h=10):
        self.name = 'square'
        self.c = np.array(
            [[h, 0], [h, h], [-h, h], [-h, -h], [h, -h], [h, 0.]],
            dtype=float
        )
        self.segs = [(self.c[i], self.c[i+1]) for i in range(5)]
        self.lens = [np.linalg.norm(b - a) for a, b in self.segs]
        self.circ = sum(self.lens)
        self.cum = np.array([0] + list(np.cumsum(self.lens)))

    def closest(self, p):
        bd, bp, ba = 1e10, self.c[0], 0.
        for i, (a, b) in enumerate(self.segs):
            v = b - a
            l2 = v @ v
            if l2 < 1e-10:
                continue
            t = np.clip((p - a) @ v / l2, 0, 1)
            pt = a + t * v
            d = np.linalg.norm(p - pt)
            if d < bd:
                bd, bp, ba = d, pt, self.cum[i] + t * self.lens[i]
        return bp, ba

    def at(self, arc):
        arc = arc % self.circ
        for i, (a, b) in enumerate(self.segs):
            if arc <= self.cum[i+1] + 1e-9:
                t = (arc - self.cum[i]) / self.lens[i]
                return a + np.clip(t, 0, 1) * (b - a)
        return self.c[-1]

    def start(self):
        return self.c[0].copy()


class Lemniscate:
    def __init__(self, a=7, n=800):
        self.name = 'lemniscate'
        ts = np.linspace(0, 2 * np.pi, n, endpoint=False)
        sn, cs = np.sin(ts), np.cos(ts)
        d = 1 + sn**2
        self.pts = np.column_stack([a * cs / d, a * sn * cs / d])
        dl = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.arcs = np.concatenate([[0], np.cumsum(dl)])
        self.circ = self.arcs[-1]

    def closest(self, p):
        d = np.linalg.norm(self.pts - p, axis=1)
        i = int(np.argmin(d))
        return self.pts[i].copy(), self.arcs[i]

    def at(self, arc):
        a = arc % self.circ
        i = min(int(np.searchsorted(self.arcs, a)), len(self.pts) - 1)
        return self.pts[i].copy()

    def start(self):
        return self.pts[0].copy()


class Zigzag:
    def __init__(self, amp=5, ns=10, sx=5):
        self.name = 'zigzag'
        pts = [np.array([0., 0.])]
        for i in range(ns):
            pts.append(np.array([(i+1) * sx, amp if i % 2 == 0 else 0.]))
        self.c = np.array(pts)
        self.segs = [(self.c[i], self.c[i+1]) for i in range(ns)]
        self.lens = [np.linalg.norm(b - a) for a, b in self.segs]
        self.circ = sum(self.lens)
        self.cum = np.array([0] + list(np.cumsum(self.lens)))

    def closest(self, p):
        bd, bp, ba = 1e10, self.c[0], 0.
        for i, (a, b) in enumerate(self.segs):
            v = b - a
            l2 = v @ v
            if l2 < 1e-10:
                continue
            t = np.clip((p - a) @ v / l2, 0, 1)
            pt = a + t * v
            d = np.linalg.norm(p - pt)
            if d < bd:
                bd, bp, ba = d, pt, self.cum[i] + t * self.lens[i]
        return bp, ba

    def at(self, arc):
        arc = arc % self.circ
        for i, (a, b) in enumerate(self.segs):
            if arc <= self.cum[i+1] + 1e-9:
                t = (arc - self.cum[i]) / self.lens[i]
                return a + np.clip(t, 0, 1) * (b - a)
        return self.c[-1]

    def start(self):
        return self.c[0].copy()


# ====================================================================
# SECTION 2: VOTE GENERATION (identical to ceiling_verify.py)
# ====================================================================
def gen_votes(ideal_angle, prev_angle, troll_ratio, N_agents, rng):
    """
    Generate N_agents votes given ideal direction.
    
    Agent composition (ratio preserved regardless of N):
      accurate: 70% of non-troll population, ±3° noise
      slow:     20% of non-troll population, lagged direction
      other:    remaining non-troll, ±30° noise
      troll:    troll_ratio × N, random 8-direction
    """
    # Troll count first (controlled variable)
    n_troll = round(N_agents * troll_ratio)
    n_troll = min(n_troll, N_agents)  # safety clamp
    
    # Distribute remaining among non-troll types
    remaining = N_agents - n_troll
    n_accurate = round(remaining * 0.7368)   # 70/95 ≈ 0.7368
    n_slow     = round(remaining * 0.2105)   # 20/95 ≈ 0.2105
    
    # Other absorbs rounding error — guarantees sum = N
    n_other = remaining - n_accurate - n_slow
    
    # If rounding made n_other negative, steal from n_accurate
    if n_other < 0:
        n_accurate += n_other  # reduce accurate
        n_other = 0

    # --- Agent composition check ---
    assert n_accurate + n_slow + n_troll + n_other == N_agents, \
        f"Agent sum mismatch: {n_accurate}+{n_slow}+{n_troll}+{n_other} != {N_agents}"
    assert n_accurate >= 0 and n_slow >= 0, \
        f"Negative agent count: acc={n_accurate}, slow={n_slow}"

    # Non-troll angles
    angles = np.empty(n_accurate + n_slow + n_other)
    idx = 0

    # Accurate agents: ideal ± 3°
    angles[idx:idx + n_accurate] = ideal_angle + rng.uniform(-3, 3, n_accurate)
    idx += n_accurate

    # Slow agents: lagged direction (20-50% behind)
    diff = ideal_angle - prev_angle
    if diff > 180: diff -= 360
    if diff < -180: diff += 360
    if n_slow > 0:
        lag = rng.uniform(0.2, 0.5, n_slow)
        angles[idx:idx + n_slow] = prev_angle + diff * (1 - lag)
        idx += n_slow

    # Other agents: ideal ± 30°
    if n_other > 0:
        angles[idx:idx + n_other] = ideal_angle + rng.uniform(-30, 30, n_other)
        idx += n_other

    # Quantize to 8 directions
    non_troll_votes = angle_to_dir(angles[:idx])

    # Troll votes: uniform random
    troll_votes = rng.integers(0, 8, n_troll) if n_troll > 0 else np.array([], dtype=int)

    return np.concatenate([non_troll_votes, troll_votes])


# ====================================================================
# SECTION 3: SIMULATION (identical to ceiling_verify.py + speed_override)
# ====================================================================
def simulate(traj, N_agents, troll_ratio, seed,
             method='fixed', speed_override=None):
    """
    Run one simulation.
    
    Parameters:
      traj:           trajectory object (Circle, Square, etc.)
      N_agents:       number of participants
      troll_ratio:    fraction of adversarial participants [0, 0.5]
      seed:           random seed for reproducibility
      method:         'fixed' | 'majority' | 'quadratic'
      speed_override: None → use MSPD(5.0), float → use that speed
    
    Returns:
      dict with rmse, mae, gamma_mean, gamma_std, speed_mean
    """
    rng = np.random.default_rng(seed)
    pos = traj.start()
    vel = np.zeros(2)
    pos_hist = [pos.copy()]
    prev_angle = 0.0
    cur_dir = np.array([1., 0.])
    cur_gamma = 0.5
    maj_dir = DIRS[0]

    V = speed_override if speed_override is not None else MSPD

    errors = np.empty(FRAMES)
    gammas = np.empty(FRAMES)
    speeds = np.empty(FRAMES)

    for f in range(FRAMES):
        # Vote at intervals (every VOTE_INT frames)
        if f % VOTE_INT == 0:
            # Delayed position
            delay_idx = max(0, len(pos_hist) - 1 - DELAY_F)
            delayed_pos = pos_hist[delay_idx]

            # Ideal direction from delayed position
            _, arc = traj.closest(delayed_pos)
            look_ahead_point = traj.at(arc + LOOK)
            ideal_dir = look_ahead_point - delayed_pos
            norm = np.linalg.norm(ideal_dir)
            if norm > 1e-10:
                ideal_dir /= norm
            ideal_angle = np.degrees(np.arctan2(ideal_dir[1], ideal_dir[0]))

            # Generate votes
            votes = gen_votes(ideal_angle, prev_angle, troll_ratio,
                              N_agents, rng)
            prev_angle = ideal_angle

            # Aggregate: vector average (blending)
            vote_vectors = DIRS[votes]
            blend = vote_vectors.mean(axis=0)
            cur_gamma = np.linalg.norm(blend)
            if cur_gamma > 1e-10:
                cur_dir = blend / cur_gamma
            else:
                cur_dir = np.array([1., 0.])

            # Majority (for majority method)
            counts = np.bincount(votes, minlength=8)
            maj_dir = DIRS[np.argmax(counts)]

        gammas[f] = cur_gamma

        # Velocity based on method
        if method == 'fixed':
            target_vel = cur_dir * V
        elif method == 'majority':
            target_vel = maj_dir * V
        elif method == 'quadratic':
            target_vel = cur_dir * (cur_gamma ** 2) * V
        elif method == 'linear':
            target_vel = cur_dir * cur_gamma * V
        elif method == 'sqrt':
            target_vel = cur_dir * np.sqrt(cur_gamma) * V
        else:
            target_vel = cur_dir * V

        # Lerp smoothing + move
        vel += SMOOTH * (target_vel - vel)
        pos = pos + vel * DT
        pos_hist.append(pos.copy())

        # Track error (distance to closest point on path)
        closest_point, _ = traj.closest(pos)
        errors[f] = np.linalg.norm(pos - closest_point)
        speeds[f] = np.linalg.norm(vel)

    return {
        'rmse':      float(np.sqrt(np.mean(errors**2))),
        'mae':       float(np.mean(errors)),
        'gamma_mean': float(np.mean(gammas)),
        'gamma_std':  float(np.std(gammas)),
        'speed_mean': float(np.mean(speeds)),
    }


# ====================================================================
# SECTION 4: MULTI-RUN WRAPPER
# ====================================================================
def run_condition(traj, N_agents, troll_ratio, mc_runs,
                  method='fixed', speed_override=None,
                  seed_base=31, seed_offset=0):
    """
    Run MC simulations and aggregate.
    Seed strategy: seed = i * seed_base + seed_offset
    """
    results = []
    for i in range(mc_runs):
        seed = i * seed_base + seed_offset
        r = simulate(traj, N_agents, troll_ratio, seed,
                     method=method, speed_override=speed_override)
        results.append(r)

    rmses = [r['rmse'] for r in results]
    return {
        'rmse_mean':  round(np.mean(rmses), 4),
        'rmse_std':   round(np.std(rmses), 4),
        'rmse_ci95':  round(1.96 * np.std(rmses) / np.sqrt(mc_runs), 4),
        'gamma_mean': round(np.mean([r['gamma_mean'] for r in results]), 4),
        'gamma_std':  round(np.mean([r['gamma_std'] for r in results]), 4),
        'speed_mean': round(np.mean([r['speed_mean'] for r in results]), 4),
        'mc_runs':    mc_runs,
    }


# ====================================================================
# SECTION 5: VALIDATION (must pass before experiments)
# ====================================================================
def run_validation():
    print("=" * 65)
    print("VALIDATION: Engine consistency check against existing data")
    print("=" * 65)

    c = Circle()
    sq = Square()
    lm = Lemniscate()
    zz = Zigzag()

    checks = [
        # (label, traj, N, troll, speed_ov, expected_rmse, tolerance)
        ("circle  tr=5%  N=150 v=5.0", c,  150, 0.05, None,  0.800, 0.03),
        ("circle  tr=20% N=50  v=5.0", c,   50, 0.20, None,  0.807, 0.03),
        ("circle  tr=40% N=200 v=5.0", c,  200, 0.40, None,  0.821, 0.03),
        ("square  tr=20% N=25  v=5.0", sq,  25, 0.20, None,  1.169, 0.05),
        ("circle  tr=10% N=150 v=2.0", c,  150, 0.10, 2.0,   0.243, 0.03),
        ("circle  tr=10% N=150 v=5.0", c,  150, 0.10, None,  0.806, 0.03),
    ]

    all_pass = True
    for label, tj, N, tr, spd, expected, tol in checks:
        r = run_condition(tj, N, tr, mc_runs=10,
                          speed_override=spd, seed_offset=N)
        actual = r['rmse_mean']
        ok = abs(actual - expected) < tol
        status = "PASS" if ok else "FAIL"
        print(f"  {label}: {actual:.3f} (expect {expected:.3f} +/-{tol}) [{status}]")
        if not ok:
            all_pass = False

    # Agent composition check
    print("\n  Agent composition check (N=100, tr=30%):")
    sc = (1 - 0.30) / 0.95
    na = round(100 * 0.70 * sc)
    ns = round(100 * 0.20 * sc)
    nt = round(100 * 0.30)
    no = max(0, 100 - na - ns - nt)
    total = na + ns + nt + no
    print(f"    accurate={na}, slow={ns}, troll={nt}, other={no}, sum={total}",
          "PASS" if total == 100 else "FAIL")
    if total != 100:
        all_pass = False

    # Speed override check
    r_slow = run_condition(c, 150, 0.10, mc_runs=8, speed_override=2.0)
    r_fast = run_condition(c, 150, 0.10, mc_runs=8, speed_override=5.0)
    speed_ok = r_slow['rmse_mean'] < r_fast['rmse_mean']
    print(f"\n  Speed override: v=2.0→{r_slow['rmse_mean']:.3f}, "
          f"v=5.0→{r_fast['rmse_mean']:.3f} "
          f"[{'PASS' if speed_ok else 'FAIL'}]")
    if not speed_ok:
        all_pass = False

    # Lemniscate / Zigzag basic check
    r_lm = simulate(lm, 150, 0.10, 42)
    r_zz = simulate(zz, 150, 0.10, 42)
    print(f"  Lemniscate: RMSE={r_lm['rmse']:.3f}, gamma={r_lm['gamma_mean']:.3f}")
    print(f"  Zigzag:     RMSE={r_zz['rmse']:.3f}, gamma={r_zz['gamma_mean']:.3f}")

    print(f"\n  VALIDATION {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    if not all_pass:
        print("  WARNING: Validation failures detected. Results may be inconsistent.")
    print()
    return all_pass


# ====================================================================
# SECTION 6: EXPERIMENT E2e — N Sweep × New Troll Ratios
# ====================================================================
def run_E2e(MC=15):
    print("=" * 65)
    print("E2e: N Sweep × New Troll Ratios (10%, 15%, 30%)")
    print(f"     Trajectories: circle, square | MC={MC}")
    print("     Control: method=fixed, speed=5.0")
    print("=" * 65)

    Ns = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
    trolls = [0.10, 0.15, 0.30]
    trajs = {'circle': Circle(), 'square': Square()}

    results = {}
    total = len(Ns) * len(trolls) * len(trajs)
    cnt = 0
    t0 = time.time()

    for tn, tj in trajs.items():
        for tr in trolls:
            print(f"\n  --- {tn}, troll={tr:.0%} ---")
            print(f"  {'N':>5} | {'RMSE':>8} | {'CI95':>8} | {'gamma':>6} | ETA")
            print(f"  {'-'*45}")

            for N in Ns:
                cnt += 1
                r = run_condition(tj, N, tr, MC, seed_offset=N)
                key = f"{tn}_tr{tr:.2f}_N{N}"
                results[key] = r

                elapsed = time.time() - t0
                eta = elapsed / cnt * (total - cnt)
                print(f"  {N:>5} | {r['rmse_mean']:>8.4f} | "
                      f"{r['rmse_ci95']:>8.4f} | {r['gamma_mean']:>6.3f} | "
                      f"{eta:.0f}s")

    print(f"\n  E2e done: {time.time()-t0:.0f}s, {cnt} conditions\n")
    return results


# ====================================================================
# SECTION 7: EXPERIMENT E2f — N Sweep × Lemniscate, Zigzag
# ====================================================================
def run_E2f(MC=15):
    print("=" * 65)
    print("E2f: N Sweep × Lemniscate + Zigzag (ALL troll ratios)")
    print(f"     MC={MC}")
    print("     Control: method=fixed, speed=5.0")
    print("=" * 65)

    Ns = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
    trolls = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    trajs = {'lemniscate': Lemniscate(), 'zigzag': Zigzag()}

    results = {}
    total = len(Ns) * len(trolls) * len(trajs)
    cnt = 0
    t0 = time.time()

    for tn, tj in trajs.items():
        for tr in trolls:
            print(f"\n  --- {tn}, troll={tr:.0%} ---")
            print(f"  {'N':>5} | {'RMSE':>8} | {'CI95':>8} | {'gamma':>6} | ETA")
            print(f"  {'-'*45}")

            for N in Ns:
                cnt += 1
                r = run_condition(tj, N, tr, MC, seed_offset=N)
                key = f"{tn}_tr{tr:.2f}_N{N}"
                results[key] = r

                elapsed = time.time() - t0
                eta = elapsed / cnt * (total - cnt)
                print(f"  {N:>5} | {r['rmse_mean']:>8.4f} | "
                      f"{r['rmse_ci95']:>8.4f} | {r['gamma_mean']:>6.3f} | "
                      f"{eta:.0f}s")

    print(f"\n  E2f done: {time.time()-t0:.0f}s, {cnt} conditions\n")
    return results


# ====================================================================
# SECTION 8: EXPERIMENT E1e — Speed Sweep × Troll
# ====================================================================
def run_E1e(MC=10):
    print("=" * 65)
    print("E1e: Speed Sweep × Troll Ratio")
    print(f"     N=150 fixed | MC={MC}")
    print("     Trajectories: circle, square")
    print("     Control: method=fixed, N=150")
    print("=" * 65)

    speeds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    trolls = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    N_FIXED = 150
    trajs = {'circle': Circle(), 'square': Square()}

    results = {}
    total = len(speeds) * len(trolls) * len(trajs)
    cnt = 0
    t0 = time.time()

    for tn, tj in trajs.items():
        for tr in trolls:
            print(f"\n  --- {tn}, troll={tr:.0%} ---")
            row_data = []
            for v in speeds:
                cnt += 1
                r = run_condition(tj, N_FIXED, tr, MC,
                                  speed_override=v,
                                  seed_base=997,
                                  seed_offset=int(v * 10))
                key = f"{tn}_tr{tr:.2f}_v{v:.1f}"
                results[key] = r
                row_data.append(f"v={v:.1f}:{r['rmse_mean']:.3f}")

            elapsed = time.time() - t0
            eta = elapsed / cnt * (total - cnt)
            print(f"  {' | '.join(row_data)} [{eta:.0f}s]")

    print(f"\n  E1e done: {time.time()-t0:.0f}s, {cnt} conditions\n")
    return results


# ====================================================================
# SECTION 9: EXPERIMENT E3d — New Condition Predictions
# ====================================================================
def run_E3d(MC=10):
    print("=" * 65)
    print("E3d: New Condition Prediction Validation")
    print(f"     MC={MC} | circle only")
    print("=" * 65)

    c = Circle()
    conditions = [
        # (label, N, troll, speed)
        ("v=5.0_N=35_tr=25%",   35, 0.25, 5.0),
        ("v=2.0_N=35_tr=25%",   35, 0.25, 2.0),
        ("v=3.0_N=80_tr=15%",   80, 0.15, 3.0),
        ("v=1.5_N=120_tr=35%", 120, 0.35, 1.5),
        ("v=4.0_N=60_tr=10%",   60, 0.10, 4.0),
        ("v=2.5_N=45_tr=45%",   45, 0.45, 2.5),
    ]

    results = {}
    for label, N, tr, v in conditions:
        r = run_condition(c, N, tr, MC,
                          speed_override=v,
                          seed_base=777,
                          seed_offset=N)
        results[label] = r
        print(f"  {label}: RMSE={r['rmse_mean']:.4f} +/- {r['rmse_ci95']:.4f}")

    print()
    return results


# ====================================================================
# SECTION 10: ANALYSIS — E3c Cross-Validation + Model Comparison
# ====================================================================
def run_analysis(e1e_results, e2e_results, e3d_results):
    print("=" * 65)
    print("ANALYSIS: Model fitting, cross-validation, prediction check")
    print("=" * 65)

    # --- Collect all data points: (v, N, p, RMSE) ---
    points = []

    # E1e speed sweep data (circle, N=150)
    for key, row in e1e_results.items():
        parts = key.split('_')
        traj = parts[0]
        if traj != 'circle':
            continue
        tr = float(parts[1].replace('tr', ''))
        v = float(parts[2].replace('v', ''))
        points.append((v, 150, tr, row['rmse_mean']))

    # Existing N sweep data (circle, v=5.0) — hardcoded from validated runs
    existing_n = {
        (0.05, 5): 0.819, (0.05, 10): 0.815, (0.05, 25): 0.807,
        (0.05, 50): 0.810, (0.05, 100): 0.798, (0.05, 200): 0.810,
        (0.20, 5): 0.935, (0.20, 10): 0.909, (0.20, 25): 0.833,
        (0.20, 50): 0.807, (0.20, 100): 0.819, (0.20, 200): 0.817,
        (0.40, 5): 1.135, (0.40, 10): 0.977, (0.40, 25): 0.903,
        (0.40, 50): 0.831, (0.40, 100): 0.837, (0.40, 200): 0.821,
    }
    for (tr, N), rmse in existing_n.items():
        points.append((5.0, N, tr, rmse))

    # E2e new troll N sweep data (circle, v=5.0)
    for key, row in e2e_results.items():
        parts = key.split('_')
        if parts[0] != 'circle':
            continue
        tr = float(parts[1].replace('tr', ''))
        N = int(parts[2].replace('N', ''))
        points.append((5.0, N, tr, row['rmse_mean']))

    data = np.array(points)
    np.random.seed(42)
    np.random.shuffle(data)
    v, N, p, rmse = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    print(f"\n  Total data points: {len(data)}")
    print(f"  v range: [{v.min():.1f}, {v.max():.1f}]")
    print(f"  N range: [{N.min():.0f}, {N.max():.0f}]")
    print(f"  p range: [{p.min():.2f}, {p.max():.2f}]")

    # --- Model definitions ---
    def model_additive(X, a0, a1, a2, a3):
        v, N, p = X
        return a0 + a1*v + a2*v**2 + a3*p/np.sqrt(N)

    def model_multiplicative(X, a0, a1, a2, a3):
        v, N, p = X
        return (a0 + a1*v + a2*v**2) * (1 + a3*p/np.sqrt(N))

    # --- Fit both models ---
    print(f"\n  --- Full-data model fit ---")
    model_results = {}

    for name, model, p0 in [
        ('Additive',        model_additive,       [0.5, -0.2, 0.06, 1.5]),
        ('Multiplicative',  model_multiplicative,  [0.5, -0.2, 0.06, 1.0]),
    ]:
        try:
            popt, _ = curve_fit(model, (v, N, p), rmse, p0=p0, maxfev=10000)
            pred = model((v, N, p), *popt)
            r2 = 1 - np.sum((rmse - pred)**2) / np.sum((rmse - np.mean(rmse))**2)
            mae = np.mean(np.abs(rmse - pred))
            model_results[name] = {'coeffs': [float(x) for x in popt],
                                   'r2': float(r2), 'mae': float(mae)}
            print(f"  {name}: R2={r2:.4f}, MAE={mae:.4f}")
            print(f"    coeffs: {[round(x, 5) for x in popt]}")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    # --- 5-fold Cross Validation ---
    print(f"\n  --- 5-fold Cross Validation ---")
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    fold_size = len(data) // 5

    cv_results = {}
    for name, model, p0 in [
        ('Additive',        model_additive,       [0.5, -0.2, 0.06, 1.5]),
        ('Multiplicative',  model_multiplicative,  [0.5, -0.2, 0.06, 1.0]),
    ]:
        fold_r2s = []
        fold_maes = []
        for fold in range(5):
            test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.concatenate([indices[:fold * fold_size],
                                        indices[(fold + 1) * fold_size:]])
            tr_d = data[train_idx]
            te_d = data[test_idx]
            try:
                po, _ = curve_fit(model,
                                  (tr_d[:, 0], tr_d[:, 1], tr_d[:, 2]),
                                  tr_d[:, 3], p0=p0, maxfev=10000)
                pred = model((te_d[:, 0], te_d[:, 1], te_d[:, 2]), *po)
                ss_res = np.sum((te_d[:, 3] - pred)**2)
                ss_tot = np.sum((te_d[:, 3] - np.mean(te_d[:, 3]))**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                mae = np.mean(np.abs(te_d[:, 3] - pred))
                fold_r2s.append(r2)
                fold_maes.append(mae)
            except:
                fold_r2s.append(0)
                fold_maes.append(1)

        cv_results[name] = {
            'r2_mean': float(np.mean(fold_r2s)),
            'r2_std':  float(np.std(fold_r2s)),
            'mae_mean': float(np.mean(fold_maes)),
            'fold_r2s': [float(x) for x in fold_r2s],
        }
        print(f"  {name}: R2={np.mean(fold_r2s):.4f} +/- {np.std(fold_r2s):.4f}, "
              f"MAE={np.mean(fold_maes):.4f}")

    # --- E3d prediction test (multiplicative model) ---
    print(f"\n  --- E3d Prediction Test (Multiplicative) ---")
    if 'Multiplicative' in model_results:
        coeffs = model_results['Multiplicative']['coeffs']
        e3d_predictions = {}

        print(f"  {'Condition':<25} {'Predicted':>9} {'Actual':>9} {'Error%':>8}")
        print(f"  {'-'*55}")

        errors_pct = []
        for label, row in e3d_results.items():
            parts = label.split('_')
            v_val = float(parts[0].split('=')[1])
            N_val = float(parts[1].split('=')[1])
            tr_val = float(parts[2].split('=')[1].replace('%', '')) / 100

            predicted = model_multiplicative((v_val, N_val, tr_val), *coeffs)
            actual = row['rmse_mean']
            err = abs(predicted - actual) / actual * 100
            errors_pct.append(err)
            status = "OK" if err < 15 else "WARN"

            print(f"  {label:<25} {predicted:>9.4f} {actual:>9.4f} {err:>7.1f}% [{status}]")

            e3d_predictions[label] = {
                'predicted': float(predicted),
                'actual': float(actual),
                'error_pct': float(err),
            }

        print(f"\n  Mean prediction error: {np.mean(errors_pct):.1f}%")
        print(f"  Within 15%: {sum(1 for e in errors_pct if e < 15)}/{len(errors_pct)}")
    else:
        e3d_predictions = {}

    # --- Optimal speed analysis ---
    print(f"\n  --- Optimal Speed v* ---")
    if 'Multiplicative' in model_results:
        c = model_results['Multiplicative']['coeffs']
        # base(v) = c[0] + c[1]*v + c[2]*v^2
        # v* = -c[1] / (2*c[2])
        if c[2] > 0:
            vstar = -c[1] / (2 * c[2])
            rmse_star = (c[0] + c[1]*vstar + c[2]*vstar**2)
            print(f"  v* = {vstar:.2f} m/s")
            print(f"  RMSE*(v*) = {rmse_star:.4f}")
            print(f"  Noise coefficient = {c[3]:.4f}")
        else:
            vstar = None
            print(f"  WARNING: quadratic term is negative, no minimum")

    return {
        'model_results': model_results,
        'cv_results': cv_results,
        'e3d_predictions': e3d_predictions,
        'n_data_points': len(data),
    }


# ====================================================================
# SECTION 11: CEILING ANALYSIS
# ====================================================================
def run_ceiling_analysis(e2e_results, e2f_results):
    print("=" * 65)
    print("CEILING ANALYSIS: a + b/sqrt(N) fit across all conditions")
    print("=" * 65)

    # Existing N sweep data
    existing = {
        ('circle', 0.05): {5:0.819,10:0.815,15:0.818,20:0.828,25:0.807,
                           30:0.812,40:0.819,50:0.810,75:0.803,100:0.798,150:0.800,200:0.810},
        ('circle', 0.20): {5:0.935,10:0.909,15:0.873,20:0.833,25:0.833,
                           30:0.832,40:0.820,50:0.807,75:0.822,100:0.819,150:0.807,200:0.817},
        ('circle', 0.40): {5:1.135,10:0.977,15:0.919,20:0.967,25:0.903,
                           30:0.878,40:0.854,50:0.831,75:0.843,100:0.837,150:0.820,200:0.821},
        ('square', 0.05): {5:1.160,10:1.162,15:1.165,20:1.168,25:1.162,
                           30:1.155,40:1.148,50:1.147,75:1.145,100:1.143,150:1.141,200:1.142},
        ('square', 0.20): {5:1.237,10:1.210,15:1.195,20:1.183,25:1.169,
                           30:1.174,40:1.163,50:1.158,75:1.152,100:1.150,150:1.148,200:1.147},
        ('square', 0.40): {5:1.327,10:1.265,15:1.238,20:1.223,25:1.195,
                           30:1.192,40:1.175,50:1.170,75:1.161,100:1.157,150:1.155,200:1.155},
    }

    # Merge with new data
    all_n_data = dict(existing)

    for results_dict in [e2e_results, e2f_results]:
        for key, row in results_dict.items():
            parts = key.split('_')
            traj = parts[0]
            tr = float(parts[1].replace('tr', ''))
            N = int(parts[2].replace('N', ''))
            dk = (traj, tr)
            if dk not in all_n_data:
                all_n_data[dk] = {}
            all_n_data[dk][N] = row['rmse_mean']

    # Fit ceiling model
    def ceiling_model(N, a, b):
        return a + b / np.sqrt(N)

    print(f"\n  {'Traj':<12} {'Troll':>5} {'a':>8} {'b':>8} {'R2':>7} "
          f"{'RMSE(5)':>8} {'RMSE(200)':>9} {'Improv':>7}")
    print(f"  {'-'*68}")

    ceiling_fits = []
    for traj in ['circle', 'square', 'lemniscate', 'zigzag']:
        for tr in sorted(set(k[1] for k in all_n_data if k[0] == traj)):
            d = all_n_data[(traj, tr)]
            ns = np.array(sorted(d.keys()))
            rs = np.array([d[n] for n in ns])

            if len(ns) < 4:
                continue

            try:
                popt, _ = curve_fit(ceiling_model, ns, rs, p0=[0.8, 0.5])
                pred = ceiling_model(ns, *popt)
                ss_res = np.sum((rs - pred)**2)
                ss_tot = np.sum((rs - np.mean(rs))**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0

                imp = (rs[0] - rs[-1]) / rs[0] * 100
                print(f"  {traj:<12} {tr:>4.0%} {popt[0]:>8.4f} {popt[1]:>8.4f} "
                      f"{r2:>7.3f} {rs[0]:>8.4f} {rs[-1]:>9.4f} {imp:>6.1f}%")

                ceiling_fits.append({
                    'traj': traj, 'troll': tr,
                    'a': float(popt[0]), 'b': float(popt[1]),
                    'r2': float(r2), 'improvement_pct': float(imp),
                })
            except Exception as e:
                print(f"  {traj:<12} {tr:>4.0%}  FIT FAILED: {e}")

    print()
    return ceiling_fits


# ====================================================================
# SECTION 12: MAIN
# ====================================================================
def main():
    t_start = time.time()
    print("\n" + "=" * 65)
    print("  IEEE Access Paper — Full-Scale Simulation")
    print("  Estimated runtime: 40-50 minutes")
    print("=" * 65 + "\n")

    # Step 0: Validation
    valid = run_validation()
    if not valid:
        print("WARNING: Validation failed. Continue anyway? (y/n)")
        # In batch mode, continue anyway
        # response = input().strip().lower()
        # if response != 'y': sys.exit(1)

    # Step 1: E2e — N sweep, new trolls
    e2e = run_E2e(MC=15)

    # Step 2: E2f — N sweep, new trajectories
    e2f = run_E2f(MC=15)

    # Step 3: E1e — Speed sweep
    e1e = run_E1e(MC=10)

    # Step 4: E3d — New condition predictions
    e3d = run_E3d(MC=10)

    # Step 5: Ceiling analysis
    ceiling = run_ceiling_analysis(e2e, e2f)

    # Step 6: Model analysis (E3c + E3d)
    analysis = run_analysis(e1e, e2e, e3d)

    # ====== SAVE ALL RESULTS ======
    output = {
        'e2e_results': e2e,
        'e2f_results': e2f,
        'e1e_results': e1e,
        'e3d_results': e3d,
        'ceiling_fits': ceiling,
        'analysis': analysis,
        'metadata': {
            'engine': 'ceiling_verify.py compatible',
            'params': {
                'DT': DT, 'MSPD': MSPD, 'SMOOTH': SMOOTH,
                'WIN': WIN, 'DELAY_F': DELAY_F, 'DUR': DUR,
                'LOOK': LOOK, 'VOTE_INT': VOTE_INT,
            },
            'total_time_sec': round(time.time() - t_start),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
    }

    outfile = 'paper_fullscale_results.json'
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # ====== FINAL SUMMARY ======
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    print(f"  Total time: {time.time()-t_start:.0f}s")
    print(f"  Results saved to: {outfile}")

    if 'Multiplicative' in analysis.get('model_results', {}):
        m = analysis['model_results']['Multiplicative']
        c = m['coeffs']
        vstar = -c[1] / (2 * c[2]) if c[2] > 0 else None
        print(f"\n  Unified Model (Multiplicative):")
        print(f"    RMSE = ({c[0]:.4f} + ({c[1]:.4f})v + ({c[2]:.5f})v^2)")
        print(f"         x (1 + ({c[3]:.4f})p/sqrt(N))")
        print(f"    R2 = {m['r2']:.4f}")
        if vstar:
            print(f"    v* = {vstar:.2f} m/s")

    cv = analysis.get('cv_results', {}).get('Multiplicative', {})
    if cv:
        print(f"    5-fold CV: R2={cv['r2_mean']:.4f} +/- {cv['r2_std']:.4f}")

    e3d_preds = analysis.get('e3d_predictions', {})
    if e3d_preds:
        errs = [v['error_pct'] for v in e3d_preds.values()]
        print(f"    E3d prediction error: {np.mean(errs):.1f}%")
        print(f"    E3d within 15%: {sum(1 for e in errs if e < 15)}/{len(errs)}")

    print(f"\n  Done.\n")


if __name__ == '__main__':
    main()
