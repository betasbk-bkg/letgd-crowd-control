"""
================================================================================
LETG-d : 단일 파일 데이터 생산 스크립트  (BK 직접 실행용)
================================================================================
필요한 것: 같은 폴더에 simulation_main.py (네 시뮬레이션 엔진)
  실행:  python letgd_all_in_one.py      (윈도우)
         python3 letgd_all_in_one.py     (맥/리눅스)
출력:  letgd_roc.json, letgd_robustness.json, letgd_schedules.json

이 파일 하나 + simulation_main.py 면 끝. 복사도 다른 파일도 필요 없음.
숫자(MC/반복/표본)는 아래 KNOBS에 이미 권장값으로 박아둠.
================================================================================
"""
import numpy as np
import json
import simulation_main as S

# ====================== KNOBS (이미 권장값으로 설정됨) ======================
N_PER_TRAJ = 25      # 궤적당 trace 수 (총 = 25 x 4궤적 = 100 traces/pool)
REPS       = 5       # 독립 pool 반복 횟수 -> 평균 +/- 표준편차
BASE_TROLL = 0.05    # 평상시(clean) troll 비율
H_ADAPT    = 14      # robustness/schedule 그리드용 adaptive 운영점
FIXED_KS   = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]   # ROC: fixed sweep
ADAPT_HS   = [3, 5, 7, 10, 14, 20, 30, 50]                    # ROC: adaptive sweep
ATK_RATIOS = [0.30, 0.20, 0.15, 0.10]   # robustness: 공격 비율
N_VALUES   = [50, 120, 200]             # robustness: 군중 크기
ROC_ATK, ROC_N = 0.20, 120              # ROC 기준 조건
# 더 단단히 하려면: N_PER_TRAJ=40, REPS=10 (시간만 더 걸림)
# =============================================================================

TRAJS = [S.Circle(R=10), S.Square(), S.Lemniscate(), S.Zigzag()]

# ---------------------------------------------------------------------------
# (gate3) 공격 주입 trace 생성: step troll at t_attack_frame
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# (gate4) 시간가변 스케줄 trace 생성
# ---------------------------------------------------------------------------
def trace_with_schedule(traj, schedule_fn, N=120, seed=0):
    rng = np.random.default_rng(seed)
    pos = traj.start(); vel = np.zeros(2); ph = [pos.copy()]
    pa = 0.; cd = np.array([1., 0.]); cg = 0.5
    gammas = []; first_attack_tick = None; tick = 0
    for f in range(S.FRAMES):
        if f % S.VOTE_INT == 0:
            troll = schedule_fn(tick)
            if first_attack_tick is None and troll > 0.10:
                first_attack_tick = tick
            di = max(0, len(ph)-1-S.DELAY_F); dp = ph[di]
            _, arc = traj.closest(dp); la = traj.at(arc + S.LOOK)
            idd = la - dp; nrm = np.linalg.norm(idd)
            if nrm > 1e-10: idd /= nrm
            ia = np.degrees(np.arctan2(idd[1], idd[0]))
            votes = S.gen_votes(ia, pa, troll, N, rng); pa = ia
            bl = S.DIRS[votes].mean(axis=0); cg = float(np.linalg.norm(bl))
            cd = bl/cg if cg > 1e-10 else np.array([1., 0.])
            gammas.append(cg); tick += 1
        tv = cd * S.MSPD; vel += S.SMOOTH*(tv - vel); pos = pos + vel*S.DT; ph.append(pos.copy())
    return np.array(gammas), first_attack_tick

# ---------------------------------------------------------------------------
# 스케줄 팩토리
# ---------------------------------------------------------------------------
def make_ramp(t_start=20, t_end=50, p_lo=0.05, p_hi=0.25):
    def f(tick):
        if tick < t_start: return p_lo
        if tick > t_end: return p_hi
        return p_lo + (tick - t_start)/(t_end - t_start) * (p_hi - p_lo)
    return f
def make_burst(t_start, duration=10, p_atk=0.30, p_lo=0.05):
    def f(tick): return p_atk if t_start <= tick < t_start+duration else p_lo
    return f
def make_periodic(period=15, p_lo=0.05, p_hi=0.25, phase=0):
    def f(tick):
        if tick < 10: return p_lo
        return p_hi if ((tick-10+phase)//period) % 2 == 1 else p_lo
    return f
def make_clean(p_lo=0.05):
    def f(tick): return p_lo
    return f

# ---------------------------------------------------------------------------
# 두 검출기:  FIXED z-score  vs  ADAPTIVE (EWMA+CUSUM+freeze)  <- LETG-d 핵심
# ---------------------------------------------------------------------------
def detect_fixed_zscore(gamma, k_z, warmup=15, win=10):
    if len(gamma) < warmup + 5: return None
    for i in range(warmup, len(gamma)):
        w = gamma[max(0,i-win):i]
        if len(w) < 3: continue
        mu, sd = w.mean(), w.std() + 1e-6
        if (mu - gamma[i])/sd > k_z: return i
    return None

def detect_adaptive(gamma, h, warmup=15, lam=0.15):
    """EWMA baseline + CUSUM accumulation + suspicion-freeze. (LETG-d method)"""
    if len(gamma) < warmup + 5: return None
    mu = gamma[:warmup].mean()
    var = max(gamma[:warmup].var(), 1e-4)
    cusum = 0.0
    for i in range(warmup, len(gamma)):
        sd = np.sqrt(var) + 1e-6
        dev = (mu - gamma[i]) / sd
        cusum = max(0.0, cusum + dev - 0.5)
        if cusum > h: return i
        if cusum < h * 0.5:                 # freeze baseline when suspicious
            mu = (1-lam)*mu + lam*gamma[i]
            var = (1-lam)*var + lam*(gamma[i]-mu)**2
    return None

# ---------------------------------------------------------------------------
# pool 생성 / 스코어링 / 반복
# ---------------------------------------------------------------------------
def build_pool(attack_troll, N, seed_off, attack=True):
    out = []; rng = np.random.default_rng(seed_off)
    for i in range(N_PER_TRAJ * len(TRAJS)):
        traj = TRAJS[i % len(TRAJS)]
        if attack:
            t = int(rng.uniform(0.30, 0.70)*S.FRAMES); t = (t//S.VOTE_INT)*S.VOTE_INT
            g, ta = trace_with_attack(traj, N=N, base_troll=BASE_TROLL,
                                       attack_troll=attack_troll, t_attack_frame=t,
                                       seed=seed_off + i*7 + 1)
            if ta is not None: out.append((g, ta))
        else:
            g, _ = trace_with_attack(traj, N=N, base_troll=BASE_TROLL,
                                      attack_troll=attack_troll, t_attack_frame=None,
                                      seed=seed_off + i*11 + 5000)
            out.append(g)
    return out

def score(detector, param, pos, neg):
    tl, tp, fp = [], 0, 0
    for g, ta in pos:
        d = detector(g, param)
        if d is not None and d >= ta: tp += 1; tl.append(d - ta)
    for g in neg:
        if detector(g, param) is not None: fp += 1
    return (tp/len(pos) if pos else 0.0,
            fp/len(neg) if neg else 0.0,
            float(np.median(tl)) if tl else float('nan'))

def repeated(detector, param, attack_troll, N):
    ts, fs, ls = [], [], []
    for r in range(REPS):
        pos = build_pool(attack_troll, N, seed_off=r*1000+1, attack=True)
        neg = build_pool(attack_troll, N, seed_off=r*1000+1, attack=False)
        t, f, l = score(detector, param, pos, neg)
        ts.append(t); fs.append(f)
        if not np.isnan(l): ls.append(l)
    return dict(tpr=float(np.mean(ts)), tpr_sd=float(np.std(ts)),
                fpr=float(np.mean(fs)), fpr_sd=float(np.std(fs)),
                lat=float(np.mean(ls)) if ls else float('nan'),
                lat_sd=float(np.std(ls)) if ls else float('nan'))

# ===========================================================================
# 1. ROC
# ===========================================================================
print(f"[1/3] ROC fixed vs adaptive (atk={ROC_ATK}, N={ROC_N}, reps={REPS}, pool={N_PER_TRAJ*4})")
roc = {'fixed': [], 'adaptive': []}
for k in FIXED_KS:
    m = repeated(detect_fixed_zscore, k, ROC_ATK, ROC_N); m['param'] = k; roc['fixed'].append(m)
    print(f"  fixed  k={k:<4} TPR={m['tpr']:.2f}±{m['tpr_sd']:.2f} FPR={m['fpr']:.2f}±{m['fpr_sd']:.2f} lat={m['lat']:.1f}")
for h in ADAPT_HS:
    m = repeated(detect_adaptive, h, ROC_ATK, ROC_N); m['param'] = h; roc['adaptive'].append(m)
    print(f"  adapt  h={h:<4} TPR={m['tpr']:.2f}±{m['tpr_sd']:.2f} FPR={m['fpr']:.2f}±{m['fpr_sd']:.2f} lat={m['lat']:.1f}")
json.dump(roc, open('letgd_roc.json', 'w'), indent=2)

# ===========================================================================
# 2. Robustness grid
# ===========================================================================
print(f"\n[2/3] Robustness grid (adaptive h={H_ADAPT}, reps={REPS})")
grid = []
for atk in ATK_RATIOS:
    for N in N_VALUES:
        m = repeated(detect_adaptive, H_ADAPT, atk, N); m['atk'] = atk; m['N'] = N; grid.append(m)
        print(f"  atk={atk:.2f} N={N:<4} TPR={m['tpr']:.2f}±{m['tpr_sd']:.2f} FPR={m['fpr']:.2f}±{m['fpr_sd']:.2f} lat={m['lat']:.1f}±{m['lat_sd']:.1f}")
json.dump({'robustness': grid}, open('letgd_robustness.json', 'w'), indent=2)

# ===========================================================================
# 3. Attack schedules
# ===========================================================================
print(f"\n[3/3] Attack schedules (adaptive h={H_ADAPT}, reps={REPS})")
def sched_pool(factory, seed_off, attack=True):
    out = []; rng = np.random.default_rng(seed_off)
    for i in range(N_PER_TRAJ * len(TRAJS)):
        traj = TRAJS[i % len(TRAJS)]
        sched = factory(rng, i) if attack else make_clean()
        g, ta = trace_with_schedule(traj, sched, seed=seed_off + i*7 + 1)
        if attack:
            if ta is not None: out.append((g, ta))
        else:
            out.append(g)
    return out

schedules = {
    'step':       lambda rng, i: (lambda tick: 0.30 if tick >= int(rng.integers(20,40)) else 0.05),
    'ramp':       lambda rng, i: make_ramp(t_start=int(rng.integers(15,25)), t_end=int(rng.integers(45,55))),
    'burst':      lambda rng, i: make_burst(t_start=int(rng.integers(20,40)), duration=10, p_atk=0.30),
    'burst-weak': lambda rng, i: make_burst(t_start=int(rng.integers(20,40)), duration=10, p_atk=0.20),
    'periodic':   lambda rng, i: make_periodic(period=15, phase=int(rng.integers(0,15))),
}
sched_results = {}
for name, factory in schedules.items():
    ts, fs, ls = [], [], []
    for r in range(REPS):
        pos = sched_pool(factory, seed_off=r*1000+1, attack=True)
        neg = sched_pool(factory, seed_off=r*1000+1, attack=False)
        t, f, l = score(detect_adaptive, H_ADAPT, pos, neg)
        ts.append(t); fs.append(f)
        if not np.isnan(l): ls.append(l)
    m = dict(tpr=float(np.mean(ts)), tpr_sd=float(np.std(ts)),
             fpr=float(np.mean(fs)), fpr_sd=float(np.std(fs)),
             lat=float(np.mean(ls)) if ls else float('nan'))
    sched_results[name] = m
    print(f"  {name:<11} TPR={m['tpr']:.2f}±{m['tpr_sd']:.2f} FPR={m['fpr']:.2f}±{m['fpr_sd']:.2f} lat={m['lat']:.1f}")
json.dump(sched_results, open('letgd_schedules.json', 'w'), indent=2)

print("\nDONE. saved: letgd_roc.json, letgd_robustness.json, letgd_schedules.json")
