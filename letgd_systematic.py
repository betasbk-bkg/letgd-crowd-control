"""
================================================================================
LETG-d : 체계적 트롤 실험 + 이론(theta) 유도 데이터  — 단일 파일
================================================================================
필요한 것: 같은 폴더에 simulation_main.py
실행:  python letgd_systematic.py        (윈도우)
       python3 letgd_systematic.py       (맥/리눅스)

출력 (JSON 4개):
  sys_detection.json   : 조건별 TPR/FPR/latency  (트롤 규명 — LETG-d용)
  sys_signal.json      : 조건별 Delta_gamma, sigma_baseline, sigma_eff, mu0
  sys_cusum.json       : 공격 후 CUSUM 궤적 (정지시간 유도 — theta용)
  sys_summary.txt      : 사람이 읽는 요약

이 한 번으로: (1) 트롤 유형/집중도/혼합 규명  (2) theta 유도 재료 둘 다 확보.
================================================================================
"""
import numpy as np
import json
import simulation_main as S

# ====================== KNOBS ======================
N        = 120
REPS     = 5            # 반복 (평균±표준편차)
N_PER    = 12           # 궤적당 trace 수
H_ADAPT  = 14           # CUSUM 임계
SLACK    = 0.5          # CUSUM slack
LAM      = 0.15         # EWMA 적응률
WARMUP   = 15
# 조건 축
P_LIST   = [0.08, 0.10, 0.12, 0.15, 0.20, 0.30]   # 공격 비율
C_LIST   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]         # 집중도
# 혼합 모드 (행동 조합)
MIX_MODES = ['opposite', 'bandwagon', 'scatter', 'mixed_ob', 'mixed_bs']
# ===================================================

TRAJS = [S.Circle(R=10), S.Square(), S.Lemniscate(), S.Zigzag()]
DIRS  = S.DIRS

# ---------- 트롤 투표 생성 (집중도 c, 혼합 모드) ----------
def troll_votes(mode, ideal_angle, n_troll, c, rng, state):
    """mode별 트롤 투표. c=집중도(정렬 비율). 나머지는 무작위."""
    if n_troll <= 0:
        return np.array([], dtype=int)
    n_al = int(round(c * n_troll)); n_rn = n_troll - n_al
    opp = (ideal_angle + 180) % 360
    base = S.angle_to_dir(np.array([opp]))[0]
    rand = rng.integers(0, 8, n_rn)
    if mode == 'opposite':
        aligned = (base + rng.integers(-1, 2, n_al)) % 8
    elif mode == 'bandwagon':
        if 'bw' not in state: state['bw'] = int(rng.integers(0, 8))
        aligned = np.full(n_al, state['bw'], dtype=int)
    elif mode == 'scatter':
        if 'sc' not in state: state['sc'] = rng.choice(8, 3, replace=False)
        aligned = rng.choice(state['sc'], n_al)
    elif mode == 'mixed_ob':        # opposite + bandwagon
        h = n_al // 2
        if 'bw' not in state: state['bw'] = int(rng.integers(0, 8))
        aligned = np.concatenate([(base + rng.integers(-1,2,h)) % 8,
                                  np.full(n_al-h, state['bw'], dtype=int)])
    elif mode == 'mixed_bs':        # bandwagon + scatter
        h = n_al // 2
        if 'bw' not in state: state['bw'] = int(rng.integers(0, 8))
        if 'sc' not in state: state['sc'] = rng.choice(8, 3, replace=False)
        aligned = np.concatenate([np.full(h, state['bw'], dtype=int),
                                  rng.choice(state['sc'], n_al-h)])
    else:
        aligned = (base + rng.integers(-1, 2, n_al)) % 8
    return np.concatenate([aligned, rand])

def gen_votes(ideal, prev, p, mode, c, rng, state):
    n_t = min(round(N*p), N); rem = N - n_t
    na = round(rem*0.7368); ns = round(rem*0.2105); no = rem - na - ns
    if no < 0: na += no; no = 0
    ang = np.empty(na+ns+no); idx = 0
    ang[idx:idx+na] = ideal + rng.uniform(-3,3,na); idx += na
    d = ideal - prev
    if d > 180: d -= 360
    if d < -180: d += 360
    if ns > 0:
        lag = rng.uniform(0.2,0.5,ns); ang[idx:idx+ns] = prev + d*(1-lag); idx += ns
    if no > 0: ang[idx:idx+no] = ideal + rng.uniform(-30,30,no); idx += no
    nt = S.angle_to_dir(ang[:idx])
    tv = troll_votes(mode, ideal, n_t, c, rng, state)
    return np.concatenate([nt, tv])

# ---------- gamma(t) trace ----------
def trace(traj, p, mode, c, t_atk_frame, seed):
    rng = np.random.default_rng(seed)
    pos = traj.start(); vel = np.zeros(2); ph = [pos.copy()]
    pa = 0.; cd = np.array([1.,0.]); cg = 0.5
    g = []; t_atk_tick = None; state = {}
    for f in range(S.FRAMES):
        if f % S.VOTE_INT == 0:
            atk = (t_atk_frame is not None and f >= t_atk_frame)
            pp = p if atk else 0.05
            mm = mode if atk else 'opposite'   # baseline: light random-ish
            cc = c if atk else 0.0
            if atk and t_atk_tick is None: t_atk_tick = len(g)
            di = max(0, len(ph)-1-S.DELAY_F); dp = ph[di]
            _, arc = traj.closest(dp); la = traj.at(arc + S.LOOK)
            idd = la - dp; nr = np.linalg.norm(idd)
            if nr > 1e-10: idd /= nr
            ia = np.degrees(np.arctan2(idd[1], idd[0]))
            v = gen_votes(ia, pa, pp, mm, cc, rng, state); pa = ia
            bl = DIRS[v].mean(axis=0); cg = float(np.linalg.norm(bl))
            cd = bl/cg if cg > 1e-10 else np.array([1.,0.])
            g.append(cg)
        tv = cd*S.MSPD; vel += S.SMOOTH*(tv-vel); pos = pos+vel*S.DT; ph.append(pos.copy())
    return np.array(g), t_atk_tick

# ---------- adaptive detector (CUSUM 궤적도 반환) ----------
def detect(gamma, h=H_ADAPT, return_trace=False):
    if len(gamma) < WARMUP+5:
        return (None, []) if return_trace else None
    mu = gamma[:WARMUP].mean(); var = max(gamma[:WARMUP].var(), 1e-4); cs = 0.0
    cs_trace = []
    for i in range(WARMUP, len(gamma)):
        sd = np.sqrt(var)+1e-6; dev = (mu-gamma[i])/sd
        cs = max(0.0, cs + dev - SLACK)
        cs_trace.append(cs)
        if cs > h:
            return (i, cs_trace) if return_trace else i
        if cs < h*0.5:
            mu = (1-LAM)*mu + LAM*gamma[i]; var = (1-LAM)*var + LAM*(gamma[i]-mu)**2
    return (None, cs_trace) if return_trace else None

# ---------- 신호 측정: Delta_gamma, sigma_baseline, sigma_eff, mu0 ----------
def measure_signal(traj, p, mode, c, t_atk_tick_frame, seed):
    g, ta = trace(traj, p, mode, c, t_atk_tick_frame, seed)
    if ta is None or ta < 3 or ta >= len(g)-3:
        return None
    pre = g[WARMUP:ta] if ta > WARMUP else g[:ta]
    post = g[ta:]
    if len(pre) < 3 or len(post) < 3: return None
    mu0 = float(np.mean(pre)); sigma_base = float(np.std(pre))
    dgamma = float(mu0 - np.mean(post))
    # sigma_eff: 탐지 시점 부근 유효 변동 (post 초기 구간의 표준편차 + baseline)
    sigma_eff = float(np.std(g[max(0,ta-5):ta+5]))
    return dict(mu0=mu0, sigma_base=sigma_base, sigma_eff=sigma_eff, dgamma=dgamma)

# ================== 실험 1: 트롤 규명 (TPR/FPR/lat) ==================
print("[1/3] 체계적 트롤 규명: p x c x mode")
detection = []
for mode in MIX_MODES:
    for p in P_LIST:
        for c in C_LIST:
            ts, fs, ls = [], [], []
            for r in range(REPS):
                rng = np.random.default_rng(r*1000+1)
                pos, neg = [], []
                for i in range(N_PER*len(TRAJS)):
                    traj = TRAJS[i % len(TRAJS)]
                    t = int(rng.uniform(0.3,0.7)*S.FRAMES); t=(t//S.VOTE_INT)*S.VOTE_INT
                    g, ta = trace(traj, p, mode, c, t, seed=r*9999+i*7+1)
                    if ta is not None: pos.append((g, ta))
                for i in range(N_PER*len(TRAJS)):
                    traj = TRAJS[i % len(TRAJS)]
                    g, _ = trace(traj, p, mode, c, None, seed=r*9999+i*11+5000)
                    neg.append(g)
                tl, tp, fp = [], 0, 0
                for g, ta in pos:
                    d = detect(g)
                    if d is not None and d >= ta: tp+=1; tl.append(d-ta)
                for g in neg:
                    if detect(g) is not None: fp+=1
                ts.append(tp/len(pos)); fs.append(fp/len(neg))
                if tl: ls.append(np.median(tl))
            detection.append(dict(mode=mode, p=p, c=c,
                tpr=float(np.mean(ts)), tpr_sd=float(np.std(ts)),
                fpr=float(np.mean(fs)), lat=float(np.mean(ls)) if ls else None))
    print(f"   done mode={mode}")
json.dump(detection, open('sys_detection.json','w'), indent=2)

# ================== 실험 2: 신호 측정 (theta 재료) ==================
print("[2/3] 신호 측정: Delta_gamma, sigma_base, sigma_eff, mu0")
signal = []
for mode in ['opposite', 'bandwagon', 'scatter']:
    for p in P_LIST:
        for c in C_LIST:
            vals = []
            rng = np.random.default_rng(7)
            for i in range(N_PER*len(TRAJS)):
                traj = TRAJS[i % len(TRAJS)]
                t = int(rng.uniform(0.3,0.7)*S.FRAMES); t=(t//S.VOTE_INT)*S.VOTE_INT
                m = measure_signal(traj, p, mode, c, t, seed=i*7+1)
                if m: vals.append(m)
            if vals:
                signal.append(dict(mode=mode, p=p, c=c,
                    mu0=float(np.mean([v['mu0'] for v in vals])),
                    sigma_base=float(np.mean([v['sigma_base'] for v in vals])),
                    sigma_eff=float(np.mean([v['sigma_eff'] for v in vals])),
                    dgamma=float(np.mean([v['dgamma'] for v in vals]))))
    print(f"   done mode={mode}")
json.dump(signal, open('sys_signal.json','w'), indent=2)

# ================== 실험 3: CUSUM 궤적 (정지시간 유도) ==================
print("[3/3] CUSUM 궤적: 공격 후 누적 동역학 (theta 핵심)")
cusum = []
for mode in ['opposite', 'scatter']:        # 쉬운 것 vs 어려운 것
    for p in [0.10, 0.15, 0.20]:
        for c in [0.2, 0.6, 1.0]:
            traces = []
            rng = np.random.default_rng(11)
            for i in range(20):
                traj = TRAJS[i % len(TRAJS)]
                t = int(0.5*S.FRAMES); t=(t//S.VOTE_INT)*S.VOTE_INT
                g, ta = trace(traj, p, mode, c, t, seed=i*7+1)
                if ta is None: continue
                _, cst = detect(g, return_trace=True)
                # 공격 시작(ta) 이후 CUSUM 궤적만 (warmup offset 보정)
                start = max(0, ta - WARMUP)
                traces.append(cst[start:start+40])  # 공격 후 40틱
            # 평균 궤적
            maxlen = max((len(t) for t in traces), default=0)
            if maxlen == 0: continue
            avg = []
            for k in range(maxlen):
                vals = [t[k] for t in traces if len(t) > k]
                avg.append(float(np.mean(vals)))
            cusum.append(dict(mode=mode, p=p, c=c, avg_cusum_after_attack=avg))
    print(f"   done mode={mode}")
json.dump(cusum, open('sys_cusum.json','w'), indent=2)

# ================== 요약 ==================
with open('sys_summary.txt', 'w') as f:
    f.write("LETG-d 체계적 실험 요약\n" + "="*50 + "\n\n")
    f.write("[검출 — 집중도 임계 확인] mode=opposite\n")
    for d in detection:
        if d['mode']=='opposite' and d['p'] in (0.10,0.15):
            f.write(f"  p={d['p']} c={d['c']}: TPR={d['tpr']:.2f} lat={d['lat']}\n")
    f.write("\n[신호 — Delta_gamma vs p,c] mode=opposite\n")
    for s in signal:
        if s['mode']=='opposite':
            f.write(f"  p={s['p']} c={s['c']}: Dg={s['dgamma']:.3f} "
                    f"sig_base={s['sigma_base']:.3f} sig_eff={s['sigma_eff']:.3f}\n")
print("\nDONE. saved: sys_detection.json, sys_signal.json, sys_cusum.json, sys_summary.txt")
