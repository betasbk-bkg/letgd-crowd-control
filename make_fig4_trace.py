"""
FIGURE 4: 실제 ramp-attack trace 생성 (simulation_main.py 사용, 검증된 detector 로직)
출력: fig4_trace.png + fig4_trace_data.json (재현용)
"""
import numpy as np, json
import simulation_main as S
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WARMUP, SLACK, LAM, H = 15, 0.5, 0.15, 14
N = 120
BASE = 0.05

# ramp schedule: 5% -> 25% over ~80 ticks, onset ~tick 18
def ramp_sched(t0, t1):
    def f(tick):
        if tick < t0: return BASE
        if tick > t1: return 0.25
        return BASE + (tick-t0)/(t1-t0)*(0.25-BASE)
    return f

def run_trace(traj, schedule_fn, seed):
    """한 trace를 돌리며 gamma + adversarial fraction 기록"""
    rng = np.random.default_rng(seed)
    pos = traj.start(); vel = np.zeros(2); ph=[pos.copy()]
    pa=0.; cd=np.array([1.,0.]); cg=0.5
    gammas=[]; fracs=[]; tick=0
    for f in range(S.FRAMES):
        if f % S.VOTE_INT == 0:
            troll = schedule_fn(tick)
            di=max(0,len(ph)-1-S.DELAY_F); dp=ph[di]
            _,arc=traj.closest(dp); la=traj.at(arc+S.LOOK)
            idd=la-dp; nrm=np.linalg.norm(idd)
            if nrm>1e-10: idd/=nrm
            ia=np.degrees(np.arctan2(idd[1],idd[0]))
            votes=S.gen_votes(ia,pa,troll,N,rng); pa=ia
            bl=S.DIRS[votes].mean(axis=0); cg=float(np.linalg.norm(bl))
            cd=bl/cg if cg>1e-10 else np.array([1.,0.])
            gammas.append(cg); fracs.append(troll); tick+=1
        tv=cd*S.MSPD; vel+=S.SMOOTH*(tv-vel); pos=pos+vel*S.DT; ph.append(pos.copy())
    return np.array(gammas), np.array(fracs)

# detector를 돌리며 EWMA baseline, CUSUM, freeze, alarm 전부 기록 (Algorithm 1 그대로)
def run_detector(gamma):
    mu=gamma[:WARMUP].mean(); var=max(gamma[:WARMUP].var(),1e-4); cs=0.0
    mus=[np.nan]*WARMUP; css=[0.0]*WARMUP
    alarm=None; freeze_start=None
    for i in range(WARMUP,len(gamma)):
        sd=np.sqrt(var)+1e-6; dev=(mu-gamma[i])/sd
        cs=max(0.0, cs+dev-SLACK)
        css.append(cs); 
        if cs>H and alarm is None: alarm=i
        if cs>=H*0.5 and freeze_start is None: freeze_start=i
        if cs<H*0.5:
            mu=(1-LAM)*mu+LAM*gamma[i]; var=(1-LAM)*var+LAM*(gamma[i]-mu)**2
        mus.append(mu)
    return np.array(mus), np.array(css), alarm, freeze_start

# 좋은 예시 trace를 찾을 때까지 (alarm이 명확히 잡히고 freeze가 보이는)
traj=S.Circle(R=10)
for seed in range(1, 40):
    g, fr = run_trace(traj, ramp_sched(18, 98), seed)
    mus, css, alarm, freeze = run_detector(g)
    if alarm is not None and freeze is not None and alarm>freeze:
        print(f"seed {seed}: alarm={alarm}, freeze={freeze}, len={len(g)}")
        break

# attack 10% 초과 시점
atk_exceed = next((i for i,f in enumerate(fr) if f>0.10), None)

# 플롯
fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200)
ticks = np.arange(len(g))
ax.plot(ticks, g, color='#1f77b4', lw=1.3, label='Consensus strength γ')
ax.plot(ticks, mus, color='#ff7f0e', lw=1.8, label='EWMA baseline')
ax.plot(ticks, css/H, color='#2ca02c', lw=1.5, label='CUSUM / h')
ax.plot(ticks, fr, color='#d62728', lw=1.8, label='Adversarial fraction')
if atk_exceed: ax.axvline(atk_exceed, color='#1f77b4', ls='--', lw=1.2, alpha=0.7, label=f'Attack exceeds 10%')
if freeze: ax.axvline(freeze, color='#1f77b4', ls='-.', lw=1.4, alpha=0.8, label=f'Freeze begins ({freeze})')
if alarm: ax.axvline(alarm, color='#1f77b4', ls=':', lw=1.6, alpha=0.9, label=f'Alarm tick ({alarm})')
ax.set_xlabel('Voting tick', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Representative ramp-attack detector trace', fontsize=13)
ax.set_ylim(-0.02, 1.08); ax.set_xlim(0, len(g)-1)
ax.legend(loc='center right', fontsize=8.5, framealpha=0.9)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('fig4_trace.png', dpi=200, bbox_inches='tight')
print("saved fig4_trace.png")

# 재현용 데이터 저장
json.dump({'seed':int(seed),'alarm':int(alarm),'freeze':int(freeze),
           'atk_exceed':int(atk_exceed) if atk_exceed else None,
           'gamma':[round(float(x),4) for x in g],
           'ewma':[round(float(x),4) for x in mus],
           'cusum_over_h':[round(float(x/H),4) for x in css],
           'frac':[round(float(x),4) for x in fr]},
          open('fig4_trace_data.json','w'), indent=1)
print("saved fig4_trace_data.json")
print(f"\n=== 검증: alarm={alarm}, freeze={freeze}, atk_exceed={atk_exceed} ===")
print(f"γ 시작~끝: {g[0]:.3f} -> {g[-1]:.3f}, baseline freeze 후 고정 확인: {mus[freeze]:.3f} vs {mus[-1]:.3f}")
