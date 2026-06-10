"""
LETG-d ABLATION (standalone) — run and send back letgd_ablation.json
Needs ONLY: simulation_main.py in the same folder.
Run:  python letgd_ablation.py    ->   letgd_ablation.json

Four variants x THREE conditions:
  STEP       (abrupt 20% attack)        : all CUSUM variants do well (freeze not critical)
  RAMP_FAST  (5%->25% over ~30 ticks)   : CUSUM catches before baseline contaminates
  RAMP_SLOW  (5%->25% over ~80 ticks)   : *** freeze is decisive *** — no-freeze baseline
                                          adapts to the creeping attack and misses it.
Variants: fixed / ewma_only / cusum_nofreeze / proposed(EWMA+CUSUM+freeze).

Expected: on RAMP_SLOW, cusum_nofreeze TPR collapses (~0.3) while proposed stays ~1.0.
This is the honest demonstration that the suspicion-freeze defends against slow,
stealthy attacks that a naively-adaptive baseline would absorb.
"""
import numpy as np, json
import simulation_main as S

WARMUP, SLACK, LAM = 15, 0.5, 0.15
N = 120
BASE_TROLL = 0.05
REPS = 5
N_PER_TRAJ = 25
TRAJS = [S.Circle(R=10), S.Square(), S.Lemniscate(), S.Zigzag()]

def _run(traj, N, schedule_fn, seed):
    rng=np.random.default_rng(seed)
    pos=traj.start(); vel=np.zeros(2); ph=[pos.copy()]
    pa=0.; cd=np.array([1.,0.]); cg=0.5; gammas=[]; first=None; tick=0
    for f in range(S.FRAMES):
        if f % S.VOTE_INT == 0:
            troll=schedule_fn(tick)
            if first is None and troll>0.10: first=tick
            di=max(0,len(ph)-1-S.DELAY_F); dp=ph[di]
            _,arc=traj.closest(dp); la=traj.at(arc+S.LOOK)
            idd=la-dp; nrm=np.linalg.norm(idd)
            if nrm>1e-10: idd/=nrm
            ia=np.degrees(np.arctan2(idd[1],idd[0]))
            votes=S.gen_votes(ia,pa,troll,N,rng); pa=ia
            bl=S.DIRS[votes].mean(axis=0); cg=float(np.linalg.norm(bl))
            cd=bl/cg if cg>1e-10 else np.array([1.,0.])
            gammas.append(cg); tick+=1
        tv=cd*S.MSPD; vel+=S.SMOOTH*(tv-vel); pos=pos+vel*S.DT; ph.append(pos.copy())
    return np.array(gammas), first

def step_sched(t_atk):    return lambda tick: 0.20 if tick>=t_atk else BASE_TROLL
def ramp_fast(t0,t1):
    def f(tick):
        if tick<t0: return 0.05
        if tick>t1: return 0.25
        return 0.05+(tick-t0)/(t1-t0)*0.20
    return f
def ramp_slow(t0,t1):  # over ~80 ticks
    def f(tick):
        if tick<t0: return 0.05
        if tick>t1: return 0.25
        return 0.05+(tick-t0)/(t1-t0)*0.20
    return f
def clean_sched():        return lambda tick: BASE_TROLL

def build(cond, seed_off, attack):
    out=[]; rng=np.random.default_rng(seed_off)
    for i in range(N_PER_TRAJ*len(TRAJS)):
        traj=TRAJS[i%len(TRAJS)]
        if not attack:
            g,_=_run(traj,N,clean_sched(),seed_off+i*11+5000); out.append(g); continue
        if cond=='step':
            t=int(rng.uniform(0.30,0.70)*(S.FRAMES//S.VOTE_INT))
            g,ta=_run(traj,N,step_sched(t),seed_off+i*7+1)
            out.append((g, ta if ta is not None else t))
        elif cond=='ramp_fast':
            g,ta=_run(traj,N,ramp_fast(int(rng.integers(15,25)),int(rng.integers(45,55))),seed_off+i*7+1)
            if ta is not None: out.append((g,ta))
        else: # ramp_slow
            t0=int(rng.integers(15,20))
            g,ta=_run(traj,N,ramp_slow(t0, t0+80),seed_off+i*7+1)
            if ta is not None: out.append((g,ta))
    return out

def det_fixed(gamma,k,warmup=WARMUP,win=10):
    if len(gamma)<warmup+5: return None
    for i in range(warmup,len(gamma)):
        w=gamma[max(0,i-win):i]
        if len(w)<3: continue
        mu,sd=w.mean(),w.std()+1e-6
        if (mu-gamma[i])/sd>k: return i
    return None
def det_ewma(gamma,z,warmup=WARMUP,lam=LAM):
    if len(gamma)<warmup+5: return None
    mu=gamma[:warmup].mean(); var=max(gamma[:warmup].var(),1e-4)
    for i in range(warmup,len(gamma)):
        sd=np.sqrt(var)+1e-6; dev=(mu-gamma[i])/sd
        if dev>z: return i
        mu=(1-lam)*mu+lam*gamma[i]; var=(1-lam)*var+lam*(gamma[i]-mu)**2
    return None
def det_nofreeze(gamma,h,warmup=WARMUP,lam=LAM):
    if len(gamma)<warmup+5: return None
    mu=gamma[:warmup].mean(); var=max(gamma[:warmup].var(),1e-4); cs=0.0
    for i in range(warmup,len(gamma)):
        sd=np.sqrt(var)+1e-6; dev=(mu-gamma[i])/sd
        cs=max(0.0,cs+dev-SLACK)
        if cs>h: return i
        mu=(1-lam)*mu+lam*gamma[i]; var=(1-lam)*var+lam*(gamma[i]-mu)**2
    return None
def det_proposed(gamma,h,warmup=WARMUP,lam=LAM):
    if len(gamma)<warmup+5: return None
    mu=gamma[:warmup].mean(); var=max(gamma[:warmup].var(),1e-4); cs=0.0
    for i in range(warmup,len(gamma)):
        sd=np.sqrt(var)+1e-6; dev=(mu-gamma[i])/sd
        cs=max(0.0,cs+dev-SLACK)
        if cs>h: return i
        if cs<h*0.5:
            mu=(1-lam)*mu+lam*gamma[i]; var=(1-lam)*var+lam*(gamma[i]-mu)**2
    return None

def score(det,par,pos,neg):
    tl,tp,fp=[],0,0
    for g,ta in pos:
        d=det(g,par)
        if d is not None and d>=ta: tp+=1; tl.append(d-ta)
    for g in neg:
        if det(g,par) is not None: fp+=1
    return (tp/len(pos) if pos else 0., fp/len(neg) if neg else 0., float(np.median(tl)) if tl else float('nan'))
def rep(det,par,cond):
    ts,fs,ls=[],[],[]
    for r in range(REPS):
        pos=build(cond,r*1000+1,True); neg=build(cond,r*1000+1,False)
        t,f,l=score(det,par,pos,neg); ts.append(t);fs.append(f)
        if not np.isnan(l): ls.append(l)
    return dict(tpr=float(np.mean(ts)),fpr=float(np.mean(fs)),lat=float(np.mean(ls)) if ls else float('nan'))
def best(det,params,cond):
    ms=[dict(rep(det,p,cond),param=p) for p in params]
    ok=[m for m in ms if m['fpr']<=0.10]
    return max(ok,key=lambda x:x['tpr']) if ok else min(ms,key=lambda x:x['fpr'])

if __name__=='__main__':
    variants=[('fixed',det_fixed,[1.5,2,2.5,3,3.5,4,4.5,5,6]),
              ('ewma_only',det_ewma,[2,2.5,3,3.5,4,4.5,5]),
              ('cusum_nofreeze',det_nofreeze,[5,7,10,14,20,30]),
              ('proposed',det_proposed,[5,7,10,14,20,30])]
    result={'step':{},'ramp_fast':{},'ramp_slow':{}}
    for cond in ['step','ramp_fast','ramp_slow']:
        print(f"\n=== {cond.upper()} ===")
        for name,det,params in variants:
            m=best(det,params,cond); result[cond][name]=m
            print(f"  {name:16s}: TPR={m['tpr']:.2f} FPR={m['fpr']:.2f} lat={m['lat']:.1f} (param={m['param']})")
    json.dump(result,open('letgd_ablation.json','w'),indent=2)
    print("\nsaved letgd_ablation.json — send this back.")
