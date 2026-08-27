from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from .seihfr import PARAMETER_NAMES, IDX, SEIHFRCalibrationResults
from .utils import PROJECT_ROOT, weighted_quantile

SENSITIVITY_NAMES=["Community transmission multiplier","Known-contact follow-up","Symptom-to-isolation delay","Safe burial coverage","Health-facility IPC","Response disruption severity"]
@dataclass
class SensitivityResults:
    indices: pd.DataFrame
    metadata: pd.DataFrame

def _med_params(cal):
    p=cal.ensemble[PARAMETER_NAMES].to_numpy(float); return np.array([weighted_quantile(p[:,j],[.5],cal.weights)[0] for j in range(p.shape[1])])
def _med_state(cal):
    s=np.array([weighted_quantile(cal.final_states[:,j],[.5],cal.weights)[0] for j in range(cal.final_states.shape[1])]); s[7],s[6],s[3]=5290.,2516.,837.; return s

def run_global_sensitivity(calibration:SEIHFRCalibrationResults,config:dict,save:bool=True)->SensitivityResults:
    """Sobol/PRCC for the same 90-day SEIHFR policy layer as the main scenarios."""
    cfg=config["seihfr"]; sc=config["sensitivity"]; med=_med_params(calibration); base=_med_state(calibration); ne=float(cfg["effective_population"]); horizon=int(sc["horizon_days"]); dt=float(cfg["integration_step_days"]); low=np.array([.70,.55,1.5,.50,.45,0.]); high=np.array([1.45,.98,7.,.99,.99,1.])
    sigma=1/float(cfg["latent_period_days"]); ce=1/float(cfg["community_outcome_days"]); pcd=float(cfg["community_fatality_probability"]); mui=pcd*ce; gammai=(1-pcd)*ce; he=1/float(med[IDX["hospital_outcome_days"]]); ph=float(med[IDX["hospital_fatality_probability"]]); muh=ph*he; gammah=(1-ph)*he; kappa=1/float(cfg["unsafe_funeral_duration_days"]); beta0=float(weighted_quantile(calibration.final_beta,[.5],calibration.weights)[0]); eh0=float(med[IDX["hospital_relative_infectiousness"]]); ef0=float(med[IDX["funeral_relative_infectiousness"]])
    def sim(p):
        n=len(p); y=np.tile(base,(n,1)).astype(float); y[:,7]=0.; y[:,6]=0.; peak=np.full(n,base[3]); steps=int(round(horizon/dt))
        for _ in range(steps):
            cm,fu,iso,safe,ipc,dis=[p[:,j] for j in range(6)]; tr=np.clip((fu-.55)/.40,0,1); delta=1/np.maximum(iso,1.); eh=eh0*(1-ipc)/.25; ef=ef0*(1-safe)/.18; beta=beta0*cm*(1+.60*dis)*(1-.25*tr); S,E,I,H,F,R,D,C=[y[:,j] for j in range(8)]; force=beta*(I+eh*H+ef*F)/ne; nx=np.minimum(force*S,S/max(dt,1e-9)); onset=sigma*E; dy=np.column_stack([-nx,nx-onset,onset-(delta+gammai+mui)*I,delta*I-(gammah+muh)*H,mui*I+muh*H-kappa*F,gammai*I+gammah*H,mui*I+muh*H,onset]); y=np.maximum(y+dt*dy,0); peak=np.maximum(peak,y[:,3])
        return np.column_stack([y[:,7],y[:,6],peak])
    rng=np.random.default_rng(int(config["random_seed"])+404); nb=int(sc["sobol_base_size"]); d=6; a=low+(high-low)*rng.random((nb,d)); b=low+(high-low)*rng.random((nb,d)); fa=sim(a); fb=sim(b); vars=[np.var(np.r_[fa[:,k],fb[:,k]],ddof=1) for k in range(3)]; first=[[] for _ in range(3)]; total=[[] for _ in range(3)]
    for i in range(d):
        ab=a.copy(); ab[:,i]=b[:,i]; fab=sim(ab)
        for k in range(3):
            v=max(vars[k],1e-12); first[k].append(max(0.,float(1-np.mean((fb[:,k]-fab[:,k])**2)/(2*v)))); total[k].append(max(0.,float(np.mean((fa[:,k]-fab[:,k])**2)/(2*v))))
    npcc=int(sc["prcc_size"]); pp=low+(high-low)*rng.random((npcc,d)); oo=sim(pp); xr=np.column_stack([rankdata(pp[:,i]) for i in range(d)])
    def prcc(y):
        yr=rankdata(y); out=[]
        for i in range(d):
            z=np.delete(xr,i,axis=1); z=np.column_stack([np.ones(len(z)),z]); rx=xr[:,i]-z@np.linalg.lstsq(z,xr[:,i],rcond=None)[0]; ry=yr-z@np.linalg.lstsq(z,yr,rcond=None)[0]; out.append(float(np.corrcoef(rx,ry)[0,1]))
        return out
    ind=pd.DataFrame({"parameter":SENSITIVITY_NAMES,"lower_bound":low,"upper_bound":high,"first_order_sobol_cases90":first[0],"total_order_sobol_cases90":total[0],"first_order_sobol_deaths90":first[1],"total_order_sobol_deaths90":total[1],"first_order_sobol_peak_hospital90":first[2],"total_order_sobol_peak_hospital90":total[2],"PRCC_cases90":prcc(oo[:,0]),"PRCC_deaths90":prcc(oo[:,1]),"PRCC_peak_hospital90":prcc(oo[:,2])})
    meta=pd.DataFrame([("model","SEIHFR policy layer conditioned on modular Bayesian cut posterior"),("primary_outcome","90-day cumulative cases"),("secondary_outcomes","90-day deaths and peak hospital/isolation stock"),("varied_levers","community transmission, contact follow-up, isolation delay, safe burial, IPC, operational disruption"),("held_fixed","clinical fatality and reporting/ascertainment in the main policy sensitivity"),("sobol_design",f"Jansen/Saltelli A-B design, base size {nb}"),("prcc_design",f"rank-based partial correlation design, size {npcc}")],columns=["item","definition"])
    if save:
        ind.to_csv(PROJECT_ROOT/"results"/"seihfr_global_sensitivity.csv",index=False); meta.to_csv(PROJECT_ROOT/"results"/"seihfr_global_sensitivity_metadata.csv",index=False)
    return SensitivityResults(indices=ind,metadata=meta)
