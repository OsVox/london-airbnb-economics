"""Verified ordinary least-squares and cluster covariance routines, with explicit data arguments."""
import numpy as np
import pandas as pd
from scipy import linalg,stats

def estimate(W,T,index,d,y,weight=None,cluster='host_id',direct_check=False):
    w=W[index]; t=T[index]; yy=y[index]; n=len(index); q=t.shape[1]
    sw=np.sqrt(np.ones(n) if weight is None else weight)
    ww=w*sw[:,None]; targets=np.column_stack([yy,t])*sw[:,None]
    a,_,r,_=linalg.lstsq(ww,targets,cond=1e-12,lapack_driver='gelsd'); residual=targets-ww@a
    ry=residual[:,0]; rt=residual[:,1:]
    beta,_,rq,_=linalg.lstsq(rt,ry,cond=1e-12,lapack_driver='gelsd'); assert rq==q
    error=ry-rt@beta; k=r+rq
    keys=d.iloc[index][cluster].fillna('missing').to_numpy(); codes,labels=pd.factorize(keys); g=len(labels)
    scores=np.zeros((g,q)); np.add.at(scores,codes,rt*error[:,None])
    bread=linalg.inv(rt.T@rt); cov=g/(g-1)*(n-1)/(n-k)*bread@(scores.T@scores)@bread; cov=(cov+cov.T)/2
    assert np.all(np.diag(cov)>0) and np.allclose(cov,cov.T)
    # Dimensionless orthogonality checks stay meaningful for squared controls
    # whose raw numerical magnitudes can differ greatly across samples.
    control_orthogonality=float(np.max(np.abs(ww.T@residual)/np.maximum(np.linalg.norm(ww,axis=0)[:,None]*np.linalg.norm(residual,axis=0)[None,:],1e-12)))
    focus_orthogonality=float(np.max(np.abs(rt.T@error)/np.maximum(np.linalg.norm(rt,axis=0)*np.linalg.norm(error),1e-12)))
    assert control_orthogonality<1e-7 and focus_orthogonality<1e-7,(control_orthogonality,focus_orthogonality)
    if direct_check:
        full=np.column_stack([ww,t*sw[:,None]]); direct,_,_,_=linalg.lstsq(full,yy*sw,cond=1e-12,lapack_driver='gelsd')
        assert np.allclose(beta,direct[-q:],rtol=1e-7,atol=1e-8)
        influence=linalg.pinv(full,rtol=1e-12)[-q:,:].T; errors=yy*sw-full@direct
        score=np.zeros((g,q)); np.add.at(score,codes,influence*errors[:,None])
        direct_cov=g/(g-1)*(n-1)/(n-k)*(score.T@score)
        assert np.allclose(cov,direct_cov,rtol=1e-6,atol=1e-10)
    return {'beta':beta,'covariance':cov,'n':n,'hosts':d.iloc[index].host_id.nunique(),'clusters':g,'rank':int(k),'df':g-1,'control_orthogonality':control_orthogonality,'focus_orthogonality':focus_orthogonality}

def contrast(f,c,null=0.):
    c=np.array(c,float); b=float(c@f['beta']); se=float(np.sqrt(c@f['covariance']@c)); z=stats.t.ppf(.975,f['df']); lo=b-z*se; hi=b+z*se
    return {'coefficient':b,'se':se,'null':float(null),'ci_low':lo,'ci_high':hi,'p_two_sided':float(2*stats.t.sf(abs((b-null)/se),f['df'])),'pct_change':float(100*np.expm1(b)),'pct_ci_low':float(100*np.expm1(lo)),'pct_ci_high':float(100*np.expm1(hi))}
