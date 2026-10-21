import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
z=np.linspace(1e-4,4*np.pi,2000)
g=np.exp(-z**2/6); b=np.sin(z)/z; h=np.where(z<np.pi,b,0)
fig,ax=plt.subplots(1,2,figsize=(7,2.5))
ax[0].axhline(1,color="#888",ls=":",lw=1,label="PE (no attenuation)")
ax[0].plot(z/np.pi,g,color="#1f6fb4",lw=1.6,label="Gaussian IPE  exp(-z²/6)")
ax[0].plot(z/np.pi,b,color="#c0392b",lw=1.6,label="Box-IPE  sin z / z (exact)")
ax[0].plot(z/np.pi,h,color="#27ae60",lw=1.2,ls="--",label="Lobe-IPE (main lobe)")
ax[0].axhline(0,color="k",lw=.5); ax[0].set_xlabel("z/π = ωw/(2π)  (cycles per footprint)"); ax[0].set_ylabel("feature gain")
ax[0].legend(fontsize=7,frameon=False); ax[0].set_title("(a) attenuation vs. frequency",fontsize=9)
# per-level gains for L=10 at footprint widths of the 4 training scales
L=10; om=2.0**np.arange(L)*np.pi
for s,c in zip([2,4,8,16],["#bbb","#888","#555","#000"]):
    w=s*2/512; zz=om*w/2
    ax[1].plot(range(L),np.sin(zz)/zz,"o-",color=c,ms=3,lw=1,label=f"box, s={s}")
    ax[1].plot(range(L),np.exp(-zz**2/6),"x--",color=c,ms=3,lw=.8)
ax[1].axhline(0,color="k",lw=.5); ax[1].set_xlabel("frequency level l  (ω = 2^l π)"); ax[1].set_ylabel("gain")
ax[1].set_title("(b) gains per level (o box, x Gaussian)",fontsize=9); ax[1].legend(fontsize=7,frameon=False)
plt.tight_layout(); plt.savefig("figs/fig_attenuation.png",dpi=300)
print("max |gauss-sinc| on main lobe:",np.max(np.abs(g-b)[z<np.pi]), "argmax z/pi", z[np.argmax(np.abs(g-b)*(z<np.pi))]/np.pi)
print("first negative lobe min sinc:",b.min())
# Proposition 2 Monte-Carlo check: oriented 3D box
rng=np.random.default_rng(0); errs=[]
for trial in range(200):
    Q,_=np.linalg.qr(rng.normal(size=(3,3))); hw=rng.uniform(.01,.3,3); mu=rng.normal(size=3)
    e=rng.normal(size=3); e/=np.linalg.norm(e); omg=rng.uniform(1,40)
    u=rng.uniform(-1,1,(400000,3))*hw; x=mu+u@Q.T
    mc=np.mean(np.sin(omg*x@e))
    a=Q.T@e  # components of e along box axes
    cf=np.prod(np.sinc(omg*a*hw/np.pi))  # np.sinc(t)=sin(pi t)/(pi t)
    an=np.sin(omg*mu@e)*cf
    errs.append(abs(mc-an))
print("Prop2 MC abs err: mean %.2e max %.2e (MC std ~ %.1e)"%(np.mean(errs),np.max(errs),1/np.sqrt(400000)))
