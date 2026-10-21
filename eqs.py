import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["mathtext.fontset"]="cm"
E={
"pe": r"$\gamma(x)=\left[\sin(\omega_0 x),\cos(\omega_0 x),\ldots,\sin(\omega_{L-1}x),\cos(\omega_{L-1}x)\right]^{\top},\quad \omega_\ell=2^{\ell}\pi$",
"obs": r"$I_s(\mathbf{p})=\frac{1}{|\Omega_s(\mathbf{p})|}\int_{\Omega_s(\mathbf{p})} f(\mathbf{x})\,d\mathbf{x},\qquad \Omega_s(\mathbf{p})=\mathbf{p}+\left[-\frac{w_s}{2},\frac{w_s}{2}\right]^{2}$",
"gauss": r"$\mathrm{E}_{x\sim\mathcal{N}(\mu,\sigma^2)}[\sin(\omega x)]=\sin(\omega\mu)\,e^{-\omega^{2}\sigma^{2}/2},\qquad \sigma^{2}=w^{2}/12\;\Rightarrow\; g_{\mathrm{G}}(z)=e^{-z^{2}/6}$",
"box": r"$\mathrm{E}_{x\sim\mathcal{U}[\mu-\frac{w}{2},\,\mu+\frac{w}{2}]}[\sin(\omega x)]=\sin(\omega\mu)\,\frac{\sin z}{z},\qquad z=\frac{\omega w}{2}\;\Rightarrow\; g_{\mathrm{B}}(z)=\mathrm{sinc}(z)$",
"cos": r"$\mathrm{E}[\cos(\omega x)]=\cos(\omega\mu)\,g(z)$",
"feat": r"$\gamma_{g}(\mathbf{p},w)=\left[\,\sin(\omega_\ell p_d)\,g(\omega_\ell w/2),\;\cos(\omega_\ell p_d)\,g(\omega_\ell w/2)\,\right]_{d\in\{1,2\},\;\ell<L}$",
"lobe": r"$g_{\mathrm{L}}(z)=\mathrm{sinc}(z)\cdot\mathbf{1}[z<\pi]$",
"prop2": r"$\mathrm{E}_{\mathbf{x}\sim\mathcal{U}(\mathcal{B})}\left[\sin(\omega\,\mathbf{e}^{\top}\mathbf{x})\right]=\sin(\omega\,\mathbf{e}^{\top}\mathbf{\mu})\prod_{k=1}^{3}\mathrm{sinc}\!\left(\omega\,h_k\,\mathbf{e}^{\top}\mathbf{a}_k\right)$",
"box3": r"$\mathcal{B}=\left\{\mathbf{\mu}+\sum_{k=1}^{3}u_k\mathbf{a}_k\;:\;|u_k|\leq h_k\right\},\quad \{\mathbf{a}_k\}\ \mathrm{orthonormal}$",
"loss": r"$\mathcal{L}(\Theta)=\frac{\sum_{s}\sum_{\mathbf{p}} s^{2}\,\|F_\Theta(\gamma_g(\mathbf{p},w_s))-I_s(\mathbf{p})\|_2^2}{\sum_{s}\sum_{\mathbf{p}} s^{2}}$",
"refine": r"$\Omega(\mathbf{p},w)=\bigcup_{k=1}^{4}\Omega\!\left(\mathbf{p}+\frac{w}{4}\delta_k,\frac{w}{2}\right)\;\Rightarrow\; I(\mathbf{p},w)=\frac{1}{4}\sum_{k=1}^{4} I\!\left(\mathbf{p}+\frac{w}{4}\delta_k,\frac{w}{2}\right),\quad \delta_k\in\{\pm1\}^2$",
"cons": r"$\mathcal{L}_{\mathrm{cons}}=\mathrm{E}_{\mathbf{p},w}\left\|\,\hat{I}(\mathbf{p},w)-\mathrm{sg}\!\left[\frac{1}{4}\sum_{k=1}^{4}\,\hat{I}\!\left(\mathbf{p}+\frac{w}{4}\delta_k,\frac{w}{2}\right)\right]\right\|_2^2,\qquad \mathcal{L}=\mathcal{L}_{\mathrm{rec}}+\lambda\,\mathcal{L}_{\mathrm{cons}}$",
"zero": r"$z=\frac{\omega_\ell w_s}{2}=\pi\,2^{\ell}\frac{s}{N}\in\pi\mathbb{Z}_{>0}\;\Rightarrow\; g_{\mathrm{B}}(z)=0,\qquad g_{\mathrm{G}}(\pi)=e^{-\pi^{2}/6}\approx0.193$",
}
for k,v in E.items():
    fig=plt.figure(figsize=(0.01,0.01)); fig.text(0,0,v,fontsize=13)
    plt.savefig(f"eq/{k}.png",dpi=250,bbox_inches="tight",pad_inches=0.04,transparent=False,facecolor="white"); plt.close()
print("ok")
