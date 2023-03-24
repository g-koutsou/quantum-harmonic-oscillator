from collections import defaultdict
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import gvar as gv
import hashlib
import pickle
import lsqfit
import string
import json

def short_hash(ens):
    return hashlib.sha1(pickle.dumps(ens)).hexdigest()[:10]

class HarmonicOscillator:
    def __init__(self, mass, omega, T, delta_t, fname):
        self.mass = mass
        self.omega = omega
        self.delta_t = delta_t
        self.T = T
        self.om = omega*delta_t
        self.ma = mass*delta_t
        self.x = np.load(open(fname, "rb"))["arr_0"]/delta_t
        assert self.x.shape[1] == int(T/delta_t)
        
    def S(self):
        dx = (np.roll(self.x, axis=1, shift=-1) - self.x)
        return (0.5*self.ma*dx**2 + 0.5*self.ma*self.om**2*self.x**2).sum(axis=1)

    def xsq(self):
        x2 = (self.x**2).mean(axis=1)
        return x2*self.delta_t**2

class Observable:
    def __init__(self, meas):
        self.meas = np.array(meas)

    def jackknife(self, n):
        if not hasattr(self, "jkdata"):
            self.jkdata = {}
        old_shape = self.meas.shape
        data = self.meas.reshape([self.meas.shape[0], -1])
        N = int(data.shape[0]/n)*n
        data = data[:N,:].reshape([int(N/n),n,-1])
        data = (data.sum(axis=0).sum(axis=0) - data.sum(axis=1))/(N-n)
        self.jkdata[n] = data.reshape((data.shape[0],) + old_shape[1:])

    def mean(self):
        return self.meas.mean(axis=0)
        
    def _err(self, n):
        return self.jkdata[n].std(axis=0)*np.sqrt(self.jkdata[n].shape[0]-1)
        
    def jkerr(self, n):
        if not hasattr(self, "jkdata"):
            self.jackknife(n)
        elif n not in self.jkdata:
            self.jackknife(n)
        return self._err(n)

    def autocorr(self):
        N = len(self.meas)
        i = np.arange(N)
        ct = np.fft.fft(self.meas - self.meas.mean(axis=0), axis=0)
        r = np.fft.ifft(ct * ct[(N-i)%N,...]).real
        return r/((self.meas-self.meas.mean(axis=0))**2).sum(axis=0)

    def tau_int(self):
        rho = self.autocorr()
        return 0.5 + rho[1:,...].cumsum(axis=0)

params = json.load(open("params.json", "r"))
n_therm = 4_000
harmos = {}
ensembles = {}
for i,ens in enumerate(params):
    n = short_hash(ens)
    fname = f"data/x-{n}.txt.npz"
    N = ens["N"]
    mass = ens["mass"]
    omega = ens["omega"]
    alat = ens["delta_t"]
    T = N*alat
    harmos[n] = HarmonicOscillator(mass, omega, T, alat, fname)
    ensembles[n] = {"alat": alat, "T": T, "N": N}

def int_T(n):
    return int(np.round(ensembles[n]["T"]))

def ens_T(ensembles, T):
    return [n for n in ensembles if int_T(n) == T]

Ts = sorted(set([int_T(n) for n in ensembles]))
NTs = len(Ts)

#%%
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']*2

fig = plt.figure(1, figsize=(3+2*NTs, 6))
fig.clf()
gs = mpl.gridspec.GridSpec(1, NTs, left=0.075, right=0.99, bottom=0.075, top=0.99)
for j,T in enumerate(Ts):
    ax = fig.add_subplot(gs[j])
    for i,ens in enumerate(ens_T(ensembles, T)):
        T = ensembles[ens]["T"]
        delta_t = ensembles[ens]["alat"]
        y = harmos[ens].S()
        x = np.arange(y.shape[0])
        ax.plot(x[:], y[:], lw=0.5, color=colors[i], label=f"$T={T}$ $δt={delta_t}$", zorder=-1)
    ax.legend(loc="upper left", frameon=False, ncols=1)
    ax.set_xlabel("$i_{mc}$")
fig.axes[0].set_ylabel("$S$")
fig.show()

#%%
tau_int = {}
t_cut = 1_000
fig = plt.figure(2, figsize=(3+2*NTs, 6))
fig.clf()
gs = mpl.gridspec.GridSpec(1, NTs, left=0.075, right=0.99, bottom=0.075, top=0.99)
for j,T in enumerate(Ts):
    ax = fig.add_subplot(gs[j])
    for i,ens in enumerate(ens_T(ensembles, T)):
        T = ensembles[ens]["T"]
        delta_t = ensembles[ens]["alat"]
        y = Observable(harmos[ens].xsq()[n_therm:])
        rho = y.autocorr()[:t_cut]
        t = np.arange(t_cut)
        ax.plot(t, rho, color=colors[i], marker=".", ls="", ms=2, label=f"$T={T}$ $δt={delta_t}$")
        tm = np.where(rho < 0.1)[0][0]
        if tm == 1:
            alpha = 1
        else:
            alpha = -(np.log(rho[:tm])*t[:tm]).sum()/(t[:tm]**2).sum()
        ax.plot(t, np.exp(-t*alpha), ls="--", color=colors[i])
        tau_int[ens] = int(3/alpha)#int((1+np.exp(-alpha))/(2*(1-np.exp(-alpha))))
    ax.set_xlabel(r"$t$")
    ax.hlines(0, 0, len(rho), color="k", alpha=0.2, ls="--")
    ax.set_xlim(0, t_cut)
    ax.legend(loc="upper left", frameon=False, ncols=1)
fig.axes[0].set_ylabel(r"$\rho_{\langle x^2\rangle}(t)$")    
fig.show()

#%%
fig = plt.figure(3, figsize=(6, 4))
fig.clf()
gs = mpl.gridspec.GridSpec(1, 1, left=0.125, right=0.99, bottom=0.125, top=0.99)
fig.clf()
ax = fig.add_subplot(gs[0])
ax.set_yscale("log")
for j,T in enumerate(Ts):
    for i,ens in enumerate(ens_T(ensembles, T)):
        y = tau_int[ens]
        x = ensembles[ens]["T"]/ensembles[ens]["N"]
        m, = ax.plot(x, y, ls="", marker="o", color=colors[j])
    m.set_label(f"T={T}")
ax.legend(loc="upper right", frameon=False)
ax.set_xlabel("δt")
ax.set_ylabel(r"$\tau_{int}$")
fig.show()

#%%
fig = plt.figure(4, figsize=(6, 4))
fig.clf()
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
xy = defaultdict(list)
fits = dict()
n_jk = 10
for j,T in enumerate(Ts):
    for i,ens in enumerate(ens_T(ensembles, T)):
        alat = ensembles[ens]["T"]/ensembles[ens]["N"]
        xsq = Observable(harmos[ens].xsq()[n_therm::max(tau_int[ens], 1)])
        ave = xsq.mean()
        m,_,_ = ax.errorbar(alat, ave, xsq.jkerr(n_jk), color=colors[j], marker="o", zorder=j+1)
        xy[T].append([alat, gv.gvar(ave, xsq.jkerr(n_jk))])
    m.set_label(f"T={T}")
    x,y = np.array(xy[T]).T
    fcn = lambda x,p: p[0] + p[1]*x + p[2]*x**2
    xcut = 0.2
    y = y[x <= xcut]
    x = x[x <= xcut]
    fits[T] = lsqfit.nonlinear_fit(data=(x,y), fcn=fcn, p0=(1,1,1))
    print(f"T={T} | χ2/d.o.f = {fits[T].chi2/fits[T].dof:3.2f}")
    x = np.linspace(0, max(x))
    y = fcn(x, fits[T].p)
    ax.plot(x, gv.mean(y), ls="--", color=colors[j], lw=0.5)
    ax.fill_between(x, gv.mean(y)+gv.sdev(y), gv.mean(y)-gv.sdev(y),
                    color=colors[j], alpha=0.15)
ax.legend(loc="upper right", frameon=False)
ax.set_ylabel(r"$\langle x^2\rangle$")
ax.set_xlabel(r"$\delta_t$")
ax.text(0.5, 0, r"Separate $\delta t\rightarrow 0$ fits", ha="center", va="bottom", transform=ax.transAxes)
ax.set_ylim(0.3, 1.2)
ax.set_xlim(0)
fig.show()


#%%
fig = plt.figure(5, figsize=(6, 4))
fig.clf()
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
xy = list()
n_jk = 10
for j,T in enumerate(Ts):
    for i,ens in enumerate(ens_T(ensembles, T)):
        alat = ensembles[ens]["T"]/ensembles[ens]["N"]
        xsq = Observable(harmos[ens].xsq()[n_therm::max(tau_int[ens], 1)])
        ave = xsq.mean()
        m,_,_ = ax.errorbar(alat, ave, xsq.jkerr(n_jk), color=colors[j], marker="o", zorder=j+1)
        xy.append([alat, ensembles[ens]["T"], gv.gvar(ave, xsq.jkerr(n_jk))])
    m.set_label(f"T={T}")
def a_term(T, p):
    return p[0] + p[1]*np.exp(-T*p[2])
def fcn(x, prms):
    alat,L = x
    ret = 0
    for i,p in enumerate(np.array(prms).reshape([-1,3])):
        ret += alat**i*a_term(L, p)
    return ret
alat,L,y = zip(*xy)
alat = np.array(alat)
L = np.array(L)
fits["comb"] = lsqfit.nonlinear_fit(data=((alat,L),y), fcn=fcn, p0=(1,1,1)*3)
print(f"T=all | χ2/d.o.f = {fits['comb'].chi2/fits['comb'].dof:3.2f}")
for j,T in enumerate(Ts):
    x = np.linspace(0, 0.15)
    y = fcn((x, x**0*T), fits["comb"].p)
    ax.plot(x, gv.mean(y), ls="--", color=colors[j], lw=0.5)
    ax.fill_between(x, gv.mean(y)+gv.sdev(y), gv.mean(y)-gv.sdev(y),
                    color=colors[j], alpha=0.15)
ax.legend(loc="upper right", frameon=False)
ax.set_ylabel(r"$\langle x^2\rangle$")
ax.set_xlabel(r"$\delta_t$")
ax.text(0.5, 0, r"Combined $\delta t\rightarrow 0$, $T\rightarrow\infty$ fit",
        ha="center", va="bottom", transform=ax.transAxes)
ax.set_ylim(0.3, 1.2)
ax.set_xlim(0)
fig.show()

#%%
fig = plt.figure(6, figsize=(6, 4))
fig.clf()
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
xy = list()
for j,T in enumerate(Ts):
    x = 1/T
    y = fits[T].p[0]
    ax.errorbar(x, gv.mean(y), gv.sdev(y), ls="", marker="o", color=colors[0])
    xy.append((x,y))
x,y = np.array(list(zip(*xy)))
x = np.array(x, dtype=float)
fcn = lambda x,p: p[0]+p[1]*np.exp(-p[2]/x)
fits["inf"] = lsqfit.nonlinear_fit(data=(x,y), fcn=fcn, p0=(1,1,1))
x = np.linspace(0.001, 1)
y = fcn(x, fits["inf"].p)
ax.plot(x, gv.mean(y), ls="--", color=colors[0], label=r"$T\rightarrow\infty$ fit to separate continuum limits")
ax.fill_between(x, gv.mean(y)+gv.sdev(y), gv.mean(y)-gv.sdev(y),
                color=colors[0], alpha=0.15)
print()
print(f" ** {fits['inf'].p[0]:s} | χ2/d.o.f = {fits['inf'].chi2/fits['inf'].dof:3.2f}")
def fcn(x, prms):
    alat,L = x
    ret = 0
    for i,p in enumerate(np.array(prms).reshape([-1,3])):
        ret += alat**i*a_term(L, p)
    return ret
y = fcn((x*0, 1/x), fits["comb"].p)
ax.plot(x, gv.mean(y), ls=":", color="k", label=r"Combined $\delta t\rightarrow 0$, $T\rightarrow\infty$ fit")
ax.fill_between(x, gv.mean(y)+gv.sdev(y), gv.mean(y)-gv.sdev(y),
                color="k", alpha=0.15)
print(f" ** {fits['comb'].p[0]:s} | χ2/d.o.f = {fits['comb'].chi2/fits['comb'].dof:3.2f}")
ax.legend(loc="lower right", frameon=False)
ax.set_ylabel(r"$\langle x^2\rangle$")
ax.set_xlabel(r"$\frac{1}{T}$")
ax.set_xlim(0)
ax.set_ylim(0)
fig.show()

