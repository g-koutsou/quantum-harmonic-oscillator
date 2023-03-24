from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

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
        else:
            if n not in self.jkdata:
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
    
class HarmonicOscillator:
    def __init__(self, mass, omega, T, N_t):
        self.mass = mass
        self.omega = omega
        self.N_t = N_t
        self.delta_t = T/N_t
        self.T = T
        self.ma = mass*self.delta_t
        self.om = omega*self.delta_t
        self.x = np.zeros(shape=[self.N_t], dtype=np.float64)

    def x_init(self, start="random"):
        assert start in ["random", "zeros"]
        if start == "random":
            self.x = np.random.normal(size=[self.N_t])
        if start == "zeros":
            self.x = np.zeros(shape=[self.N_t], dtype=np.float64)

    def set_delta(self, delta):
        self.delta = delta

    def get_delta(self):
        return self.delta
            
    def update_delta(self, r):
        self.delta *= r
            
    def S(self, x=None):
        if x is None:
            x = np.array(self.x)
        dx = (np.roll(x, shift=-1) - x)
        return (0.5*self.ma*dx**2 + 0.5*self.ma*self.om**2*x**2).sum()

    def dS(self, i0, x_trial):
        dx2 = x_trial**2 - self.x[i0]**2
        dx = x_trial - self.x[i0]
        xp1 = np.roll(self.x, shift=-1)[i0]
        xm1 = np.roll(self.x, shift=+1)[i0]
        return self.ma*dx2*(1+self.om**2/2) - self.ma*(xp1+xm1)*dx

    def x_trial(self, i0):
        return self.x[i0] + self.delta*(2*np.random.random()-1)

    def update(self, i0, x_trial):
        self.x[i0] = x_trial
        
    def update_all(self, x_trial):
        self.x = np.array(x_trial)

    def checkpoint(self):
        if not hasattr(self, "xs"):
            self.xs = np.array([self.x])
        else:
            self.xs = np.concatenate((self.xs, [self.x]))
            
    def Ss(self):
        dx = (np.roll(self.xs, axis=1, shift=-1) - self.xs)
        return (0.5*self.ma*dx**2 + 0.5*self.ma*self.om**2*self.xs**2).sum(axis=1)

    def xsq(self):
        x2 = (self.xs**2).mean(axis=1)
        return x2*self.delta_t**2
    
    def metr_sweep(self):
        r_acc = 0
        S0 = self.S()
        for _ in range(self.N_t):
            i = np.random.randint(N_t)
            xt = self.x_trial(i)
            delS = self.dS(i, xt)
            if delS < 0:
                self.update(i, xt)
                S0 += delS
                r_acc += 1
            elif np.random.random() < np.exp(-delS):
                self.update(i, xt)
                S0 += delS
                r_acc += 1
        return r_acc/self.N_t

    def force(self, x):
        xp1 = np.roll(x, shift=-1)
        xm1 = np.roll(x, shift=+1)
        return (xp1 + xm1 - x * (2+self.om**2))*self.ma

    def H(self, pi, x):
        return np.sum(pi**2)/2 + self.S(x=x)
    
    def hmc(self, delta_tau, N_tau):
        x0 = np.array(self.x)        
        pi = np.random.normal(size=[self.N_t])
        H0 = self.H(pi, x0)
        pi = pi + delta_tau/2 * self.force(x0)
        for i in range(1, N_tau+1):
            x0 = x0 + delta_tau * pi
            if i == N_tau:
                pi = pi + delta_tau/2 * self.force(x0)
            else:
                pi = pi + delta_tau * self.force(x0)
        H1 = self.H(pi, x0)
        delH = H1 - H0
        acc = 0
        if delH < 0:
            self.update_all(x0)
            acc = 1
        elif np.random.random() < np.exp(-delH):
            self.update_all(x0)
            acc = 1
        return acc
        
        
#%%

mass = 1
omega = 1
T = 3
delta = 0.5
target_racc = 0.5
N_ts = [150,]
n_therm = 300
n_mc = 2_500_000
n_print = n_mc//10

algo = "hmc"

harmos = {}
for N_t in N_ts:
    harmo = HarmonicOscillator(mass, omega, T, N_t)
    harmo.x_init(start="random")
    harmo.set_delta(delta)
    acc = 0
    for i_mc in range(n_therm + n_mc):
        if algo == "metr":
            r_acc = harmo.metr_sweep()
            if i_mc < n_therm:
                if r_acc > target_racc:
                    harmo.update_delta(1.005)
                else:
                    harmo.update_delta(0.995)
        if algo == "hmc":
            acc += harmo.hmc(0.5, int(1/.5))
            r_acc = acc/(i_mc+1)
        S = harmo.S()
        if i_mc % n_print == 0:
            print(f"N={N_t:4.0f}\ti_mc={i_mc:8.0f}\t S={S:+8.3e}\t r={r_acc:3.2f}\t delta={harmo.get_delta():3.2e}")
        harmo.checkpoint()
    harmos[N_t] = harmo
    del harmo
        
