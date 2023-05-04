import subprocess
import hashlib
import pickle
import string
import json
import sys
import os

def short_hash(ens):
    return hashlib.sha1(pickle.dumps(ens)).hexdigest()[:10]

cmd = "./ho-sim"

def run_sim(mode, delta_t, tau, n_tau, ntherm, ntraj, target_acc, mass, omega, T, delta, print_every, store_every, fname):
    c = list()
    c.append(r"{}".format(cmd))
    c.append(r"{}".format(mode))
    c.append(r"--alat={}".format(delta_t))
    c.append(r"--ntraj={}".format(ntraj))
    c.append(r"--mass={}".format(mass))
    c.append(r"--freq={}".format(omega))
    c.append(r"--time={}".format(T))
    c.append(r"--print-every={}".format(print_every))
    c.append(r"--store-every={}".format(store_every))
    c.append(r"--output={}".format(str(fname)))
    if mode == "metr":
        c.append(r"--ntherm={}".format(ntherm))
        c.append(r"--target-accept={}".format(target_acc))
        c.append(r"--delta={}".format(delta))
    if mode == "hmc":
        c.append(r"--n-tau={}".format(n_tau))
        c.append(r"--tau={}".format(tau))
    r = subprocess.run(c, text=True, bufsize=1)
    return r

params = json.load(open("params.json", "r"))
try:
    lookup = json.load(open("lookup.json", "r"))
except json.JSONDecodeError:
    lookup = {}
    
for i,ens in enumerate(params):
    mode = "hmc"
    n = short_hash(ens)
    lookup[n] = ens
    fname = f"data/{mode}/x-{n}.txt"
    print(n, *ens.items())
    if os.path.exists(fname):
        continue
    ntraj = 2_500_000
    ntherm = 10_000
    target_acc = 0.5
    delta = 1
    tau = 0.5
    n_tau = 4
    N = ens["N"]
    mass = ens["mass"]
    omega = ens["omega"]
    delta_t = ens["delta_t"]    
    T = delta_t*N
    r = run_sim(mode, delta_t, tau, n_tau, ntherm, ntraj, target_acc, mass, omega, T, delta, ntraj//10, 100, fname)
    
json.dump(lookup, open("lookup.json", "w"), indent=4)
