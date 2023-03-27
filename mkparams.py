import json

mass = 1
omega = 1

prms = []
#for T in (1, 2, 3, 4, 5, 6):
#    for delta_t in (0.01, 0.025, 0.03, 0.04, 0.05, 0.06, 0.075, 0.085, 0.1, 0.12, 0.14, 0.15):
for T in (1, 2, 3, 4):
    for delta_t in (0.025, 0.05, 0.06, 0.075, 0.085, 0.1, 0.125, 0.15):
        N = int(round(T/delta_t))
        if N % 2 == 1:
            N += 1
        Tx = N*delta_t
        print(f"δt={delta_t:6.4f}\t Nt={N:4.0f}\t T={Tx:6.3f}")
        prms.append(
            {
                "mass"          : mass,
                "omega"         : omega,
                "delta_t"       : delta_t,
                "N"             : N
            }
        )
    print()
    
json.dump(prms, open("params.json", "w"), indent=4)
