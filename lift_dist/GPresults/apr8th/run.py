import sys
sys.path.append('../..')
from lift_dist import GPLiftDistFourrier
import cvxpy as cp
import numpy as np


class RunClass(GPLiftDistFourrier):
    def CD0_2D(self, Vb2, c_2b, A, A1):
        Re = self.Re(Vb2, c_2b)
        cc = cp.multiply(A1, cp.power(c_2b, -1))
        import pdb; pdb.set_trace()
        cl = 2 * cp.multiply(self.M1, cc) # skipping A bc M can be neg
        return 0.074 / cp.power(Re, 1.1) + cp.power(cl, 2)

def gensol(LD, file_name='sol'):
    x0 = LD.initial_guess()
    x, sol = LD.optimize(x0
    )
    np.save(file_name, x)
    return x

W = 10  
Ny = 21
Na = 3
c_b_ub = 0.5 * np.ones(Ny)
c_b_lb = 1e-10 * np.ones(Ny)
LD = RunClass(W, Na=Na, Ny=Ny,
                bounds={
                    'ub':{"c_2b":c_b_ub, "A":.1, "Vb2":100.},
                    'lb':{"c_2b":c_b_lb, "A":-.1, "Vb2":1e-10}
                })

WW = np.linspace(1, 5, 10)

if __name__ == "__main__":
    np.save("data/w", WW)
    for i in range(len(WW)):
        LD.W = WW[i]
        gensol(LD, f"data/sol_{i}")
        print(f"Weight {WW[i]}")