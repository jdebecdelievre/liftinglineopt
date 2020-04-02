from lift_dist import *

rho = 1.225
pi = np.pi
nu = 1.81*1e-5
from jax import jit
from jax import numpy as np

def sigmoid(x):
    return 0.5 * (np.tanh(x / 2) + 1)

model = np.load('../airfoil_model/weights.npy', allow_pickle=True)[None][0]
#  = {"w0":w0, "w1":w1, "b0":b0, "b1":b1,
# "input_mean":i_mean, "input_std":i_std, "output_mean":o_mean, "output_std":o_std}
def cd0_value(cl, Re):
    z = np.vstack((cl, Re)).T
    z = (z - model["input_mean"])/model["input_std"]
    z = z @ model["w0"].T + model["b0"]
    z = sigmoid(z)
    z = z @ model["w1"].T + model["b1"]
    return z * model["output_std"] + model["output_mean"]

def gensol(LD, file_name='sol', restarts=1):
    x0 = LD.initial_guess()
    x, sol = LD.optimize(x0, "IPOPT", 
            obj_scale=10.,
            solver_options={"max_iter":2000, 'tol':1e-10, 'output_file':file_name+".ipopt"},
            usejit=False,
            restarts=restarts,
            smooth_penalty={"c_b":1e-8},
            constraints={
            "ReCon":[100000,
                    400000],
                "clCon":[-0.4,1.1]}
            )
    np.save(file_name, x)
    return x

W = 100
LD = LiftDistND(W, Na=4, Ny=21, 
                cd0_model ="flat_plate",
                cd0_val = cd0_value,
                cl_model='flat_plate',
                bounds={
                    'ub':{"c_b":0.9, "A":.1},
                    'lb':{"c_b":0.01, "A":-.1, "Cw_AR":0.001}
                }
                )

if __name__ == "__main__":
    # gensol(LD)
    WW = np.linspace(1,200, 8)
    for i in range(1,len(WW)):
        LD.W = WW[i]
        gensol(LD, f"rslt/sol_{i}", restarts=5)
    np.save("rslt/w", WW)