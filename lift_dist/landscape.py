from lift_dist import *

rho = 1.225
pi = np.pi
nu = 1.81*1e-5


def gensol(LD):
    x0 = LD.initial_guess()
    print(x0)
    x, sol = LD.optimize(x0, "IPOPT", 
            obj_scale=1.,
            solver_options={"max_iter":200, "acceptable_tol":1e-10},
            smooth_penalty={"c_b":0.1})
    np.save("sol", x)

if __name__ == "__main__":
    W = 10
    LD = LiftDistND(W, Na=5, Ny=51, 
                    cd0_model='flat_plate',
                    cl_model='flat_plate',
                    bounds={
                        'ub':{"c_b":0.2, "A":0.1},
                        'lb':{"c_b":0.1, "A":-0.1, "Cw_AR":1e-10}
                    })

    gensol(LD)
    x = np.load("sol.npy")
    Cw_AR, c_b, A = LD.get_vars(x)
    A1 = LD.A1(Cw_AR)
    _AR = LD._AR(c_b)
    CDi = LD.CDi(A1, A, _AR)
    cd0 = LD.CD0_2D(Cw_AR, c_b, A, A1)
    print("AR", 1/_AR)
    print("CDi", CDi)
    print('cd0',cd0)
    print((CDi + LD.w_th @ (cd0 * c_b)))
    plt.plot(LD.theta_pts, c_b)
    plt.show()





    