from lift_dist import *

def test_get_var():
    LLopt = LiftDistVb(1, Na=5, Ny=11, cd0_model='constant',cl_model='flat_plate')
    x = LLopt.initial_guess()
    d = LLopt.get_vars(x, dic=True)
    x_ = LLopt.set_vars(d)
    assert np.allclose(x, x_),f"Set var {x} different from getvar {x_}"
    print("get var test passed <-- \n")

def test_ellipse_ND():
    print("\nTesting ellipse -->")
    W = 10
    LD = LiftDistND(W, Na=5, Ny=51, cd0_model='constant',cl_model='flat_plate')
    b = 5
    V = 1
    Cw_AR = W / (1/2 * rho * V**2 * b**2)
    c = np.sin(LD.theta_pts)[:, 0]
    c_b = c/b
    S = pi * (b/2) * 1 / 2
    _AR = S / b**2
    assert np.isclose(_AR, LD._AR(c_b)), f"{_AR} is different from computed 1/AR {LD._AR(c/2/b)} for ND param"
    print("\n Ellipse area test passed")

    A1 = LD.A1(Cw_AR)
    ali = A1
    A = np.zeros(LD.Na)
    ali_nd = LD.alpha_i(A1, np.zeros(LD.Na))
    assert np.allclose(ali_nd, ali), f"computed induced AOA {ali_nd} different from {ali} for nd param"
    print("\n Ellipse induced AoA test passed")

    CL = pi * A1 / _AR
    CL_ = LD.W / (1/2 * rho * V**2 * S)
    cl = LD.cl(A1, A, c_b)
    CLi = 1/_AR * LD.w_th[1:] @ (cl * c_b )[1:]
    assert np.isclose(CL, CL_) , f"computed CL {CL} different from {CL_}"
    assert np.isclose(CL, CLi) , f"integrated CL {CLi} different from {CL_} for nd param"
    print("\n Ellipse CL test passed (runtime warning is ok)")

    CDi = LD.CDi(A1, A, _AR)
    CDi_ = CL**2 * _AR / (pi)
    assert np.allclose(CDi, CDi_), f"computed CDi {CDi} different from {CDi_}"
    print("\n Ellipse CDi test passed")

    Re  = rho * V * c_b * b / nu
    Re_ = LD.Re(Cw_AR, c_b)
    assert np.allclose(Re[:51], Re_), f"Computed Re {Re} different from {Re_} for nd param"
    print("\n Ellipse Re test passed")

    LD = LiftDistND(1, Na=5, Ny=11, cd0_model='flat_plate',cl_model='flat_plate')
    LD.update_bounds({"lb":{"c_b": 0.1, "A": -0.1 * np.ones(LD.Na)}, "ub":{"c_b":0.3, "A": 0.1 * np.ones(LD.Na)}})
    x0 = LD.initial_guess() 
    x, sol = LD.optimize(x0, alg="IPOPT", restarts=10)
    Cw_AR, c_b, A = LD.get_vars(x)
    assert np.allclose(A, 0., atol= 1e-4), f"Optimal distribution is not elliptical: {A}"
    
    print(sol)


def test_ellipse_Vb():
    print("\nTesting ellipse -->")
    W = 10
    LD = LiftDistVb(W, Na=5, Ny=101, cd0_model='flat_plate',cl_model='flat_plate')
    b = 5
    V = 1
    Vb2 = (V*b)**2
    c = np.sin(LD.theta_pts)[:, 0]
    c_2b = c/2/b
    S = pi * (b/2) * 1 / 2
    _AR = S / b**2

    assert np.isclose(_AR, LD._AR(c_2b)), f"{_AR} is different from computed 1/AR {LD._AR(c/2/b)} for Vb param"
    print("\n Ellipse area test passed")

    A1 = LD.A1(Vb2)
    ali = A1
    A = np.zeros(LD.Na)
    ali_ = LD.alpha_i(A1, np.zeros(LD.Na))
    assert np.allclose(ali_, ali), f"computed induced AOA {ali_} different from {ali}"
    print("\n Ellipse induced AoA test passed")

    CL = pi * A1 / _AR
    CL_ = LD.W / (1/2 * rho * Vb2 * _AR)
    cl = LD.cl(A1, A, c_2b)
    CLi = 1/_AR * LD.w_th[1:-1] @ (cl * c_2b)[1:-1]
    assert np.isclose(CL, CL_) , f"computed CL {CL} different from {CL_}"
    assert np.isclose(CL, CLi) , f"integrated CL {CLi} different from {CL_}"
    print("\n Ellipse CL test passed (runtime warning is ok)")

    CDi = LD.CDi(A1, A, _AR)
    CDi_ = CL**2 * _AR / (pi)
    assert np.allclose(CDi, CDi_), f"computed CDi {CDi} different from {CDi_}"
    print("\n Ellipse CDi test passed")

    Re  = rho * V * c_2b * 2 * b / nu
    Re_ = LD.Re(Vb2, c_2b)
    assert np.allclose(Re, Re_), f"Computed Re {Re} different from {Re_}"
    print("\n Ellipse Re test passed")

    LD = LiftDistVb(1, Na=5, Ny=11, cd0_model='flat_plate',cl_model='flat_plate')
    LD.update_bounds({"lb":{"c_2b": 0.1, "A": -0.1 * np.ones(LD.Na)}, "ub":{"c_2b":0.3, "A": 0.1 * np.ones(LD.Na)}})
    x0 = LD.initial_guess() 
    x, sol = LD.optimize(x0, alg="IPOPT", restarts=20)
    Vb2, c_2b, A = LD.get_vars(x)
    assert np.allclose(A, 0., atol= 1e-5), f"Optimal distribution is not elliptical: {A}"
    
    print(sol)

def test_GP():
    LD = GPLiftDist(1, Na=5, Ny=11, cd0_model='flat_plate',cl_model='flat_plate')
    x0 = LD.initial_guess({"A":np.zeros(LD.Na)})
    Vb20, c_2b0, A0 = LD.get_vars(x0)
    
    Vb2 = cp.Variable(pos=True, value=np.squeeze(Vb20))
    A = cp.Variable(pos=True, value=A0, shape=LD.Na)
    c_2b = cp.Variable(pos=True, value=c_2b0, shape=LD.Ny)

    d = dict(
    Vb2 = cp.Variable(pos=True, value=np.squeeze(Vb20)),
    A = cp.Variable(pos=True, value=A0, shape=LD.Na),
    c_2b = cp.Variable(pos=True, value=c_2b0, shape=LD.Ny)
    )

    assert LD.Re(d["Vb2"], d['c_2b']).is_dgp()
    assert LD._AR(d['c_2b']).is_dgp()
    _AR = LD._AR(d['c_2b'])
    assert LD.A1(d["Vb2"]).is_dgp(), "A1 is not DGP"
    A1 = LD.A1(d["Vb2"])
    assert LD.CD0_2D(d["Vb2"], d['c_2b'], d["A"], A1).is_dgp(), "CD0 is not DGP"
    assert LD.obj(d).is_dgp(), "Objective is not DGP"

    d_ = {key: d[key].value for key in d}
    LD_ = LiftDistVb(1, Na=5, Ny=11, cd0_model='flat_plate',cl_model='flat_plate')

    LD.update_bounds({"lb":{"c_2b": 0.1}, "ub":{"c_2b":np.inf}})
    x, prob = LD.optimize(x0)
    assert prob.status == "optimal", ("Problem did not converge")
    Vb2, c_2b, A = LD.get_vars(x)
    assert np.allclose(c_2b, 0.1), f"c_2b {c_2b} not equal to lower bound 0.1"
    assert np.allclose(A, 0., atol=1e-4), f"A {A} is not close to 0: converged to non elliptica distribution"


class LDtestUB(LiftDistND):
    def objective_(self, xdict, constraints):
        obj = super().objective_(xdict, constraints)
        A1 = self.A1(xdict["Cw_AR"])
        obj['obj'] = np.sum((self.cl(A1, xdict['A'], xdict['c_b']))**2)
        return obj

class LDtestLB(LiftDistND):
    def objective_(self, xdict, constraints):
        obj = super().objective_(xdict, constraints)
        A1 = self.A1(xdict["Cw_AR"])
        obj['obj'] = -np.sum((self.cl(A1, xdict['A'], xdict['c_b']))**2) #\
                    #  -np.sum(self.Re(xdict['Cw_AR'], xdict['c_b']**2)/100000**2) 
        return obj

def test_bounds():
    LD = LDtestUB(1, Na=5, Ny=11, cd0_model='flat_plate',cl_model='flat_plate',
                    bounds={"lb":{"Cw_AR":0.2}, "ub":{"Cw_AR":0.8}})
    x0 = LD.initial_guess()
    A1, AR, Re, cd0, CD0, cl, CL, CDi, L_D = LD.metrics(LD.get_vars(x0, dic=True))
    print(Re)
    print(cl)
    x, sol = LD.optimize(x0, alg='IPOPT', 
                    # constraints=[dict(name="clCon", ub=1., lb=0.1),
                    #             dict(name="ReCon", ub=200000., lb=100000)], 
                    solver_options={'tol':1e-12, "max_iter":300}, sens='fd')
    Cw_AR, c_b, A = LD.get_vars(x)
    A1, AR, Re, cd0, CD0, cl, CL, CDi, L_D = LD.metrics(LD.get_vars(x, dic=True))
    print(Re)
    print(cl)

if __name__ == "__main__":
    # test_get_var()
    # test_ellipse_Vb()
    # test_GP()
    # test_ellipse_ND()

    test_bounds()