from liftingline.lift_dist.lift_dist import *
import numpy as np
from liftingline.lift_dist import WLiftDist, Trap_WLiftDist


def test_get_var():
    LLopt = LiftDistVb(1, Na=5, Ny=11, cd0_model='constant',cl_model='flat_plate')
    d = LLopt.initial_guess()
    x = LLopt.set_vars(d)
    x_ = LLopt.get_vars(x, dic=True)
    assert [np.allclose(d[k], x_[k]) for k in d],f"Set var {d} different from getvar {x_}"
    print("get var test passed <-- \n")

def test_ellipse_ND():
    print("\nTesting ellipse -->")
    W = 10
    Na = 5
    Ny = 101
    LD = LiftDistND(W, Na=Na, Ny=Ny, cd0_model='constant',cl_model='flat_plate')
    b = 5
    V = 1
    c = np.sin(LD.theta_pts)[:, 0]
    S = pi * (b/2) * 1 / 2
    _AR = S / b**2
    x = {"Cw_AR":W / (1/2 * rho * V**2 * b**2),
        "c_b":c/b, "A":np.zeros(Na)}
    A1, AR, Re, cd0, CD0, cl, CL, CDi, L_D = LD.metrics(x)

    assert np.isclose(_AR, 1/AR), f"{_AR} is different from computed 1/AR {1/AR} for ND param"
    print("\n Ellipse area test passed")

    ali = A1
    ali_nd = LD.alpha_i(A1, np.zeros(LD.Na))
    assert np.allclose(ali_nd, ali), f"computed induced AOA {ali_nd} different from {ali} for nd param"
    print("\n Ellipse induced AoA test passed")

    CL = pi * A1 * AR
    CL_ = LD.W / (1/2 * rho * V**2 * S)
    CLi = AR * LD.w_th[1:] @ (cl * c / b )[1:]
    assert np.isclose(CL, CL_) , f"computed CL {CL} different from {CL_}"
    assert np.isclose(CL, CLi) , f"integrated CL {CLi} different from {CL_} for nd param"
    print("\n Ellipse CL test passed (runtime warning is ok)")

    CDi_ = CL**2 / (AR*pi)
    assert np.allclose(CDi, CDi_), f"computed CDi {CDi} different from {CDi_}"
    print("\n Ellipse CDi test passed")

    Re_  = rho * V * c / nu
    assert np.allclose(Re[:Ny], Re_), f"Computed Re {Re} different from {Re_} for nd param"
    print("\n Ellipse Re test passed")

    LD = LiftDistND(1, Na=5, Ny=11, cd0_model='flat_plate',cl_model='flat_plate')
    LD.variables["c_b"].lb = 0.1
    LD.variables["A"].lb =-0.1
    LD.variables["c_b"].ub = 0.3
    LD.variables["A"].ub = 0.1
    x0 = LD.initial_guess() 
    x, sol = LD.optimize(x0, alg="IPOPT", restarts=1, solver_options={'max_iter':500, 'tol':1e-6})
    assert np.allclose(x["A"], 0., atol= 1e-4), f"Optimal distribution is not elliptical: {x}"
    
    print(sol)

def test_ellipse_Vb():
    print("\nTesting ellipse -->")
    W = 10
    LD = LiftDistVb(W, Na=5, Ny=101, cd0_model='flat_plate',cl_model='flat_plate')
    b = 5
    V = 1
    c = np.sin(LD.theta_pts)[:, 0]
    A = np.zeros(LD.Na)
    xdict = {"Vb2":(V*b)**2, "c_2b": c/2/b, "A":A}

    A1, AR, Re, cd0, CD0, cl, CL, CDi, L_D = LD.metrics(xdict)

    S = pi * (b/2) * 1 / 2
    _AR = S / b**2
    assert np.isclose(_AR, 1/AR), f"{_AR} is different from computed 1/AR {1/AR} for Vb param"
    print("\n Ellipse area test passed")

    ali = A1
    ali_ = LD.alpha_i(A1, np.zeros(LD.Na))
    assert np.allclose(ali_, ali), f"computed induced AOA {ali_} different from {ali}"
    print("\n Ellipse induced AoA test passed")

    CL0 = pi * A1 / _AR
    CL_ = LD.W / (1/2 * rho * V**2 * S)
    assert np.isclose(CL0, CL_) , f"computed CL {CL0} different from {CL_}"
    assert np.isclose(CL0, CL) , f"integrated CL {CL} different from {CL_}"
    print("\n Ellipse CL test passed (runtime warning is ok)")

    CDi_ = CL**2 / (AR * pi)
    assert np.allclose(CDi, CDi_), f"computed CDi {CDi} different from {CDi_}"
    print("\n Ellipse CDi test passed")

    Re_  = rho * V * c / nu
    assert np.allclose(Re, Re_), f"Computed Re {Re_} different from {Re}"
    print("\n Ellipse Re test passed")

    LD = LiftDistVb(1, Na=5, Ny=11, cd0_model='flat_plate',cl_model='flat_plate')

    LD.variables["c_2b"].lb = 0.1
    LD.variables["c_2b"].ub = 0.3
    LD.variables["A"].lb = -0.1
    LD.variables["A"].ub = 0.1
    x0 = LD.initial_guess() 
    x, sol = LD.optimize(x0, alg="IPOPT", restarts=1, solver_options={'max_iter':500, 'tol':1e-6})
    assert np.allclose(x["A"], 0., atol= 1e-5), f"Optimal distribution is not elliptical: {x}"
    print(sol)

def test_GPFourrier():
    LD = GPLiftDistFourrier(1, Na=5, Ny=11, cd0_model='flat_plate',cl_model='flat_plate')
    x0 = LD.initial_guess({"A":np.zeros(LD.Na)})
    A0 = x0['A']
    c_2b0 = x0['c_2b']
    Vb20 = x0['Vb2']
    
    
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

    LD.c_2b.lb = 0.1
    LD.c_2b.ub = np.inf
    x, prob = LD.optimize(x0)
    assert prob.status == "optimal", ("Problem did not converge")
    assert np.allclose(x["c_2b"], 0.1), f"c_2b {x} not equal to lower bound 0.1"
    assert np.allclose(x["A"], 0., atol=1e-4), f"A {x} is not close to 0: converged to non elliptica distribution"


def test_GP():
    LD = GPLiftDist(1, Ny=11, cd0_model='flat_plate',cl_model='flat_plate')
    x0 = LD.initial_guess()
    
    d = {
        k:cp.Variable(pos=True, shape=np.atleast_1d(x0[k]).shape, value=np.atleast_1d(x0[k])) for k in x0
    }

    met = LD.metrics(d)

    for m in met:
        assert m.is_dgp(), m.name + " is not DGP"

    obj = LD.objective(d, [])['obj']
    assert obj.is_dgp(), "Objecive function is not DGP"

    LD.AR.ub = 10.
    x, prob = LD.optimize(x0)
    # assert prob.status == "optimal", ("Problem did not converge")
    # assert np.allclose(x["c_2b"], 0.1), f"c_2b {x} not equal to lower bound 0.1"
    # assert np.allclose(x["A"], 0., atol=1e-4), f"A {x} is not close to 0: converged to non elliptica distribution"

class LDtestUB(LiftDistND):
    def objective_(self, xdict, constraints):
        obj = super().objective_(xdict, constraints)
        A1 = self.A1(xdict["Cw_AR"])
        obj['obj'] = np.sum((self.cl(A1, xdict['A'], xdict['c_b']))**2)
        return obj


def test_bounds():
    LD = LDtestUB(1, Na=5, Ny=11, cd0_model='flat_plate',cl_model='flat_plate',
                    bounds={"lb":{"Cw_AR":0.}, "ub":{"Cw_AR":0.8}})
    x0 = LD.initial_guess()
    constraints=[dict(name="clCon", ub=1., lb=0.1),
                                dict(name="ReCon", ub=200000., lb=100000)]
    x, sol = LD.optimize(x0, alg='IPOPT', 
                    constraints=constraints, 
                    solver_options={'tol':1e-18, "max_iter":700}, sens=None, restarts=5)
    A1, AR, Re, cd0, CD0, cl, CL, CDi, L_D = LD.metrics(x)
    import pdb; pdb.set_trace()
    assert np.allclose(Re, 100000, atol=00, rtol=0.25), f"Re is not at its min: {Re}"
    assert np.allclose(cl[1:], 0.1, atol=0, rtol=.2), f"cl is not at its min: {cl[1:]}"


def test_metrics(A, C, W=2, Ny=11):
    Nc = C.shape[0]
    Na = A.shape[0]

    
    LD = [ WLiftDist(W=W, Ny=Ny, Na=Na, cd0_model='flat_plate'),
            LiftDistND(W=W, Ny=Ny, Na=Na, cd0_model='flat_plate'),
            FreeChordLiftDist(W=W, Ny=Ny, Na=Na, cd0_model='flat_plate'),
            # LiftDistVb(W=W, Ny=Ny, Na=Na, cd0_model='flat_plate'),
            SpecifiedLiftDist(W=W, Ny=Ny, Na=Na, A=A, cd0_model='flat_plate'),
            ChebChordLiftDist(W=W, Ny=Ny, Nc=Nc, Na=Na, cd0_model='flat_plate')]
    
    CL = 0.8
    AR = 12
    c = LD[-1].c(C, AR)

    if np.allclose(c, 1):
        LD.append(RectangularWingLiftDist(W=W, Ny=Ny, Na=Na, cd0_model='flat_plate'))
    
    if np.allclose(A, 0):
            LD.append(
            SimpleLiftDist(W=W, Ny=Ny, cd0_model='flat_plate'))
    
    c_b  = c / AR
    Vb2 = AR/CL/(1/2*rho)
    b = 1.2
    xdict = [
        dict(CL=CL, c_b=c_b, A=A, b=b),
        dict(Cw_AR=CL/AR, c_b=c_b, A=A),
        # dict(Vb2=Vb2, c_2b=c_b/2, A=A),
        dict(CL=CL, AR=AR, A=A, c=c),
        dict(Cw_AR=CL/AR, c_b=c_b),
        dict(CL=CL, AR=AR, A=A, C=C),
        dict(CL=CL, AR=AR, A=A),
        dict(CL=CL, AR=AR)
    ]


    def get(i, M):
        return np.vstack([m[i] for m in M])

    # Set weight to the one computed with weight model
    m0 = LD[0].metrics(xdict[0])
    for ld in LD[1:]:
        ld.W = m0[-1]

    # Compare metrics
    metrics = [m0] + [ld.metrics(x) for ld, x in zip(LD, xdict)]

    stacked_metrics = [get(i, metrics) for i in range(len(metrics[0]))]

    for i in range(len(stacked_metrics)):
        si = stacked_metrics[i]
        for j in range(si.shape[0]):
            assert np.allclose(si[j], si[0], rtol=1e-3, atol=1e-3), \
                f'Metric number {i} not matching for LD number {j}: {si[j]} vs {si[0]}'

    # Compare objectives
    obj = [ld.objective(x)['obj'] for ld, x in zip(LD[1:], xdict[1:])]
    for i in range(len(obj)):
        assert np.isclose(obj[i], obj[0]), f'Objective not matching for LD number {i}: {obj[i]} versus {obj[0]}'

if __name__ == "__main__":
    # test_get_var()
    # test_ellipse_Vb()
    # test_GPFourrier()
    # test_GP()
    # test_ellipse_ND()
    # test_bounds()

    # test metrics
    Nc = 5
    Na = 4
    W = 2

    # Elliptic loading, fixed chord
    test_metrics(A=np.zeros(Na), C=np.zeros(Nc), W=2)

    # Random loading, constant chord
    A = np.random.uniform(-1.,1.,size=Na)
    test_metrics(A=A, C=np.zeros(Nc), W=2)

    # Random loading, random chord
    C = np.random.uniform(-.1, .1, size=Nc)
    test_metrics(A=A, C=C, W=2)
