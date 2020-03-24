import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, NonlinearConstraint, LinearConstraint, approx_fprime
import torch
import quadpy
from llopt import *

rho = 1.225
pi = np.pi
nu = 1.81*1e-5


LLopt_tested = UnconstrainedLLopt
# x = [V, b, c, alpha, A]

def test_ll_equation():
    print("\nTesting LL equation -->")
    N_A = 6
    N_theta = 11

    # Shape, and theta=2pi
    theta_pts = np.ones((N_theta, 1))*pi/2
    Sa, Sa_, Nn = ll_equation(theta_pts, N_A, non_sorted=True)
    for i in range(int(N_A/2)):
        assert np.isclose(Sa[:, 2*i],(-1.)**i).all()
        assert (np.abs(Sa[:, 2*i+1]) < 1e-9).all()
        assert (np.abs(Sa_[:, 2*i+1]) < 1e-9).all()
        assert np.isclose(Sa_[:, 2*i], (-1.)**i*(2.*i+1.)).all()

    theta_pts = np.random.uniform(0.01, pi - 0.01, N_theta)[:, None]
    Sa, Sa_, Nn = ll_equation(theta_pts, N_A, non_sorted=True)
    assert np.allclose((Sa_ / Sa).T, np.arange(1, N_A+1)[:, None] @ (1/np.sin(theta_pts)).T)
    print("LL equation test passed <-- \n")


def test_get_var():
    print("\nTesting get_var -->")
    LLopt = LLopt_tested(1, N_A=5, N_th=11, cd0_model='constant', cl_model='flat_plate')
    x = LLopt.initial_guess()
    d = LLopt.get_vars(x, dic=True)
    x_ = LLopt.set_vars(d)
    if LLopt_tested == UnconstrainedLLopt:
        assert np.allclose(x[:-LLopt.N_A], x_[:-LLopt.N_A]),f"Set var {x} different from getvar {x_}"
    else:
        assert np.allclose(x, x_),f"Set var {x} different from getvar {x_}"
    print("get var test passed <-- \n")


def test_area():
    print("\nTesting area -->")
    w_th, theta_pts = quadrature_weights(100)
    b = 5
    cs = 2 
    y = b/2 * np.cos(theta_pts)
    c = cs * np.sqrt(1 - (y * 2/b)**2)
    plane = Plane(b=b, c=c, w_th=w_th, theta_pts=theta_pts)
    assert(np.allclose(plane.area, np.pi * b/2 * cs/2))
    print("area test passed <-- \n")


def test_ellipse():
    print("\nTesting ellipse -->")
    N_th = 101
    W = 100
    LLopt = LLopt_tested(W, N_A=5, N_th=N_th, 
                        cd0_model='constant', 
                        cl_model='flat_plate')
    b = 5
    cs = 2 
    y = b/2 * np.cos(LLopt.theta_pts)[:, 0]
    c = cs * np.sqrt(1 - (y * 2/b)**2)
    V = 12
    x = LLopt.set_vars(dict_var = dict(
        b=b,
        c=c,
        al=np.zeros(N_th),
        V=V
    ))
    plane = LLopt.get_plane(x)
    S = plane.area
    A1 = W / (0.5 * rho * V**2 * S) * 4 * S / b / pi
    CL = b * pi * A1 / 4 / plane.area
    CL_a_ellipse = 2 * pi / (1 + 2/plane.AR)
    al_ = CL / CL_a_ellipse
    x = LLopt.set_vars({"A":np.array([A1,0.,0.,0.,0.]), "al":al_}, x)


    # Ellipse L/D
    CD0 = 0.01
    CDi = CL**2 / pi / plane.AR
    D_L = (CD0 + CDi) / CL
    assert np.isclose(D_L, LLopt.obj(x), rtol=1e-4), \
        f"Obj function L/D {1/LLopt.obj(x)} not matching theoretical {1/D_L}"
    print("obj passed")

    # Ellipse alpha_i
    _, _, _, _, A = LLopt.get_vars(x)
    assert np.allclose(CL/(pi*plane.AR), LLopt.alpha_i(b, A)),f"Estimated al_i {LLopt.alpha_i(b, A)} not matching theoretical {CL/(pi*plane.AR)}"
    print("alpha_i passed")

    # Constraint satisfaction
    if LLopt_tested is not UnconstrainedLLopt:
        assert np.isclose(LLopt.enough_lift_const(x), 0.0), "Enough lift constraint not satisfied for ellipse"
        assert np.allclose(LLopt.lifting_line_const(x), 0.0), "Lifting line constraint not satisfied for ellipse"

    # convergence to ellipse
    bounds = dict(ub = {"b":5, "al":0.1}, lb={"al":-0.1, "c":cs})
    LLopt.update_bounds(bounds)
    x = LLopt.initial_guess()
    res = LLopt.optimize(x, alg='trust-constr')
    V, b, c, al, A = LLopt.get_vars(res.x)
    assert np.allclose(A[1:], 0., atol=1.), f"A = {A}"
    print("ellipse test passed <-- \n")


def test_wcon():
    print("\nTesting enough lift const -->")
    W = 1
    LLopt = LLopt_tested(W, N_A=5, N_th=11)
    x = LLopt.initial_guess()
    V, b, c, al, A = LLopt.get_vars(x)
    plane = LLopt.get_plane(x)
    CL = W/(1/2 * rho * V**2 * plane.area)
    A[0] = CL * 4 * plane.area / pi / b
    x = LLopt.set_vars({'A':A}, x)
    assert np.isclose(0., LLopt.enough_lift_const(x)), f"enough_lift con is not 0.: {LLopt.enough_lift_const(x)} "
    print("enough lift const test passed <-- \n")

if __name__ == "__main__":
    print("\n")
    test_ll_equation()
    test_get_var()
    test_area()
    test_ellipse()
    test_wcon()