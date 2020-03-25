import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, NonlinearConstraint, LinearConstraint
import torch
import quadpy
from abc import ABC, abstractmethod 
from pyoptsparse import Optimization, OPT

rho = 1.225
pi = np.pi
nu = 1.81*1e-5

# x = [V, b, c, alpha, A]

def RE(V, c):
    return rho * c * V / nu


def ll_equation(theta_pts, N_A, non_sorted=False):
    # non_sorted only used for tests
    assert (theta_pts >= 0).all()
    assert (theta_pts <= pi).all()
    if not non_sorted:
        assert (np.diff(theta_pts[:,0]) < 0).all() and np.isclose(theta_pts[-1],0.) and np.isclose(theta_pts[0],pi), \
            "theta_pts must be monotically decreasing from pi to 0"
    else:
        assert (theta_pts>0).all() and (theta_pts<np.pi).all()
    
    Nn = np.arange(1, N_A+1)[:, None]
    Sa = np.sin(theta_pts @ Nn.T)
    Sa_ = (Sa * Nn.T)  /  (np.sin(theta_pts) + 1e-18)
    if not non_sorted:
        Sa_[0, None] = (Nn**2).T # theta = 0
        Sa_[-1, None] = (Nn**2).T # theta = pi
    return Sa, Sa_, np.squeeze(Nn)


def quadrature_weights(N_th):
    scheme = quadpy.line_segment.clenshaw_curtis(N_th)
    w_th = scheme.weights
    pts = scheme.points[:, None]
    theta_pts = np.arccos(pts)
    return w_th, theta_pts

class AbstractLLopt(ABC):
    def __init__(self, W, N_A=5, N_th=10, 
                cd0_model = "flat_plate", 
                cl_model = "flat_plate",
                bounds=None):
        assert N_A <= N_th - 2, \
            f"Insufficient number of spanwise points ({N_th}) to solve lift distribution to {N_A}th order"
        assert 2*int(N_th/2) + 1 == N_th, f"N_th ({N_th}) must be an odd number"
        self.W = W
        self.N_A = int(N_A)
        self.N_th = (N_th)
        # x = [V, b, c, alpha, A]
        self.Nx = int(2 + 2 * N_th + N_A)

        self.w_th, self.theta_pts = quadrature_weights(N_th)
        self.Sa, self.Sa_, self.Nn = ll_equation(self.theta_pts, N_A)

        self.cd0_model = cd0_model
        self.cl_model = cl_model

        self.bounds = None
        self.update_bounds(bounds)

    def optimize(self, x0, alg='COBYLA', options={}):
        pass

    def obj(self, x):
        pass

    def alpha_i(self, b, A):
        return 1/(4*b) * self.Sa_ @ A

    def CL_2D(self, al):
        if self.cl_model=="flat_plate":
            return 2*pi*al # flat plate
        else:
            raise NotImplementedError

    def CD0_2D(self, V, c, al):
        if self.cd0_model=="flat_plate":
            Re = RE(V, c)
            return 0.074/Re**(1/5) # flat plate
        elif self.cd0_model=="constant":
            return 0.01
        else:
            raise NotImplementedError

    def initial_guess(self, dict_var=None):
        pass

    def get_vars(self, x, dic=False):
        pass

    def set_vars(self, dict_var, x=None):
        pass

    def get_plane(self, x):
        pass

    def update_bounds(self, bounds):
        pass

class LiftingLineOpt(AbstractLLopt):
    def __init__(self, W, N_A=5, N_th=10, 
                cd0_model = "flat_plate", 
                cl_model = "flat_plate",
                bounds=None):
        assert N_A <= N_th - 2, \
            f"Insufficient number of spanwise points ({N_th}) to solve lift distribution to {N_A}th order"
        assert 2*int(N_th/2) + 1 == N_th, f"N_th ({N_th}) must be an odd number"
        self.W = W
        self.N_A = int(N_A)
        self.N_th = (N_th)
        # x = [V, b, c, alpha, A]
        self.Nx = int(2 + 2 * N_th + N_A)

        self.w_th, self.theta_pts = quadrature_weights(N_th)
        self.Sa, self.Sa_, self.Nn = ll_equation(self.theta_pts, N_A)

        self.cd0_model = cd0_model
        self.cl_model = cl_model

        self.bounds = None
        self.update_bounds(bounds)

    def optimize(self, x0, alg='COBYLA', options={}):
        opt = {'verbose': 1}
        opt.update(options)
        
        if alg in "COBYLA":
            llcon = {
                'type': 'ineq',
                'fun': lambda x: -self.lifting_line_const(x)
            }
            wcon = {
                'type':'ineq',
                'fun': lambda x: -self.enough_lift_const(x) # positive
            }
            lb = {
                "type":"ineq",
                "fun": lambda x: x - self.bounds.lb # positive
            }
            ub = {
                "type":"ineq",
                "fun": lambda x: self.bounds.ub - x # positive
            }
            res = minimize(self.obj, x0, 
                        method=alg,
                        constraints=[llcon, wcon, lb, ub],
                        options=opt)

            if not np.allclose(self.lifting_line_const(res.x), 0.):
                print("Result does no satisfy lifting line constraint")
        elif alg in "SLSQP":
            llcon = {
                'type': 'eq',
                'fun': self.lifting_line_const 
            }
            wcon = {
                'type':'eq',
                'fun': lambda x: self.enough_lift_const(x)
            }
            lb = {
                "type":"ineq",
                "fun": lambda x: x - self.bounds.lb # positive
            }
            ub = {
                "type":"ineq",
                "fun": lambda x: self.bounds.ub - x # positive
            }
            res = minimize(self.obj, x0, 
                        method=alg,
                        constraints=[llcon, wcon, lb, ub],
                        options=opt)

        elif alg == 'trust-constr':
            llcon = NonlinearConstraint(self.lifting_line_const, 0., 0.)
            wcon = NonlinearConstraint(self.enough_lift_const, 0., 0.)
            res = minimize(self.obj, x0, 
                        method=alg,
                        constraints=[llcon, wcon],
                        options=opt, 
                        bounds=self.bounds)
        else:
            raise NotImplementedError(f"No routine for algorithm {alg}")
        return res

    def obj(self, x):
        V, b, c, al, A, fail = self.get_vars(x)
        L = A[0]
        al_i = self.alpha_i(b, A)
        D = 2 / pi * (self.w_th.T @ (c*self.CD0_2D(V,c, al - al_i))) + \
            1/ (4*b) * (self.Nn * A).T @ A
        return D / L

    def enough_lift_const(self,x):
        # negative
        V, b, c, al, A, fail = self.get_vars(x)
        return 8 * self.W / pi / rho - A[0] * b * V**2

    def lifting_line_const(self, x):
        # equal 0
        V, b, c, al, A, fail = self.get_vars(x)
        al_i = self.alpha_i(b, A)
        return (self.Sa @ A -  c * self.CL_2D(al - al_i))[1:-1]
   
    def alpha_i(self, b, A):
        return 1/(4*b) * self.Sa_ @ A

    def CL_2D(self, al):
        if self.cl_model=="flat_plate":
            return 2*pi*al # flat plate
        else:
            raise NotImplementedError

    def CD0_2D(self, V, c, al):
        if self.cd0_model=="flat_plate":
            Re = RE(V, c)
            return 0.074/Re**(1/5) # flat plate
        elif self.cd0_model=="constant":
            return 0.01
        else:
            raise NotImplementedError

    def initial_guess(self, dict_var=None):
        if dict_var is None:
            dict_var = {}

        # Set to default:
        dflt = {"V":10., "b":3., "c":0.3, "al":0.01, "A":np.random.uniform(0,1., self.N_A)}
        x_dflt = self.set_vars(dflt)[0]

        # If finite bounds are provided, draw randomly inbetween the bounds
        for i,u,l in zip(range(self.Nx), self.bounds.ub, self.bounds.lb):
            if np.isfinite(u) and np.isfinite(l):
                x_dflt[i] = 0.9*(u-l)/2*np.random.random() + (u+l)/2
        return self.set_vars(dict_var, x_dflt)[0]

    def get_vars(self, x, dic=False):
        V = x[0]
        b = x[1]
        c = x[2:2+self.N_th]
        al = x[2+self.N_th:2+2*self.N_th]
        A = x[-self.N_A:]
        fail = 0
        if dic:
            return dict(V=V,b=b,c=c, al=al,A=A)
        else:
            return V, b, c, al, A, fail

    def set_vars(self, dict_var, x=None):
        if x is None:
            x = np.zeros(self.Nx)
        fail = 0
        for k, val in dict_var.items():
            if k == "V":
                x[0] = val
            elif k == "b":
                x[1] = val
            elif k == "c":
                x[2:2+self.N_th] = val
            elif k == "al":
                x[2+self.N_th:2+2*self.N_th] = val
            elif k == "A":
                x[-self.N_A:] = val
            elif k == "fail":
                fail = val
            else:
                raise KeyError(f"dict_var has an invalid key {k}")
        return x, fail

    def get_plane(self, x):
        V, b, c, al, A, fail = self.get_vars(x)
        return Plane(b=b, c=c, w_th=self.w_th, 
                theta_pts=self.theta_pts, 
                twist=al - al[0], N_A=self.N_A)

    def update_bounds(self, bounds):
        if bounds is None:
            bounds = {"ub":{}, "lb":{}}

        if self.bounds is None:
            # setting self.bounds for the first time
            default_ub = np.inf * np.ones(self.Nx)
            default_lb = self.set_vars(
                {"b":0., "V":0., "c":np.zeros(self.N_th)},
                -np.inf * np.ones(self.Nx)
            )[0]
            ub = self.set_vars(bounds["ub"], default_ub)[0]
            lb = self.set_vars(bounds["lb"], default_lb)[0]
            self.bounds = Bounds(lb, ub, keep_feasible=False)
        else:
            self.bounds.ub = self.set_vars(bounds["ub"], self.bounds.ub)[0]
            self.bounds.lb = self.set_vars(bounds["lb"], self.bounds.lb)[0]

class UnconstrainedLLopt(LiftingLineOpt):
    def __init__(self, W, N_A=5, N_th=10, 
                cd0_model = "flat_plate", 
                cl_model = "flat_plate",
                bounds=None):
        assert N_A <= N_th - 2, \
            f"Insufficient number of spanwise points ({N_th}) to solve lift distribution to {N_A}th order"
        assert 2*int(N_th/2) + 1 == N_th, f"N_th ({N_th}) must be an odd number"
        self.W = W
        self.N_A = int(N_A)
        self.N_th = (N_th)
        # x = [V, b, c, alpha]
        self.Nx = int(2 + 2 * N_th + N_A)

        self.w_th, self.theta_pts = quadrature_weights(N_th)
        self.Sa, self.Sa_, self.Nn = ll_equation(self.theta_pts, N_A)

        self.cd0_model = cd0_model
        self.cl_model = cl_model

        self.bounds = None
        self.update_bounds(bounds)

    def optimize(self, x0, alg='IPOPT', options={}):
        opt = {}
        opt.update(options)
        def objfun(xdict):
            x, fail = self.set_vars(xdict)
            funcs={'obj':10000. if fail else self.obj(x)}
            return funcs, fail
        optProb = Optimization('llOpt', objfun)
        ub = self.get_vars(self.bounds.ub, dic=True)
        lb = self.get_vars(self.bounds.lb, dic=True)
        x0 = self.get_vars(x0, dic=True)
        optProb.addVar('V', upper=ub['V'], lower=lb['V'], value=x0['V'])
        optProb.addVar('b', upper=ub['b'], lower=lb['b'], value=x0['b'])
        optProb.addVarGroup('c', self.N_th, upper=ub['c'], lower=lb['c'], value=x0['c'])
        optProb.addVarGroup('al', self.N_th, upper=ub['al'], lower=lb['al'], value=x0['al'])
        optProb.addObj('obj')
        print(optProb)

        if alg== "IPOPT":
            opt = OPT(alg, options=options)
            sol = opt(optProb, sens='FD')
        else:
            raise NotImplementedError(f"No routine for algorithm {alg}")
        D = dict(
        al = [a.value for a in sol.variables['al']],
        c = [a.value for a in sol.variables['c']],
        b = sol.variables['b'][0].value,
        V = sol.variables['V'][0].value
        )
        x = self.set_vars(D)[0]
        return x

    def optimize_scipy(self, x0, alg='COBYLA', options={}):
        opt = {'verbose': 1}
        opt.update(options)
        if alg in "COBYLA":
            lb = {
                "type":"ineq",
                "fun": lambda x: x - self.bounds.lb # positive
            }
            ub = {
                "type":"ineq",
                "fun": lambda x: self.bounds.ub - x # positive
            }
            res = minimize(self.obj, x0, 
                        method=alg,
                        constraints=[lb, ub],
                        options=opt)

            if not np.allclose(self.lifting_line_const(res.x), 0.):
                print("Result does no satisfy lifting line constraint")
        elif alg in "SLSQP":
            lb = {
                "type":"ineq",
                "fun": lambda x: x - self.bounds.lb # positive
            }
            ub = {
                "type":"ineq",
                "fun": lambda x: self.bounds.ub - x # positive
            }
            res = minimize(self.obj, x0, 
                        method=alg,
                        constraints=[lb, ub],
                        options=opt)

        elif alg == 'trust-constr':
            res = minimize(self.obj, x0, 
                        method=alg,
                        options=opt, 
                        bounds=self.bounds)
        else:
            raise NotImplementedError(f"No routine for algorithm {alg}")
        return res

    def get_vars(self, x, dic=False):
        V = x[0]
        b = x[1]
        c = x[2:2+self.N_th]
        al = x[2+self.N_th:2+2*self.N_th]
        
        A = x[-self.N_A:]
        fail = 0
        if b>0 and V>0 and np.isfinite(x).all():
            try:
                A = self.ll(V, b, c, al)
            except np.linalg.LinAlgError:
                fail = 1

        if dic:
            return dict(V=V,b=b,c=c, al=al, A=A, fail=fail)
        else:
            return V, b, c, al, A, fail

    def ll(self, V, b, c, al):
        S = b/2 * self.w_th @ c
        CL = self.W / V**2 / S
        A = np.zeros(self.N_A)
        A[0] = 8 * self.W / pi / rho / (b * V**2)
        if self.cl_model != "flat_plate":
            raise NotImplementedError
        M = (self.Sa + c[:, None] * 2 * pi * 1/(4*b) * self.Sa_)[:,1:]
        r = c * 2 * pi * al
        A[1:] = np.linalg.lstsq(M, r, rcond=None)[0]
        return A

    def set_vars(self, dict_var, x=None):
        x, fail = super().set_vars(dict_var, x=x)
        dic = self.get_vars(x, dic=True) # will compute A
        return super().set_vars(dic)



class Plane:
    def __init__(self, b, c, w_th, theta_pts, twist=None, N_A=None):
        self.N_th = w_th.shape[0]
        self.w_th = w_th
        self.theta_pts = theta_pts
        self.b = b
        self.c = c
        self.twist = twist if twist is None else np.zeros(self.N_th)
        if N_A:
            self.N_A = N_A
            self.Sa, self.Sa_, self.Nn = ll_equation(self.theta_pts, N_A)

    @property
    def area(self):
        return self.b/2 * self.w_th @ self.c

    @property
    def AR(self):
        return self.b**2/self.area