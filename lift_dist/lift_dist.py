import numpy as np
import matplotlib.pyplot as plt
import torch
import quadpy
from abc import ABC, abstractmethod
import cvxpy as cp
from pyoptsparse import Optimization, OPT
from collections import OrderedDict

from jax import grad, jit, jacfwd

rho = 1.225
pi = np.pi
nu = 1.81*1e-5

# x = [(Vb)^2, c/2b, A]
#      1,      Ny,  Na 

def quadrature_weights(Ny):
    scheme = quadpy.line_segment.clenshaw_curtis(Ny)
    w_th = scheme.weights
    y_pts = scheme.points[:, None]
    theta_pts = np.arccos(y_pts)
    return w_th, theta_pts,y_pts

def even_sine_exp(theta_pts, Na, non_sorted=False):
    # non_sorted only used for tests
    assert (theta_pts >= 0).all()
    assert (theta_pts <= pi).all()
    if not non_sorted:
        assert (np.diff(theta_pts[:,0]) < 0).all() and np.isclose(theta_pts[-1],0.) and np.isclose(theta_pts[0],pi), \
            "theta_pts must be monotically decreasing from pi to 0"
    else:
        assert (theta_pts>0).all() and (theta_pts<np.pi).all()
    
    Nn = 2*np.arange(1, Na+1)[:, None] + 1
    M = np.sin(theta_pts @ Nn.T)
    M1 = np.sin(theta_pts)[:, 0]
    M_ = (M * Nn.T)  /  (np.sin(theta_pts) + 1e-18)
    M1_ = M1 / (np.sin(theta_pts) + 1e-18)[:, 0]
    if not non_sorted:
        M_[0, None] = (Nn**2).T # theta = 0
        M1_[0] = 1
        M_[-1, None] = (Nn**2).T # theta = pi
        M1_[-1] = 1
    return M, M1, M_, M1_, np.squeeze(Nn)

class BaseLiftDist():
    def __init__(self,
                varNames,
                varSize,
                varDflt,
                default_ub,
                default_lb,
                W,
                Ny=11, Na=3,
                cd0_model = "flat_plate",
                cd0_val=0.001,
                cl_model = "flat_plate",
                bounds=None):

        assert 2*int(Ny/2) + 1 == Ny, f"Ny ({Ny}) must be an odd number"
        self.W = W
        self.Ny = (Ny)
        self.Na = (Na)

        self.cd0_model = cd0_model
        self.cd0_val = cd0_val
        self.cl_model = cl_model

        self.varNames = varNames
        self.default_lb = default_lb
        self.default_ub = default_ub
        self.varIndex = OrderedDict()
        self.varSize  = OrderedDict() 
        self.varDflt  = OrderedDict() 
        
        index = 0
        self.Nx = 0
        for key in varNames:
            self.varSize[key]  = varSize[key]
            self.Nx += varSize[key]
            self.varDflt[key]  = varDflt[key]
            self.varIndex[key] = (index, index + self.varSize[key])
            index += self.varSize[key]

        self.ub, self.lb = None, None
        self.update_bounds(bounds)

    @abstractmethod
    def objective(self, xdict, constraints):
        pass

    def optimize(self, x0=None, alg='IPOPT', restarts=1, 
                    solver_options={}, obj_scale=1.,
                    usejit=True,
                    smooth_penalty={},
                    constraints={}):
        options = {}
        options.update(solver_options)
        if x0 is None:
            x0 = self.initial_guess()
        x0 = self.get_vars(x0, dic=True)

        # Define objective
        def objective(xdict):
            obj = self.objective(xdict, constraints)
            for key in smooth_penalty:
                obj["obj"] = obj["obj"] + (smooth_penalty[key] / self.w_th[1:]) @ ((xdict[key][1:] - xdict[key][:-1])**2)
            return obj

        # Precompile and differentiate
        gradient = jacfwd(objective)
        if usejit:
            objective = jit(objective)
            gradient = jit(gradient)

        # Add fail flag
        def objfun(xdict): return objective(xdict), 0
        def sensfun(xdict, funcs): return gradient(xdict), 0
        init = objfun(x0)[0]

        # Create optimization problem
        optProb = Optimization('llOpt', objfun)
        ub = self.default_ub
        lb = self.default_lb
        for key in self.varNames:
            optProb.addVarGroup(key, self.varSize[key], 
                    upper=ub[key], lower=lb[key],  
                    value=x0[key])
        optProb.addObj('obj', scale=obj_scale)
        for con in constraints:
            optProb.addConGroup(con, init[con].shape[0],
                        lower=constraints[con][0],
                        upper=constraints[con][1]
                        )
        
        # Run optimizer with various initial conditions
        best = init["obj"]
        best_sol = optProb
        if alg in ["IPOPT", "SNOPT"]:
            opt = OPT(alg, options=options)
            for i in range(restarts):
                # initial condition
                d0 = self.get_vars(self.initial_guess(), dic=True)
                for key in d0:
                    for var, val0 in zip(optProb.variables[key], d0[key]):
                        var.value = val0
                    
                # solve
                sol = opt(optProb, sens='fd')
                if sol.objectives['obj'].value < best:
                    best = sol.objectives['obj'].value
                    best_sol = sol
                print(f"{restarts} - Obj: {sol.objectives['obj'].value} - ObjCalls {sol.userObjCalls}")
        else:
            raise NotImplementedError(f"No routine for algorithm {alg}")

        # Return best solution
        D = best_sol.getDVs()
        x = self.set_vars(D)
        print(sol)
        return x, sol

    def initial_guess(self, dict_var=None):
        if dict_var is None:
            dict_var = {}

        # Set to default:
        x_dflt = self.varDflt.copy()

        # If finite bounds are provided, draw randomly inbetween the bounds
        for key in self.varNames:
            if np.isfinite(self.ub[key]).all() and np.isfinite(self.lb[key]).all():
                x_dflt[key] = 0.9*(self.ub[key]-self.lb[key])/2 * np.random.random() + (self.ub[key]+self.lb[key])/2
        
        # update with provided values
        x_dflt.update(dict_var)

        return self.set_vars(x_dflt)

    def get_vars(self, x, dic=False):
        # Vb2 = x[0]
        # c_2b = x[1:1+self.Ny]
        # A = x[-self.Na:]
        if dic:
            xvars = OrderedDict()
            for key, idx in self.varIndex.items():
                xvars[key] = x[idx[0]:idx[1]]
            return xvars
        else:
            return (x[idx[0]:idx[1]] for key, idx in self.varIndex.items())

    def set_vars(self, dict_var, x=None):
        if x is None:
            x = np.zeros(self.Nx)
        fail = 0
        for k, val in self.varIndex.items():
            if k in dict_var:
                x[val[0]:val[1]] = dict_var[k]
        return x

    def update_bounds(self, bounds):
        if bounds is None:
            bounds = {"ub":{}, "lb":{}}

        if self.ub is None:
            # setting self.bounds for the first time
            self.ub = self.default_ub
            self.lb = self.default_lb
        self.ub.update(bounds["ub"])
        self.lb.update(bounds["lb"])


# x = [(Vb)^2, c/2b, A]
#      1,      Ny,  Na 
class LiftDistVb(BaseLiftDist):
    def __init__(self, W, Ny=11, Na=3,
                cd0_model = "flat_plate",
                cd0_val=0.001,
                cl_model = "flat_plate",
                bounds=None):

        varNames = ["Vb2", "c_2b", "A"]
        varSize = {"Vb2":1, "c_2b":Ny, "A":Na}
        varDflt = {"Vb2":18., "c_2b":0.15, "A":np.random.uniform(0,1., Na)}
        default_ub = {"Vb2":np.inf, "c_2b": np.ones(Ny),
                "A": 10 * np.ones(Na)}
        default_lb = {"Vb2":0., "c_2b":np.zeros(Ny),
                "A":-10. * np.ones(Na)}

        super().__init__(
                varNames = varNames,
                varSize = varSize,
                varDflt = varDflt,
                default_ub=default_ub,
                default_lb=default_lb,
                W = W,
                Ny = Ny, Na = Na,
                cd0_model = cd0_model,
                cd0_val= cd0_val,
                cl_model = cl_model,
                bounds = bounds)

        self.w_th, self.theta_pts, self.y_pts = quadrature_weights(Ny)
        self.M, self.M1, self.M_, self.M1_, self.Nn = even_sine_exp(self.theta_pts, Na)
    
    def A1(self, Vb2):
        return 2 * self.W / (rho*pi) / Vb2 

    def CDi(self, A1, A, _AR):
        return pi / _AR * (A1**2 + self.Nn @ A**2)

    def cl(self, A1, A, c_2b):
        return 2 * (self.M1 * A1 + self.M @ A) / c_2b

    def Re(self, Vb2, c_2b):
        return 2 * rho / nu * Vb2**(1/2) * c_2b

    def _AR(self, c_2b):
        return self.w_th @ c_2b
   
    def objective(self, xdict, constraints={}):
        A1 = self.A1(xdict["Vb2"])
        CDi_AR = pi * (A1**2 + self.Nn @ xdict["A"]**2)
        cd0 = self.CD0_2D(xdict["Vb2"], xdict["c_2b"], xdict["A"], A1)
        funcs = {"obj":xdict["Vb2"] * (CDi_AR + self.w_th @ (cd0 * xdict["c_2b"]))}
        for key in constraints:
            if key == "clCon":
                funcs.update({
                    "clCon": self.cl(A1, xdict["A"], xdict["c_2b"])
                })
            elif key == "ReCon":
                funcs.update({
                    "ReCon": self.Re(xdict["Vb2"], xdict["c_2b"])
                })
            else:
                raise ValueError(f"{key} is not a valid constraint")
        return funcs

    def alpha_i(self, A1, A):
        return self.M_ @ A + self.M1_ * A1

    def CD0_2D(self, Vb2, c_2b, A, A1):
        # cl = self.cl(A1, A, c_2b)
        if self.cd0_model=="flat_plate":
            Re = self.Re(Vb2, c_2b)
            return 0.074/Re**(1/5) # flat plate
        elif self.cd0_model == "constant":
            return self.cd0_val
        else:
            raise NotImplementedError


# x = [CW/AR,   c/b,     A]
#        1,   (Ny+1)/2,  Na 
class LiftDistND(BaseLiftDist):
    def __init__(self, W, Ny=11, Na=3,
                cd0_model = "flat_plate",
                cd0_val=0.001,
                cl_model = "flat_plate",
                bounds=None):

        varNames = ["Cw_AR", "c_b", "A"]
        varSize = {"Cw_AR":1, "c_b":Ny, "A":Na}
        varDflt = {"Cw_AR":0.1, "c_b":0.1, "A":0.1}
        default_ub = {"Cw_AR":1., "c_b": np.ones(Ny),
                "A": 10 * np.ones(Na)}
        default_lb = {"Cw_AR":0., "c_b":np.zeros(Ny),
                "A":-10. * np.ones(Na)}

        super().__init__(
                varNames = varNames,
                varSize = varSize,
                varDflt = varDflt,
                default_ub=default_ub,
                default_lb=default_lb,
                W = W,
                Ny = Ny, Na = Na,
                cd0_model = cd0_model,
                cd0_val= cd0_val,
                cl_model = cl_model,
                bounds = bounds)

        self.w_th, self.theta_pts, self.y_pts = quadrature_weights(2*Ny - 1)
        self.M, self.M1, self.M_, self.M1_, self.Nn = even_sine_exp(self.theta_pts, Na)

        # Only consider half of the chord
        self.w_th = self.w_th[:Ny]
        self.w_th[-1] /= 2
        self.theta_pts = self.theta_pts[:Ny]
        self.y_pts = self.y_pts[:Ny]
        self.M = self.M[:Ny]
        self.M_ = self.M_[:Ny]
        self.M1 = self.M1[:Ny]
        self.M1_ = self.M1_[:Ny]

    def A1(self, Cw_AR):
        return Cw_AR / pi

    def CDi(self, A1, A, _AR):
        return pi / _AR * (A1**2 + self.Nn @ A**2)

    def cl(self, A1, A, c_b):
        return 4 * (self.M1 * A1 + self.M @ A) / c_b

    def Re(self, Cw_AR, c_b):
        return (1 / 2 * rho / self.W * Cw_AR)**(-0.5) * rho/ nu * c_b 

    def _AR(self, c_b):
        return self.w_th @ c_b

    def metrics(self, xdict):
        A1 = self.A1(xdict["Cw_AR"])
        _AR = self._AR(xdict["c_b"])
        AR = 1 / _AR 
        Re = self.Re(xdict["Cw_AR"], xdict["c_b"])
        cd0 = self.CD0_2D(xdict["Cw_AR"], xdict["c_b"], xdict["A"], A1)
        CD0 = self.w_th @ (cd0 * xdict["c_b"]) / _AR
        CDi = self.CDi(A1, xdict["A"], _AR)
        cl = self.cl(A1, xdict["A"], xdict["c_b"])
        CL = self.w_th @ (cl * xdict["c_b"]) / _AR
        L_D = CL / (CD0 + CDi)
        return A1, AR, Re, cd0, CD0, cl, CL, CDi, L_D

    def objective(self, xdict, constraints={}):
        A1 = self.A1(xdict["Cw_AR"])
        CDi_AR = pi * (A1**2 + self.Nn @ xdict["A"]**2)
        cd0 = self.CD0_2D(xdict["Cw_AR"], xdict["c_b"], xdict["A"], A1)
        funcs = {"obj":(CDi_AR + self.w_th @ (cd0 * xdict["c_b"]))/xdict["Cw_AR"]}
        for key in constraints:
            if key == "clCon":
                funcs.update({
                    "clCon": self.cl(A1, xdict["A"], xdict["c_b"])
                })
            elif key == "ReCon":
                funcs.update({
                    "ReCon": self.Re(xdict["Cw_AR"], xdict["c_b"])
                })
            else:
                raise ValueError(f"{key} is not a valid constraint")
        return funcs

    def alpha_i(self, A1, A):
        return self.M_ @ A + self.M1_ * A1

    def CD0_2D(self, Cw_AR, c_b, A, A1):
        # cl = self.cl(A1, A, c_2b)
        if self.cd0_model=="flat_plate":
            Re = self.Re(Cw_AR, c_b)
            return 0.074/Re**(1/5) # flat plate
        elif self.cd0_model == "constant":
            return self.cd0_val
        elif self.cd0_model == 'function':
            Re = self.Re(Cw_AR, c_b)
            cl = self.cl(A1, A, c_b)
            return self.cd0_val(cl, Re)
        else:
            raise NotImplementedError

class GPLiftDist(LiftDistVb):   

    def optimize(self, x0, alg=None, options={}):
        opt = {}
        opt.update(options)
        Vb20, c_2b0, A0 = self.get_vars(x0)
        d = dict(
            Vb2 = cp.Variable(pos=True, value=np.squeeze(Vb20)),
            A = cp.Variable(pos=True, value=A0, shape=self.Na),
            c_2b = cp.Variable(pos=True, value=c_2b0, shape=self.Ny)
        )
        obj = cp.Minimize(self.obj(d))
        con = []
        for key in self.varNames:
            if np.isfinite(self.ub[key]).any():
                con.append(d[key]/self.ub[key] <= 1.)
            if np.isfinite(self.lb[key]).all() and (np.sign(self.lb[key]) > 0).all():
                con.append(self.lb[key]/d[key] <= 1.)
        prob = cp.Problem(obj, con)
        prob.solve(gp=True, **opt)
        if prob.status == "unbounded":
            x = self.set_vars({})
            raise ValueError("Problem is unbounded")
        else:
            x = self.set_vars(
                {key: d[key].value for key in self.varNames})
        return x, prob

   
    def obj(self, d):
        _AR = self._AR(d["c_2b"])
        A1 = self.A1(d["Vb2"])
        CDi_AR = pi * (A1**2 + cp.multiply(self.Nn, d["A"]) @ d["A"])
        cd0 = self.CD0_2D(d["Vb2"], d["c_2b"], d["A"], A1)
        return d["Vb2"] * (CDi_AR + self.w_th @ cp.multiply(cd0, d["c_2b"]))