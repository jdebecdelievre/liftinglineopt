from jax.config import config
config.update("jax_enable_x64", True)
from jax import grad, jit, jacfwd
import numpy as np
from jax import numpy as npj
import matplotlib.pyplot as plt
import torch
import quadpy
from abc import ABC, abstractmethod
import cvxpy as cp
from pyoptsparse import Optimization, OPT
from collections import OrderedDict, namedtuple


rho = 1.225
pi = np.pi
nu = 1.81*1e-5

class ArrayItem(object):
    def __init__(self, name):
        self.name = "_" + name
    def __get__(self, instance, owner):
        value = getattr(instance, self.name)
        try:
            if np.atleast_1d(value).shape[0] != instance.size:
                return value * np.ones(instance.size)
            else:
                return value
        except AttributeError:
            raise AttributeError("Size must be set first")
    def __set__(self, instance, value):
        setattr(instance, self.name, value)

class Variable():
    ub = ArrayItem('ub')
    dflt = ArrayItem('dflt')
    lb = ArrayItem('lb')

    def __init__(self, name, size, index, dflt, ub, lb, scale, offset):
        self.name = name
        self.size = size
        self.index = index
        self._dflt = dflt
        self._ub = ub
        self._lb = lb
        self.scale = scale
        self.offset = offset

    def __repr__(self):
        return (f"\n Variable {self.name} \n size {self.size} \n" 
                f"index {self.index} \n default {self._dflt} \n"
                f"up bnd {self._ub} \n low bnd {self._lb} \n"
                f"scale {self.scale} \n offset {self.offset}")

class VariablesDict(dict):
    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError("Invalid key")
        else:
            return super().__setitem__(key, value)

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
                variables,
                W,
                Ny=11, Na=3,
                cd0_model = "flat_plate",
                cd0_val=0.001,
                cl_model = "flat_plate",
                bounds={"lb":{}, "ub":{}}
                ):

        assert 2*int(Ny/2) + 1 == Ny, f"Ny ({Ny}) must be an odd number"
        self.W = W
        self.Ny = (Ny)
        self.Na = (Na)

        self.cd0_model = cd0_model
        self.cd0_val = cd0_val
        self.cl_model = cl_model

        self.variables = variables
        for var in variables:
            setattr(self, var, variables[var])

        for var, bnd in bounds['lb'].items():
            self.variables[var].lb = bnd
        for var, bnd in bounds['ub'].items():
            self.variables[var].ub = bnd
        
        self.Nx = 0
        for key in variables:
            self.Nx += variables[key].size

        self.y_pts = None

        # Differentiate
        self.gradient = jacfwd(self.objective, 0)

    @abstractmethod
    def objective(self, xdict, constraints):
        pass

    def smoothness_penalty_(self, xdict, l):
        # if 'smooth_penalty' in constraints:
        dy = np.diff(np.squeeze(self.y_pts))
        obj = 0
        for key in l:
            obj = obj + \
                l[key] * np.sum((xdict[key][1:] - xdict[key][:-1])**2 / dy)
        return obj

    def addCon(self, optProb, constraint, x0):
        f0 = self.objective(x0, [constraint])
        optProb.addConGroup(constraint["name"], f0[constraint["name"]].shape[0],
        lower=constraint['lb'],
        upper=constraint["ub"]
        )
        return optProb

    def optimize(self, x0=None, alg='IPOPT', restarts=1, 
                    solver_options={}, obj_scale=1.,
                    constraints=[],
                    smooth_penalty={},
                    sens=None,
                    usejit=False):
        options = {}
        options.update(solver_options)
        if x0 is None:
            x0 = self.initial_guess()

        # Define objective
        # Required for jit/grad that it not be a method
        def objective_(xdict, constraints, smooothness_penalty={}):
            obj = self.objective(xdict, constraints)
            obj['obj'] = obj['obj'] + self.smoothness_penalty_(xdict, smooothness_penalty)
            return obj
        gradient = jacfwd(objective_, 0)
        

        # Compile
        if usejit:
            objective_ = jit(objective_, static_argnums=(1,2))
            gradient = jit(gradient, static_argnums=(1,2))
        
        # Add fail flag
        def objfun(xdict): return objective_(xdict, constraints, smooth_penalty), 0
        def sensfun(xdict, funcs): return gradient(xdict, constraints, smooth_penalty), 0
        init = objfun(x0)[0]

        # Create optimization problem
        optProb = Optimization('llOpt', objfun)
        for key, var in self.variables.items():
            optProb.addVarGroup(key, var.size, 
                    upper=var.ub, lower=var.lb,  
                    value=x0[key], scale=var.scale, offset=var.offset)
        optProb.addObj('obj', scale=obj_scale)
        for con in constraints:
            optProb = self.addCon(optProb, con, x0)

        # Run optimizer with various initial conditions
        best = init["obj"]
        best_sol = optProb
        if alg in ["IPOPT", "SNOPT"]:
            opt = OPT(alg, options=options)
            for i in range(restarts):
                # initial condition
                d0 = self.initial_guess()
                for key in d0:
                    for var, val0 in zip(optProb.variables[key], np.atleast_1d(d0[key])):
                        var.value = val0
                    
                # solve
                sol = opt(optProb, sens=sensfun if sens is None else sens)
                if sol.objectives['obj'].value < best:
                    best = sol.objectives['obj'].value
                    best_sol = sol
                print(f"{restarts} - Obj: {sol.objectives['obj'].value} - ObjCalls {sol.userObjCalls}")
        else:
            raise NotImplementedError(f"No routine for algorithm {alg}")

        # Return best solution
        D = best_sol.getDVs()
        print(sol)
        return D, sol

    def initial_guess(self, dict_var=None, dic=True):
        if dict_var is None:
            dict_var = {}

        # Set to default:
        xdict = {key:var.dflt for key, var in self.variables.items()}

        # If finite bounds are provided, draw randomly inbetween the bounds
        for key, var in self.variables.items():
            if np.isfinite(var.ub ).all() and np.isfinite(var.lb).all():
                xdict[key] = 0.9*(var.ub- var.lb)/2 * np.random.random() + (var.ub + var.lb)/2
        
        # update with provided values
        xdict.update(dict_var)

        if dic:
            return xdict
        else:
            return self.set_vars(xdict)

    def get_vars(self, x, dic=True):
        # Vb2 = x[0]
        # c_2b = x[1:1+self.Ny]
        # A = x[-self.Na:]
        if dic:
            xvars = OrderedDict()
            for key, var in self.variables.items():
                xvars[key] = x[var.index[0]:var.index[1]]
            return xvars
        else:
            return (x[var.index[0]:var.index[1]] for key, var in self.variables.items())

    def set_vars(self, dict_var, x=None):
        if x is None:
            x = np.zeros(self.Nx)
        fail = 0
        for k, var in self.variables.items():
            if k in dict_var:
                x[var.index[0]:var.index[1]] = dict_var[k]
        return x


# x = [(Vb)^2, c/2b, A]
#      1,      Ny,  Na 
class LiftDistVb(BaseLiftDist):
    def __init__(self, W, Ny=11, Na=3,
                cd0_model = "flat_plate",
                cd0_val=0.001,
                cl_model = "flat_plate",
                bounds={"lb":{}, "ub":{}},
                ):

        variables = VariablesDict({
            "Vb2":Variable(
                        name = "Vb2",
                        size = 1,
                        dflt = 0.1,
                        ub = 50.,
                        lb = 5.,
                        index = (0,1),
                        scale = 0.1,
                        offset = 0.
                    ),
            "c_2b" : Variable(
                        name = "c_b",
                        size = Ny,
                        dflt = 0.1,
                        ub = 1.,
                        lb = 0.,
                        index = (1,Ny+1),
                        scale = 1.,
                        offset = 0.
                    ),
            "A" :  Variable(
                        name = "A",
                        size = Na,
                        dflt = 0.1,
                        ub = .1,
                        lb = -.1,
                        index = (Ny + 1, Na + Ny + 1),
                        scale = 100.,
                        offset = 0.
                    )
        })

        super().__init__(
                variables = variables,
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
   
    def metrics(self, xdict):
        A1 = self.A1(xdict["Vb2"])
        _AR = self._AR(xdict["c_2b"])
        AR = 1 / _AR 
        Re = self.Re(xdict["Vb2"], xdict["c_2b"])
        cd0 = self.CD0_2D(xdict["Vb2"], xdict["c_2b"], xdict["A"], A1)
        CD0 = self.w_th @ (cd0 * xdict["c_2b"]) / _AR
        CDi = self.CDi(A1, xdict["A"], _AR)
        cl = self.cl(A1, xdict["A"], xdict["c_2b"])
        CL = self.w_th[1:-1] @ (cl * xdict["c_2b"])[1:-1] / _AR
        L_D = CL / (CD0 + CDi)
        return A1, AR, Re, cd0, CD0, cl, CL, CDi, L_D

    def objective(self, xdict, constraints=[]):
        A1 = self.A1(xdict["Vb2"])
        CDi_AR = pi * (A1**2 + self.Nn @ xdict["A"]**2)
        cd0 = self.CD0_2D(xdict["Vb2"], xdict["c_2b"], xdict["A"], A1)
        funcs = {"obj":
                    xdict["Vb2"] * (CDi_AR + self.w_th @ (cd0 * xdict["c_2b"]))
                }
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
                bounds={"lb":{}, "ub":{}},
                ):

        variables = VariablesDict({
            "Cw_AR":Variable(
                        name = "Cw_AR",
                        size = 1,
                        dflt = 0.1,
                        ub = 1.,
                        lb = 0.,
                        index = (0,1),
                        scale = 1.,
                        offset = 0.
                    ),
            "c_b" : Variable(
                        name = "c_b",
                        size = Ny,
                        dflt = 0.1,
                        ub = 1.,
                        lb = 0.,
                        index = (1,Ny+1),
                        scale = 1.,
                        offset = 0.
                    ),
            "A" :  Variable(
                        name = "A",
                        size = Na,
                        dflt = 0.,
                        ub = .1,
                        lb = -.1,
                        index = (Ny + 1, Na + Ny + 1),
                        scale = 1000,
                        offset = 0.
                    )
        })

        super().__init__(
                variables = variables,
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
        re = self.Re(xdict["Cw_AR"], xdict["c_b"])
        cd0 = self.CD0_2D(xdict["Cw_AR"], xdict["c_b"], xdict["A"], A1)
        CD0 = self.w_th @ (cd0 * xdict["c_b"]) / _AR
        CDi = self.CDi(A1, xdict["A"], _AR)
        cl = self.cl(A1, xdict["A"], xdict["c_b"])
        CL = self.w_th @ (cl * xdict["c_b"]) / _AR
        L_D = CL / (CD0 + CDi)
        Re = self.Re(xdict["Cw_AR"], _AR)
        return A1, AR, re, Re, cd0, CD0, cl, CL, CDi, L_D

    def objective(self, xdict, constraints=[]):
        A1 = self.A1(xdict["Cw_AR"])
        CDi_AR = pi * (A1**2 + self.Nn @ xdict["A"]**2)
        cd0 = self.CD0_2D(xdict["Cw_AR"], xdict["c_b"], xdict["A"], A1)
        funcs = {"obj":
                    (CDi_AR + self.w_th @ (cd0 * xdict["c_b"]))/xdict["Cw_AR"]
                }
        for con in constraints:
            if con['name'] == "clCon":
                funcs.update({
                    # "clCon": self.cl(A1, xdict["A"], xdict["c_b"])
                    "clCon_ub":  4 * (self.M1 * A1 + self.M @ xdict["A"]) - con['ub'] * xdict["c_b"],
                    "clCon_lb": -4 * (self.M1 * A1 + self.M @ xdict["A"]) + con['lb'] * xdict["c_b"],
                })
            elif con['name'] == "ReCon":
                funcs.update({
                    "ReCon": (self.Re(xdict["Cw_AR"], xdict["c_b"]) - con['lb'])/(con['ub'] - con['lb'])
                })
            else:
                raise ValueError(f"{con['name']} is not a valid constraint")
        return funcs

    def addCon(self, optProb, constraint, x0):
        x0['Cw_AR'] = np.atleast_1d(x0['Cw_AR'])
        f0 = self.objective(x0, [constraint])
        if constraint['name'] == 'clCon':

            jac=self.gradient(x0, [constraint])['clCon_ub']
            jac['c_b'] = {'coo':[np.arange(self.Ny), np.arange(self.Ny), np.diag(jac['c_b'])],
                        'shape':[self.Ny, self.Ny]}
            optProb.addConGroup('clCon'+'_ub', f0['clCon_ub'].shape[0],
                                upper=0.,
                                linear=True,
                                jac=jac,
                                )

            jac=self.gradient(x0, [constraint])['clCon_lb']
            jac['c_b'] = {'coo':[np.arange(self.Ny), np.arange(self.Ny), np.diag(jac['c_b'])],
                        'shape':[self.Ny, self.Ny]}
            optProb.addConGroup('clCon_lb', f0['clCon_lb'].shape[0],
                                upper=0.,
                                linear=True,
                                jac=jac
                                )
        elif constraint['name'] == "ReCon":
            jac=self.gradient(x0, [constraint])['ReCon']
            jac['A'] = {'coo':[], 'shape':[self.Na, self.Na]}
            jac['c_b'] = {'coo':[np.arange(self.Ny), np.arange(self.Ny), np.diag(jac['c_b'])],
                        'shape':[self.Ny, self.Ny]}
            optProb.addConGroup('ReCon', f0['ReCon'].shape[0],
            upper=1., lower=0.)
        else:
            return super().addCon(optProb, constraint, x0)
        return optProb

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

class GPLiftDistFourrier(LiftDistVb):   

    def optimize(self, x0, alg=None, options={}):
        opt = {}
        opt.update(options)
        A0 = x0['A']
        c_2b0 = x0['c_2b']
        Vb20 = x0['Vb2']
        d = dict(
            Vb2 = cp.Variable(pos=True, value=np.squeeze(Vb20)),
            A = cp.Variable(pos=True, value=A0, shape=self.Na),
            c_2b = cp.Variable(pos=True, value=c_2b0, shape=self.Ny)
        )
        obj = cp.Minimize(self.obj(d))
        con = []
        for key in self.variables:
            ub = self.variables[key].ub
            lb = self.variables[key].lb
            if np.isfinite(ub).any():
                con.append(d[key]/ub <= 1. * np.ones_like(ub))
            if np.isfinite(lb).all() and (np.sign(lb) > 0).all():
                con.append(lb/d[key] <= 1. * np.ones_like(lb) )
        prob = cp.Problem(obj, con)
        prob.solve(gp=True, **opt)
        if prob.status == "unbounded":
            x = {}
            raise ValueError("Problem is unbounded")
        else:
            x= {key: d[key].value for key in self.variables}
        return x, prob

   
    def obj(self, d):
        _AR = self._AR(d["c_2b"])
        A1 = self.A1(d["Vb2"])
        CDi_AR = pi * (A1**2 + cp.multiply(self.Nn, d["A"]) @ d["A"])
        cd0 = self.CD0_2D(d["Vb2"], d["c_2b"], d["A"], A1)
        return d["Vb2"] * (CDi_AR + self.w_th @ cp.multiply(cd0, d["c_2b"]))
