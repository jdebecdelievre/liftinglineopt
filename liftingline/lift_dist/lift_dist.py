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
import os
import warnings
from copy import deepcopy

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

# def quadrature_weights(Ny):
#     scheme = quadpy.line_segment.clenshaw_curtis(Ny)
#     w_th = scheme.weights
#     y_pts = scheme.points[:, None]
#     theta_pts = np.arccos(y_pts)
#     return w_th, theta_pts,y_pts

def quadrature_weights(Ny, halfspan=False):
    if halfspan:
        w_th, theta_pts, y_pts = quadrature_weights((2*Ny - 1), halfspan=False)
        w_th = w_th[:Ny]
        w_th[-1] /= 2
        theta_pts = theta_pts[:Ny]
        y_pts = y_pts[:Ny]
    else:
        scheme = quadpy.line_segment.clenshaw_curtis(Ny)
        w_th = scheme.weights
        y_pts = scheme.points[:, None]
        theta_pts = np.arccos(y_pts)
    return w_th, theta_pts,y_pts

def even_sine_exp(theta_pts, Na, halfspan=False, non_sorted=False):
    # non_sorted only used for tests

    if halfspan:
        Ny = theta_pts.shape[0]
        theta_pts = np.vstack(( theta_pts, pi-np.flip(theta_pts[:-1])))
        M, M1, M_, M1_, Nn = even_sine_exp(theta_pts, Na, halfspan=False)
        M = M[:Ny]
        M_ = M_[:Ny]
        M1 = M1[:Ny]
        M1_ = M1_[:Ny]
    else:
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

        # assert 2*int(Ny/2) + 1 == Ny, f"Ny ({Ny}) must be an odd number"
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
        if constraint["name"] in f0:
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
        options = {"warm_start_init_point":'yes', 'linear_solver':'ma97'}
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
        
        if alg in ["IPOPT"]:
            # First Run
            opt = OPT(alg, options=options)
            best_sol = opt(optProb, sens=sensfun if sens is None else sens)
            best = objfun(best_sol.getDVs())[0]['obj']
            best_feasibility = self.is_feasible(best_sol)
            print(f"{x0} - Obj: {best_sol.objectives['obj'].value} - ObjCalls {best_sol.userObjCalls} - Feasible: {best_feasibility}")

            # If restarts
            filename = options['output_file']
            for i in range(1, restarts):
                # initial condition
                x0 = self.initial_guess()
                for key in x0:
                    for var, val0 in zip(optProb.variables[key], np.atleast_1d(x0[key])):
                        var.value = val0
                    
                # temporary output file
                options['output_file'] = filename + f'_{i}'

                # solve
                opt = OPT(alg, options=options)
                sol = opt(optProb, sens=sensfun if sens is None else sens)
                feasible = self.is_feasible(sol)
                print(f"{x0} - Obj: {sol.objectives['obj'].value} - ObjCalls {sol.userObjCalls} - Feasible: {feasible}")
                if sol.objectives['obj'].value < best or feasible > best_feasibility:
                    best = sol.objectives['obj'].value
                    best_sol = sol
                    best_feasibility = True
                    os.rename(options['output_file'], filename)
                else:
                    os.remove(options['output_file'])     
        else:
            raise NotImplementedError(f"No routine for algorithm {alg}")

        # Return best solution
        with open(filename, 'a') as f:
            print(best_sol, file=f)

        D = best_sol.getDVs()
        if not best_feasibility:
            D = {d:val * np.nan for d, val in D.items()}
        print(best_sol)
        return D, best_sol

    def is_feasible(self, sol):
        TOL = 1e-5
        feasible = True
        for c, con in sol.constraints.items():
            con_ = sol.constraints[c].twoSidedConstraints
            feasible = feasible and \
                    (con.value <= np.array(con_['upper']) + TOL).all() and \
                        (con.value >= np.array(con_['lower']) - TOL).all()
        return feasible

    def initial_guess(self, dict_var=None, dic=True):
        if dict_var is None:
            dict_var = {}

        # Set to default:
        xdict = {key:var.dflt for key, var in self.variables.items()}

        # If finite bounds are provided, draw randomly inbetween the bounds
        for key, var in self.variables.items():
            if np.isfinite(var.ub ).all() and np.isfinite(var.lb).all():
                xdict[key] = 0.99*(var.ub- var.lb) * np.random.random() + var.lb

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
        # self.M1 = cp.Parameter(nonneg=True, value=self.M1, shape=self.M1.shape)
    
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
        return A1, AR, Re, cd0, CD0, cl, CL, CDi, L_D, self.W

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
                warnings.warn(f"{key} is not a valid constraint")
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
                        ub = .01,
                        lb = -.01,
                        index = (Ny + 1, Na + Ny + 1),
                        scale = 1,
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

        self.w_th, self.theta_pts, self.y_pts = quadrature_weights(Ny, halfspan=True)
        self.M, self.M1, self.M_, self.M1_, self.Nn = even_sine_exp(self.theta_pts, Na, halfspan=True)

    def A1(self, Cw_AR):
        return Cw_AR / pi

    def CDi(self, A1, A, _AR):
        return pi / _AR * A1**2 * (1  + self.Nn @ A**2)

    def cl(self, A1, A, c_b):
        return 4 * A1 * (self.M1 + self.M @ A) / c_b

    def Re(self, Cw_AR, c_b):
        return (1 / 2 * rho / self.W * Cw_AR)**(-0.5) * rho/ nu * c_b 

    def _AR(self, c_b):
        return self.w_th @ c_b

    def metrics(self, xdict):
        c_b = xdict["c_b"]
        A1 = self.A1(xdict["Cw_AR"])
        _AR = self._AR(xdict["c_b"])
        AR = 1 / _AR 
        c = c_b * AR

        re = self.Re(xdict["Cw_AR"], xdict["c_b"])
        Re = self.Re(xdict["Cw_AR"], _AR)

        cl = self.cl(A1, xdict["A"], xdict["c_b"])
        CL = self.w_th @ (cl * xdict["c_b"]) / _AR

        cdp = self.CD0_2D(xdict["Cw_AR"], xdict["c_b"], xdict["A"], A1)
        CDp = self.w_th @ (cdp * xdict["c_b"]) / _AR

        alpha_i = self.alpha_i(A1, xdict["A"])
        CDi = self.CDi(A1, xdict["A"], _AR)

        L_D = CL / (CDp + CDi)

        return c, AR, re, Re, cl, CL, cdp, CDp, alpha_i, CDi, L_D, self.W

    def objective(self, xdict, constraints=[]):
        A1 = self.A1(xdict["Cw_AR"])
        CDi_AR = pi * A1**2 * (1 + self.Nn @ xdict["A"]**2)
        cd0 = self.CD0_2D(xdict["Cw_AR"], xdict["c_b"], xdict["A"], A1)
        funcs = {"obj":
                    (CDi_AR + self.w_th @ (cd0 * xdict["c_b"]))/xdict["Cw_AR"]
                }
        for con in constraints:
            if con['name'] == "clCon":
                funcs.update({
                    # "clCon": self.cl(A1, xdict["A"], xdict["c_b"])
                    "clCon_ub":  4 * A1 * (self.M1 + self.M @ xdict["A"]) - con['ub'] * xdict["c_b"],
                    "clCon_lb": -4 * A1 * (self.M1 + self.M @ xdict["A"]) + con['lb'] * xdict["c_b"],
                })
            elif con['name'] == "ReCon":
                funcs.update({
                    "ReCon": (self.Re(xdict["Cw_AR"], xdict["c_b"]) - con['lb'])/(con['ub'] - con['lb'])
                })
            elif con['name'] == "CLCon":
                _AR = self._AR(xdict["c_b"])
                if 'lb' in con:
                    if 'ub' in con:
                        raise NotImplementedError("CLCon with ub and lb not implemented")
                    funcs.update({
                        "CLCon": xdict["Cw_AR"] - con['lb'] * _AR
                    })
                elif 'ub' in con:
                    if 'lb' in con:
                        raise NotImplementedError("CLCon with ub and lb not implemented")
                    funcs.update({
                        "CLCon": -xdict["Cw_AR"] + con['ub'] * _AR
                    })
            elif con['name'] == "chordVar":
                funcs.update({
                    "chordVar": xdict['c_b'][1:] - xdict["c_b"][:-1]
                })
            else:
                warnings.warn(f"{con['name']} is not a valid constraint")
        return funcs

    def addCon(self, optProb, constraint, x0):
        x0['Cw_AR'] = np.atleast_1d(x0['Cw_AR'])
        n_cb = x0['c_b'].shape[-1]
        f0 = self.objective(x0, [constraint])
        if constraint['name'] == 'clCon':

            jac=self.gradient(x0, [constraint])['clCon_ub']
            if n_cb == self.Ny:
                jac['c_b'] = {'coo':[np.arange(self.Ny), np.arange(self.Ny), np.diag(jac['c_b'])],
                            'shape':[self.Ny, self.Ny]}
            optProb.addConGroup('clCon'+'_ub', f0['clCon_ub'].shape[0],
                                upper=0.,
                                linear=True,
                                jac=jac,
                                )

            jac=self.gradient(x0, [constraint])['clCon_lb']
            if n_cb == self.Ny:
                jac['c_b'] = {'coo':[np.arange(self.Ny), np.arange(self.Ny), np.diag(jac['c_b'])],
                            'shape':[self.Ny, self.Ny]}
            optProb.addConGroup('clCon_lb', f0['clCon_lb'].shape[0],
                                upper=0.,
                                linear=True,
                                jac=jac
                                )
        elif constraint['name'] == "ReCon":
            # jac=self.gradient(x0, [constraint])['ReCon']
            # jac['c_b'] = {'coo':[np.arange(self.Ny), np.arange(self.Ny), np.ones(self.Ny)],
            #             'shape':[self.Ny, self.Ny]}
            # del jac['A']
            optProb.addConGroup('ReCon', f0['ReCon'].shape[0],
            upper=1., 
            lower=0., 
            wrt=['Cw_AR', 'c_b'], 
            # jac=jac
            )

        elif constraint['name'] == 'CLCon':
            jac=self.gradient(x0, [constraint])['CLCon']
            del jac['A']
            optProb.addConGroup('CLCon', f0['CLCon'].shape[0],
                                lower=0.,
                                linear=True,
                                jac=jac,
                                wrt=['Cw_AR', 'c_b']
                                )
        elif constraint['name'] == 'chordVar':
            jac=self.gradient(x0, [constraint])['chordVar']
            if "A" in jac:
                del jac['A']
            del jac['Cw_AR']
            optProb.addConGroup('chordVar', f0['chordVar'].shape[0],
                                lower=constraint['lb'],
                                upper=constraint['ub'],
                                linear=True,
                                jac=jac,
                                wrt=['c_b']
                                )
        else:
            return super().addCon(optProb, constraint, x0)
        return optProb

    def alpha_i(self, A1, A):
        return A1 * (self.M_ @ A + self.M1_)

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

class SpecifiedLiftDist(LiftDistND):
    def __init__(self, W, Ny=11, Na=5,
                cd0_model = "flat_plate",
                cd0_val=0.001,
                bounds={"lb":{}, "ub":{}},
                A = None
                ):
        super().__init__(W, Ny=Ny, Na=Na,
                cd0_model = cd0_model,
                cd0_val=cd0_val,
                bounds=bounds
                )
        del self.variables['A']

        if A is None:
            self.A = np.zeros(Na)
        else:
            assert A.shape[0] == Na
            self.A = A

    def objective(self, xdict, constraints=[]):
        A1 = self.A1(xdict["Cw_AR"])
        cd0 = self.CD0_2D(xdict["Cw_AR"], xdict["c_b"], self.A, A1)
        xdict["A"] = self.A
        funcs = super().objective(xdict, constraints=constraints)
        AR = 1/self._AR(xdict['c_b'])
        CL = xdict["Cw_AR"] * AR
        Re = self.Re(xdict["Cw_AR"], xdict["c_b"][0])
        funcs["obj"]=\
            (pi * AR * A1**2 * (1  + self.Nn @ self.A**2) + AR * self.w_th @ (cd0 * xdict["c_b"]))/CL
            # (CL**2/pi/AR * (1 + pi**2/6) + AR * self.w_th @ (cd0 * xdict["c_b"]))/CL
            # (CL**2/pi/AR * (1 + pi**2/6) + self.cd0_val(CL, Re))/CL
        return funcs

    def metrics(self, xdict):
        xdict["A"] = self.A
        return super().metrics(xdict)

class SimpleLiftDist(BaseLiftDist):
    def __init__(self, W, Ny,
                cd0_model = "flat_plate",
                cd0_val=0.001,
                cl_model = "flat_plate",
                bounds={"lb":{}, "ub":{}},
                ):

        variables = VariablesDict({
            "CL":Variable(
                        name = "CL",
                        size = 1,
                        dflt = 0.1,
                        ub = 1.,
                        lb = 0.,
                        index = (0,1),
                        scale = 1.,
                        offset = 0.
                    ),
            "AR" : Variable(
                        name = "AR",
                        size = 1,
                        dflt = 10.,
                        ub = 300.,
                        lb = 0.,
                        index = (1,2),
                        scale = .1,
                        offset = 0.
                    )
        })

        super().__init__(
                variables = variables,
                W = W,
                Ny = Ny, Na = 1,
                cd0_model = cd0_model,
                cd0_val= cd0_val,
                cl_model = cl_model,
                bounds = bounds)

        self.w_th, self.theta_pts, self.y_pts = quadrature_weights(2*Ny - 1)
        self.M, self.M1, self.M_, self.M1_, self.Nn = even_sine_exp(self.theta_pts, Na=1)

        # Only consider half of the chord
        self.w_th = self.w_th[:Ny]
        self.w_th[-1] /= 2
        self.theta_pts = self.theta_pts[:Ny]
        self.y_pts = self.y_pts[:Ny]
        self.M1 = self.M1[:Ny]

    def Re(self, CL, AR):
        return (1 / 2 * rho / self.W * CL/AR)**(-0.5) * rho/ nu / AR 
    
    def cl(self, CL):
        return 4 * CL * self.M1 / pi

    def metrics(self, xdict):
        AR = xdict["AR"]
        CL = xdict["CL"]
        # AR = 1/xdict["_AR"]
        # CL = xdict["CL"]
        # c = 1 / self.w_th / self.Ny
        c = np.ones(self.Ny)

        Re = self.Re(CL, AR)
        re = c * Re

        cl = self.cl(CL)
        cdp = self.CD0_2D(CL, AR)
        CDp = self.w_th @ cdp # constant chord

        alpha_i = CL / AR/pi *c
        CDi = CL**2 / AR / pi

        L_D = CL / (CDp + CDi)

        return c, AR, re, Re, cl, CL, cdp, CDp, alpha_i, CDi, L_D, self.W

    def objective(self, xdict, constraints=[]):
        # AR = npj.atleast_1d(1/xdict["_AR"])
        # CL = npj.atleast_1d(xdict["CL"])
        AR = npj.atleast_1d(xdict["AR"])
        CL = npj.atleast_1d(xdict["CL"])

        cdp = self.CD0_2D(CL, AR)
        CDp = self.w_th @ cdp # constant chord

        funcs = {"obj":
                    (CDp + CL**2 / AR / pi)/CL
                }
        for con in constraints:
            if con['name'] == "ReCon":
                funcs.update({
                    "ReCon": (self.Re(CL, AR) - con['lb'])/(con['ub'] - con['lb'])
                })
            else:
                warnings.warn(f"{con['name']} is not a valid constraint")
        return funcs

    def addCon(self, optProb, constraint, x0):
        x0['CL'] = np.atleast_1d(x0['CL'])
        x0['AR'] = np.atleast_1d(x0['AR'])
        # AR = npj.atleast_1d(1/x0["_AR"])
        # CL = npj.atleast_1d(x0["CL"])

        f0 = self.objective(x0, [constraint])
        if constraint['name'] == "ReCon":
            optProb.addConGroup('ReCon', f0['ReCon'].shape[0],
            upper=1., 
            lower=0., 
            )
        else:
            return super().addCon(optProb, constraint, x0)
        return optProb

    def CD0_2D(self, CL, AR):
        if self.cd0_model=="flat_plate":
            cl = self.cl(CL)
            re = self.Re(CL, AR) * npj.ones_like(cl)
            return 0.074/re**(1/5) # flat plate
        elif self.cd0_model == "constant":
            return self.cd0_val
        elif self.cd0_model == 'function':
            cl = self.cl(CL)
            re = self.Re(CL, AR) * npj.ones_like(cl)
            return self.cd0_val(cl, re)
        else:
            raise NotImplementedError

class RectangularWingLiftDist(BaseLiftDist):
    def __init__(self, W, Ny, Na,
                cd0_model = "flat_plate",
                cd0_val=0.001,
                cl_model = "flat_plate",
                bounds={"lb":{}, "ub":{}},
                ):

        variables = VariablesDict({
            "CL":Variable(
                        name = "CL",
                        size = 1,
                        dflt = 0.1,
                        ub = 1.,
                        lb = 0.,
                        index = (0,1),
                        scale = 1.,
                        offset = 0.
                    ),
            "AR" : Variable(
                        name = "AR",
                        size = 1,
                        dflt = 10.,
                        ub = 300.,
                        lb = 0.,
                        index = (1,2),
                        scale = .1,
                        offset = 0.
                    ),
            "A" :  Variable(
                        name = "A",
                        size = Na,
                        dflt = 0.1,
                        ub = .1,
                        lb = -.1,
                        index = (Ny + 1, Na + Ny + 1),
                        scale = 10.,
                        offset = 0.
                    )
        })

        super().__init__(
                variables = variables,
                W = W,
                Ny = Ny, Na = 1,
                cd0_model = cd0_model,
                cd0_val= cd0_val,
                cl_model = cl_model,
                bounds = bounds)

        self.w_th, self.theta_pts, self.y_pts = quadrature_weights(Ny, halfspan=True)
        self.M, self.M1, self.M_, self.M1_, self.Nn = even_sine_exp(self.theta_pts, Na=Na, halfspan=True)


    def Re(self, CL, AR):
        return (1 / 2 * rho / self.W * CL/AR)**(-0.5) * rho/ nu / AR 

    def CDi(self, CL, AR, A):
        return CL**2/pi/AR * (1  + self.Nn @ A**2)

    def cl(self, CL, A):
        return 4 * CL / pi * (self.M1 + self.M @ A)

    def metrics(self, xdict):
        AR = xdict["AR"]
        CL = xdict["CL"]
        A = xdict["A"]

        # c = 1 / self.w_th / self.Ny
        c = np.ones(self.Ny)

        Re = self.Re(CL, AR)
        re = c * Re

        cl = self.cl(CL, A)

        cdp = self.CD0_2D(CL, AR, A)
        CDp = self.w_th @ cdp # constant chord

        alpha_i = CL / AR / pi * (self.M_ @ A + self.M1_)
        CDi = self.CDi(CL, AR, A)

        L_D = CL / (CDp + CDi)
        return c, AR, re, Re, cl, CL, cdp, CDp, alpha_i, CDi, L_D, self.W

    def objective(self, xdict, constraints=[]):
        AR = npj.atleast_1d(xdict["AR"])
        A = npj.atleast_1d(xdict["A"])
        CL = npj.atleast_1d(xdict["CL"])

        cdp = self.CD0_2D(CL, AR, A)
        CDp = self.w_th @ cdp # constant chord
        CDi = self.CDi(CL, AR, A)

        funcs = {"obj":
                    (CDp + CDi)/CL
                }
        for con in constraints:
            if con['name'] == "ReCon":
                funcs.update({
                    "ReCon": (self.Re(CL, AR) - con['lb'])/(con['ub'] - con['lb'])
                })
            else:
                warnings.warn(f"{con['name']} is not a valid constraint")
        return funcs

    def addCon(self, optProb, constraint, x0):
        x0['CL'] = np.atleast_1d(x0['CL'])
        x0['AR'] = np.atleast_1d(x0['AR'])
        x0['A'] = np.atleast_1d(x0['A'])

        f0 = self.objective(x0, [constraint])
        if constraint['name'] == "ReCon":
            optProb.addConGroup('ReCon', f0['ReCon'].shape[0],
            upper=1., 
            lower=0., 
            )
        else:
            return super().addCon(optProb, constraint, x0)
        return optProb

    def CD0_2D(self, CL, AR, A):
        if self.cd0_model=="flat_plate":
            cl = self.cl(CL, A)
            re = self.Re(CL, AR) * npj.ones_like(cl)
            return 0.074/re**(1/5) # flat plate
        elif self.cd0_model == "constant":
            return self.cd0_val
        elif self.cd0_model == 'function':
            cl = self.cl(CL, A)
            re = self.Re(CL, AR) * npj.ones_like(cl)
            return self.cd0_val(cl, re)
        else:
            raise NotImplementedError

class FreeChordLiftDist(BaseLiftDist):
    def __init__(self, W, Ny, Na,
                cd0_model = "flat_plate",
                cd0_val=0.001,
                cl_model = "flat_plate",
                bounds={"lb":{}, "ub":{}},
                ):

        self.w_th, self.theta_pts, self.y_pts = quadrature_weights(Ny, halfspan=True)
        self.M, self.M1, self.M_, self.M1_, self.Nn = even_sine_exp(self.theta_pts, Na=Na, halfspan=True)
        variables = VariablesDict({
            "CL":Variable(
                        name = "CL",
                        size = 1,
                        dflt = 0.1,
                        ub = 1.,
                        lb = 0.,
                        index = (0,1),
                        scale = 1.,
                        offset = 0.
                    ),
            "AR" : Variable(
                        name = "AR",
                        size = 1,
                        dflt = 10.,
                        ub = 300.,
                        lb = 0.,
                        index = (1,2),
                        scale = .1,
                        offset = 0.
                    ),
            "c" : Variable(
                        name = "c",
                        size = Ny,
                        dflt = 1.,
                        ub = 1./self.w_th,
                        lb = 0.,
                        index = (Ny+ 2, Ny+3),
                        scale = 1.,
                        offset = 0.
                    ),
            "A" :  Variable(
                        name = "A",
                        size = Na,
                        dflt = 0.1,
                        ub = .1,
                        lb = -.1,
                        index = (Ny + 2, Na + Ny + 2),
                        scale = 100.,
                        offset = 0.
                    )
        })

        super().__init__(
                variables = variables,
                W = W,
                Ny = Ny, Na = 1,
                cd0_model = cd0_model,
                cd0_val= cd0_val,
                cl_model = cl_model,
                bounds = bounds)


    def Re(self, CL, AR, c):
        return (1 / 2 * rho / self.W * CL/ AR)**(-0.5) * rho/ nu * c / AR

    def CDi(self, CL, AR, A):
        return CL**2/pi/AR * (1  + self.Nn @ A**2)

    def cl(self, CL, c, A):
        return 4 * CL / c / pi * (self.M1 + self.M @ A)

    def metrics(self, xdict):
        AR = xdict["AR"]
        CL = xdict["CL"]
        c = xdict["c"]
        A = xdict["A"]

        re = self.Re(CL, AR, c)
        Re = self.Re(CL, AR, 1.)

        cl = self.cl(CL, c, A)

        cdp = self.CD0_2D(CL, AR, c, A)
        CDp = self.w_th @ (c * cdp)

        alpha_i = CL / AR / pi * (self.M_ @ A + self.M1_)
        CDi = self.CDi(CL, AR, A)

        L_D = CL / (CDp + CDi)

        return c, AR, re, Re, cl, CL, cdp, CDp, alpha_i, CDi, L_D, self.W

    def objective(self, xdict, constraints=[]):
        AR = npj.atleast_1d(xdict["AR"])
        A = xdict["A"]
        c_ = xdict["c"]
        CL = npj.atleast_1d(xdict["CL"])

        c = c_ / (self.w_th @ c_)

        cdp = self.CD0_2D(CL, AR, c, A)
        CDp = self.w_th @ (c * cdp)
        CDi = self.CDi(CL, AR, A)

        funcs = {"obj":
                    (CDp + CDi)/CL + 1e-3*(self.w_th @ c_ - 1.)**2
                }
        for con in constraints:
            if con['name'] == "ReCon":
                funcs.update({
                    "ReCon": (self.Re(CL, AR, c) - con['lb'])/(con['ub'] - con['lb'])
                })
            elif con['name'] == "chordDist":
                funcs.update({
                    "chordDist": self.w_th @ c -1.
                })
            else:
                warnings.warn(f"{con['name']} is not a valid constraint")
            
        return funcs

    def addCon(self, optProb, constraint, x0):
        x0['CL'] = np.atleast_1d(x0['CL'])
        x0['AR'] = np.atleast_1d(x0['AR'])
        x0['A'] = np.atleast_1d(x0['A'])

        f0 = self.objective(x0, [constraint])
        if constraint['name'] == "ReCon":
            optProb.addConGroup('ReCon', f0['ReCon'].shape[0],
            upper=1., 
            lower=0., 
            )
        elif constraint['name'] == "chordDist":
            optProb.addConGroup('chordDist', 1,
            upper=constraint['ub'], 
            lower=constraint['lb'], 
            linear=True,
            wrt=['c'], 
            jac={'c':np.ones((1, self.Ny))}
            )
        else:
            return super().addCon(optProb, constraint, x0)
        return optProb

    def CD0_2D(self, CL, AR, c, A):
        if self.cd0_model=="flat_plate":
            Re = self.Re(CL, AR, c)
            return 0.074/Re**(1/5) # flat plate
        elif self.cd0_model == "constant":
            return self.cd0_val
        elif self.cd0_model == 'function':
            cl = self.cl(CL, c, A)
            re = self.Re(CL, AR, c) * npj.ones_like(cl)
            return self.cd0_val(cl, re)
        else:
            raise NotImplementedError

class ChebChordLiftDist(BaseLiftDist):
    def __init__(self, W, Ny, Na, Nc,
                cd0_model = "flat_plate",
                cd0_val=0.001,
                cl_model = "flat_plate",
                bounds={"lb":{}, "ub":{}},
                ):
        self.Nc = Nc
        self.w_th, self.theta_pts, self.y_pts = quadrature_weights(Ny, halfspan=True)
        self.M, self.M1, self.M_, self.M1_, self.Nn = even_sine_exp(self.theta_pts, Na=max(Na, Nc), halfspan=True)
        variables = VariablesDict({
            "CL":Variable(
                        name = "CL",
                        size = 1,
                        dflt = 0.1,
                        ub = 1.,
                        lb = 0.,
                        index = (0,1),
                        scale = 1.,
                        offset = 0.
                    ),
            "AR" : Variable(
                        name = "AR",
                        size = 1,
                        dflt = 10.,
                        ub = 300.,
                        lb = 0.,
                        index = (1,2),
                        scale = .1,
                        offset = 0.
                    ),
            "C" : Variable(
                        name = "C",
                        size = Nc,
                        dflt = .1,
                        ub = .1,
                        lb = -.1,
                        index = (2, Nc+2),
                        scale = 100.,
                        offset = 0.
                    ),
            "A" :  Variable(
                        name = "A",
                        size = Na,
                        dflt = 0.1,
                        ub = .1,
                        lb = -.1,
                        index = (Nc + 2, Na + Nc + 2),
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

    def c(self, C, AR):
        # C = [C1, C3, C5, ...]
        c0 = 1 - C[0] * pi/4 # constant term such that integral is 1
        return c0 + (self.M1 * C[0] + self.M[:, :self.Nc-1] @ C[1:])

    def Re(self, CL, AR, c):
        return (1 / 2 * rho / self.W * CL/ AR)**(-0.5) * rho/ nu * c / AR

    def CDi(self, CL, AR, A):
        return CL**2/pi/AR * (1  + self.Nn[:self.Na] @ A**2)

    def cl(self, CL, c, A):
        return 4 * CL / c / pi * (self.M1 + self.M[:, :self.Na] @ A)

    def metrics(self, xdict):
        AR = xdict["AR"]
        CL = xdict["CL"]
        C = xdict["C"]
        A = xdict["A"]

        c = self.c(C, AR)

        re = self.Re(CL, AR, c)
        Re = self.Re(CL, AR, 1.)

        cl = self.cl(CL, c, A)

        cdp = self.CD0_2D(CL, AR, c, A)
        CDp = self.w_th @ (c * cdp)

        alpha_i = CL / AR / pi * (self.M_[:, :self.Na] @ A + self.M1_)
        CDi = self.CDi(CL, AR, A)

        L_D = CL / (CDp + CDi)

        return c, AR, re, Re, cl, CL, cdp, CDp, alpha_i, CDi, L_D, self.W

    def objective(self, xdict, constraints=[]):
        AR = npj.atleast_1d(xdict["AR"])
        A = xdict["A"]
        C = xdict["C"]
        CL = npj.atleast_1d(xdict["CL"])

        c = self.c(C, AR)
        cdp = self.CD0_2D(CL, AR, c, A)
        CDp = self.w_th @ (c * cdp)
        CDi = self.CDi(CL, AR, A)

        funcs = {"obj":
                    (CDp + CDi)/CL
                }
        for con in constraints:
            if con['name'] == "ReCon":
                funcs.update({
                    "ReCon": (self.Re(CL, AR, c) - con['lb'])/(con['ub'] - con['lb'])
                })
            elif con['name'] == "clCon":
                funcs.update({
                    "clCon_ub":  4 * CL / pi * (self.M1 + self.M[:, :self.Na] @ A) - con['ub'] * c,
                    "clCon_lb": -4 * CL / pi * (self.M1 + self.M[:, :self.Na] @ A) + con['lb'] * c,
                })
            elif con['name'] == "chordVar":
                funcs.update({
                    "chordVar": c[1:] -c[:-1]
                })
            else:
                warnings.warn(f"{con['name']} is not a valid constraint")
            
        return funcs

    def addCon(self, optProb, constraint, x0):
        x0['CL'] = np.atleast_1d(x0['CL'])
        x0['AR'] = np.atleast_1d(x0['AR'])
        x0['A'] = np.atleast_1d(x0['A'])

        f0 = self.objective(x0, [constraint])
        if constraint['name'] == "ReCon":
            optProb.addConGroup('ReCon', f0['ReCon'].shape[0],
            upper=1., 
            lower=0., 
            wrt=['C', 'CL', 'AR'],
            )
        elif constraint['name'] == 'clCon':
            jac=self.gradient(x0, [constraint])['clCon_ub']
            jac['c'] = {'coo':[np.arange(self.Ny), np.arange(self.Ny), np.diag(jac['c'])],
                        'shape':[self.Ny, self.Ny]}
            optProb.addConGroup('clCon'+'_ub', f0['clCon_ub'].shape[0],
                                upper=0.,
                                linear=True,
                                jac=jac,
                                wrt=['c', 'AR', 'A']
                                )

            jac=self.gradient(x0, [constraint])['clCon_lb']
            jac['c'] = {'coo':[np.arange(self.Ny), np.arange(self.Ny), np.diag(jac['c'])],
                        'shape':[self.Ny, self.Ny]}
            optProb.addConGroup('clCon_lb', f0['clCon_lb'].shape[0],
                                upper=0.,
                                linear=True,
                                jac=jac,
                                wrt=['c', 'AR', 'A']
                                )
        elif constraint['name'] == 'chordVar':
            jac=self.gradient(x0, [constraint])['chordVar']
            if "A" in jac:
                del jac['A']
            del jac['CL']
            optProb.addConGroup('chordVar', f0['chordVar'].shape[0],
                                lower=constraint['lb'],
                                upper=constraint['ub'],
                                linear=True,
                                jac=jac,
                                wrt=['c']
                                )
            return super().addCon(optProb, constraint, x0)
        return optProb

    def CD0_2D(self, CL, AR, c, A):
        if self.cd0_model=="flat_plate":
            Re = self.Re(CL, AR, c)
            return 0.074/Re**(1/5) # flat plate
        elif self.cd0_model == "constant":
            return self.cd0_val
        elif self.cd0_model == 'function':
            cl = self.cl(CL, c, A)
            re = self.Re(CL, AR, c) * npj.ones_like(cl)
            return self.cd0_val(cl, re)
        else:
            raise NotImplementedError

class Trapezoidal(LiftDistND):
    def __init__(self, W, Na=3, Ny=15,
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
                        size = 2,
                        dflt = 0.1,
                        ub = 1.,
                        lb = 0.,
                        index = (1,3),
                        scale = 1.,
                        offset = 0.
                    ),
            "A" :  Variable(
                        name = "A",
                        size = Na,
                        dflt = 0.,
                        ub = .01,
                        lb = -.01,
                        index = (3, Na + 3),
                        scale = 1,
                        offset = 0.
                    )
        })

        BaseLiftDist.__init__(self,
            variables = variables,
            W = W,
            Ny = Ny, Na = Na,
            cd0_model = cd0_model,
            cd0_val= cd0_val,
            cl_model = cl_model,
            bounds = bounds)

        self.w_th, self.theta_pts, self.y_pts = quadrature_weights(Ny, halfspan=True)
        self.M, self.M1, self.M_, self.M1_, self.Nn = even_sine_exp(self.theta_pts, Na, halfspan=True)

    def chord(self, c_b):
        # c_b = [tip, root]
        return (c_b[1] + (c_b[1] - c_b[0])* (pi/2 -  self.theta_pts) * 2/pi).squeeze()

    def metrics(self, xdict):
        xd = deepcopy(xdict)
        xd["c_b"] = self.chord(xdict["c_b"])
        return super().metrics(xd)

    def objective(self, xdict, constraints=[]):
        xd = deepcopy(xdict)
        xd["c_b"] = self.chord(xdict["c_b"])
        return super().objective(xd, constraints)

class TrapezoidalWeightModel(LiftDistND):
    def __init__(self, Wothers, Na=3, Ny=15,
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
                        size = 2,
                        dflt = 0.1,
                        ub = 1.,
                        lb = 0.,
                        index = (1,3),
                        scale = 1.,
                        offset = 0.
                    ),
            "A" :  Variable(
                        name = "A",
                        size = Na,
                        dflt = 0.,
                        ub = .01,
                        lb = -.01,
                        index = (3, Na + 3),
                        scale = 1,
                        offset = 0.
                    ),
            "b": Variable(
                        name = "b",
                        size = 1,
                        dflt = 0.,
                        ub = 80,    
                        lb = 1,
                        index = (Na + 4, Na+5),
                        scale = 1,
                        offset = 0.
                    )
        })

        self.Wothers = Wothers

        BaseLiftDist.__init__(self,
            variables = variables,
            W = Wothers,
            Ny = Ny, Na = Na,
            cd0_model = cd0_model,
            cd0_val= cd0_val,
            cl_model = cl_model,
            bounds = bounds)

        self.w_th, self.theta_pts, self.y_pts = quadrature_weights(Ny, halfspan=True)
        self.M, self.M1, self.M_, self.M1_, self.Nn = even_sine_exp(self.theta_pts, Na, halfspan=True)

    def chord(self, c_b):
        # c_b = [tip, root]
        return (c_b[1] + (c_b[1] - c_b[0])* (pi/2 -  self.theta_pts) * 2/pi).squeeze()

    def metrics(self, xdict):
        self.W = self.Wothers + self.weightModel(xdict)
        xd = deepcopy(xdict)
        xd["c_b"] = self.chord(xdict["c_b"])
        return super().metrics(xd)

    def objective(self, xdict, constraints=[]):
        self.W = self.Wothers + self.weightModel(xdict)
        xd = deepcopy(xdict)
        xd["c_b"] = self.chord(xdict["c_b"])
        return super().objective(xd, constraints)

    def weightModel(self, xdict):
        _AR = self._AR( self.chord(xdict["c_b"]))
        b = xdict['b']
        S = b**2 * _AR
        l = xdict['c_b'][0]/xdict['c_b'][1]
        t_c = 0.093
        k1 = 4.22*S
        k2 = 1.642e-6*b**3/S/(t_c)*(1+2*l)/(1+l)
        Wwing = (k1 + k2*self.Wothers)/(1-k2)
        return Wwing

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
 
    def A1(self, Vb2):
        return 2 * self.W / (rho*pi) * cp.power(Vb2,-1)


    def CDi(self, A1, A, _AR):
        return pi / _AR * (A1**2 + self.Nn @ A**2)

    def cl(self, A1, A, c_2b):
        return 2 * (self.M1 * A1 + self.M @ A) / c_2b
   
    def metrics(self, x):
        c_2b = x["c_2b"]
        _AR = (self.w_th @ c_2b)
        AR = 1 / _AR
        c = 2 * c_2b * AR
        Vb2 = x["Vb2"]
        A = x["A"]

        A1 = self.A1(Vb2)
        CL = pi * AR * A1
        cl = self.cl(A1, A, c_2b)

        re = self.Re(Vb2, c_2b)

        Re = self.w_th @ np.multiply(re, c_2b) / _AR
        
        cdp = self.CD0_2D(Vb2, c_2b, A, A1).value

        CDp = self.w_th @ np.multiply(cdp,c_2b) / _AR

        alpha_i = self.alpha_i(A1, A)

        CDi = self.CDi(A1, A, _AR)

        L_D = CL / (CDp + CDi)
        return c, AR, re, Re, cl, CL, cdp, CDp, alpha_i, CDi, L_D


    def obj(self, d):
        _AR = self._AR(d["c_2b"])
        A1 = self.A1(d["Vb2"])
        CDi_AR = pi * (A1**2 + cp.multiply(self.Nn, d["A"]) @ d["A"])
        cd0 = self.CD0_2D(d["Vb2"], d["c_2b"], d["A"], A1)
        return d["Vb2"] * (CDi_AR + self.w_th @ cp.multiply(cd0, d["c_2b"]))

class GPLiftDist(BaseLiftDist):
    def __init__(self, W, Ny=11,
                    cd0_model = "flat_plate",
                    cd0_val=0.001,
                    cl_model = "flat_plate",
                    bounds={"lb":{}, "ub":{}},
                    ):
        # use midpoints 
        Ny = Ny - 1
        
        # define variables
        variables = VariablesDict({
            "CL":Variable(
                        name = "CL",
                        size = 1,
                        dflt = 0.1,
                        ub = 1.,
                        lb = 0.,
                        index = (0,1),
                        scale = 1.,
                        offset = 0.
                    ),
            "AR" :  Variable(
                        name = "AR",
                        size = 1,
                        dflt = 0.,
                        ub = 1e3,
                        lb = 0.,
                        index = (1,2),
                        scale = 1,
                        offset = 0.
                    ),
            "c" : Variable(
                        name = "c",
                        size = Ny,
                        dflt = 1/Ny,
                        ub = 1.,
                        lb = 0.,
                        index = (2,Ny+2),
                        scale = 1.,
                        offset = 0.
                    ),
            "cl" : Variable(
                        name = "cl",
                        size = Ny,
                        dflt = 1/Ny,
                        ub = 1.,
                        lb = 0.,
                        index = (Ny+2,2*Ny+2),
                        scale = 1.,
                        offset = 0.
                    ),  
        })

        super().__init__(
                variables = variables,
                W = W,
                Ny = Ny,
                cd0_model = cd0_model,
                cd0_val= cd0_val,
                cl_model = cl_model,
                bounds = bounds)

        _, self.theta_pts, self.y_pts = quadrature_weights(Ny+1)
        self.y_pts /= 2 # -1/2 to 1/2
        
        self.y_midpts = (self.y_pts[1:] + self.y_pts[:-1])/2
        self.w_y = np.diff(np.squeeze(self.y_pts))

        dy = self.y_midpts - self.y_midpts.T
        for i in range(dy.shape[0]):
            dy[i,i] = 1.
        self.M = self.w_y / dy**2
        for i in range(self.M.shape[0]):
            self.M[i,i] = 0.

    def alpha_i(self, AR, c, CL, cl):
        return  self.M @cp.multiply(c, cl) * (CL/ AR / (8*pi))

    def Re(self, CL, AR, c):
        return (1 / 2 * rho / self.W * CL * AR)**(-0.5) * rho/ nu * c

    def metrics(self, x):
        c = x["c"]
        AR = x["AR"]
        CL = x["CL"]
        cl = x["cl"]

        re = self.Re(CL, AR, c)
        re.name = 're'

        Re = self.w_y @ re
        Re.name = 'Re'
        
        cdp = self.CD0_2D(CL, AR, c, cl)
        cdp.name = 'cdp'

        CDp = self.w_y @ cp.multiply(c, cdp)
        CDp.name = 'CDp'

        alpha_i = self.alpha_i(AR, c, CL, cl)
        alpha_i.name = 'alpha_i'

        CDi = CL * self.w_y @ cp.multiply(alpha_i, cl)
        CDi.name= 'CDi'

        L_D = CL / (CDp + CDi)
        L_D.name = 'L_D'

        return c, AR, re, Re, cl, CL, cdp, CDp, alpha_i, CDi, L_D

    def objective(self, xdict, constraints):
        AR = xdict["AR"]
        CL = xdict["CL"]
        cl = xdict["cl"]
        c = xdict["c"]

        CDi_CL = self.w_y @ cp.multiply(self.alpha_i(AR, c, CL, cl), cl)
        cdp = self.CD0_2D(CL, AR, c, cl)
        CDp_CL = self.w_y @ cp.multiply(c, cdp) / CL
        return {"obj":CDi_CL + CDp_CL}

    def optimize(self, 
        x0=None, alg='cvx', restarts=1, 
        solver_options={}, obj_scale=1.,
        constraints=[],
        smooth_penalty={},
        sens=None,
        usejit=False):

        opt = {}
        opt.update(solver_options)

        d = {
            k:cp.Variable(pos=True, shape=np.atleast_1d(x0[k]).shape, 
                        value=np.atleast_1d(x0[k])) 
                        for k in x0
        }

        obj = cp.Minimize(self.objective(d, {})["obj"])
        
        # sums of cl and c
        con = [cp.sum(d["c"]) <= 1., self.w_y * d["c"] * d["cl"] <= 1.]

        # Add Bounds
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

    def CD0_2D(self, CL, AR, c, cl):
        if self.cd0_model=="flat_plate":
            Re = self.Re(CL, AR, c)
            return 0.074/Re**(1/5) # flat plate
        elif self.cd0_model == "constant":
            return self.cd0_val
        elif self.cd0_model == 'function':
            Re = self.Re(CL, AR, c)
            return self.cd0_val(CL*cl, Re)
        else:
            raise NotImplementedError
