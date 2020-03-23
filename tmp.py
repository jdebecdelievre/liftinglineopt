import numpy as np
import scipy as sp
from scipy.special import expit, logit
import scipy.optimize

def f(x,x0,g,c,k):
    y = c*expit(k*10.*(x-x0)) + g*(1.-c)
    return y

#               x0                      g                       c                       k
p0 = np.array([8.841357069490852e-01, 4.492363462957287e-19, 5.547073496706608e-01, 7.435378446218519e+00])
bounds = np.array([[-1.,1.], [0.,1.], [0.,1.], [0.,20.]])
x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.8911796599834791, 1.0, 1.0, 1.0, 0.33232919909076103, 1.0])
y = np.array([0.999, 0.999, 0.999, 0.999, 0.999, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
s = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

print([pval >= b[0] and pval <= b[1] for pval,b in zip(p0,bounds)])

fit,cov = sp.optimize.curve_fit(f,x,y,p0=p0,sigma=s,bounds=([b[0] for b in bounds],[b[1] for b in bounds]),method='dogbox',tr_solver='exact')

print(fit)
print(cov)