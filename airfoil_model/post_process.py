from jax.config import config
config.update("jax_enable_x64", True)
from jax import numpy as npj

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
op = os.path
import torch
import json
# from abq.train_airfoil import build_model, load_data
from polar.fit_polar import build_model as build_model_pol
from polar.fit_polar import load_data as load_data_pol


def export_weights(model, filename):
    sd = {k:val.numpy() for k, val in model.state_dict().items()}
    np.save(filename, sd)
    return sd

def load_model(model_name, model_type):
    if model_type == 'abq':
        with open(op.join("abq/models", model_name+'.json'), "r") as f:
            params = json.load(f)
        model = build_model(params["hidden_layers"])
        model.load_state_dict(torch.load(op.join("abq/models", model_name + ".mdl")))
    elif model_type == 'polar':
        with open(op.join("polar/models", model_name+'.json'), "r") as f:
            params = json.load(f)
        model = build_model_pol(params["hidden_layers"])
        model.load_state_dict(torch.load(op.join("polar/models", model_name + ".mdl")))
    else:
        raise NotImplementedError(f'No model type {model_type}')
    return model, params

def sigmoid(x):
    return 0.5 * (npj.tanh(x / 2) + 1)

def get_model(model_name, model_type):
    """
    Returns a python function that evaluates to the same value as the pytorch neural network

    | model type | inputs    | outputs    |
    |     abq    | alpha, Re | Cl, Cd, Cm |
    |    polar   |  Cl, Re   |     Cd     |
    
    /!\ input order obviously matters

    """
    torchmodel, params = load_model(model_name, model_type)
    input_mean = torchmodel.input_mean.numpy()
    input_std = torchmodel.input_std.numpy()
    output_mean = torchmodel.output_mean.numpy()
    output_std = torchmodel.output_std.numpy()

    def model(*inp):
        z = npj.vstack(inp).T
        z = (z - input_mean)/input_std
        for layer in torchmodel.children():
            if type(layer) == torch.nn.Linear:
                w = layer.weight.data.numpy()
                b = layer.bias.data.numpy()
                z = z @ w.T + b
            elif type(layer) == torch.nn.Sigmoid:
                z = sigmoid(z)
            else:
                raise NotImplementedError(f'Numpy extraction not support for layer {type(layer)}')
        return z * output_std + output_mean

    return model

# def predict(model, model_type, meshgrid=False, **inputs):
#     if model_type == 'abq':
#         _, _, inp, out, msk, inp_mean, out_mean, inp_std, out_std, _ = \
#             load_data("e231.csv", params)
#         with torch.no_grad():
#             inp_data = (inp * inp_std + inp_mean).data
#             out_data = (model(inp) * out_std + out_mean).data
#         pred = pd.DataFrame.from_dict({
#             "alpha":inp_data[:,0],
#             "Cl":out_data[:,0],
#             "Cd":out_data[:,1],
#             "Cm":out_data[:,2],
#             "Re":inp_data[:,1]
#         })
#     elif model_type == 'polar':
#         pass
#     else:
#         raise NotImplementedError(f'No model type {model_type}')
#     return pred

def read_airfoil_data(file):
    data = pd.read_csv(file)
    data.insert(5,"tmp", data.index)
    data.sort_values(["Re", "tmp"], inplace = True)
    data.drop(columns="tmp", inplace=True)
    data_cm = data[np.isnan(data.Cd)]
    data_cd = data[np.isnan(data.Cm)]
    Re = data.Re.drop_duplicates().values
    return data, data_cd, data_cm

def compare_dataframes(dataframe_dict, ycol, ycolname=None,
                        xcol='alpha', xcolname="Angle of attack (deg)",
                       fig=None, styles=None, colorbar=False, adapt_ticks=False, loglog=False):
    # Read inputs
    if fig:
        f = fig
        a = fig.axes[0]
    else:
        f,a = plt.subplots(figsize=(10,8))
    
    if not styles:
        st = ["-"]*(len(dataframe_dict) - 1) + ["+"]
        styles = {k:st[i] for i, k in enumerate(dataframe_dict)}
    
    
    # Collect available Re, create colormap
    Re = []
    for dat in dataframe_dict:
        Re += (dataframe_dict[dat].Re.drop_duplicates().tolist())
    Re = np.array(Re)
    norm = mpl.colors.Normalize(vmin=Re.min(), vmax=Re.max())
    re_map = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)

    
    # Plot
    for i, r in enumerate(Re):
        for key, dat in dataframe_dict.items():
            e = dat[dat.Re==r]
            if not e[ycol].hasnans:
                a.plot(e[xcol], e[ycol], styles[key], 
                        linewidth=1, 
                        color=re_map.to_rgba(r),
                        label=f"{key}, Re {r}")
    
    a.set_title(",     ".join([f"{key} linestyle : {styles[key]}" 
                    for key in dataframe_dict]))
    if loglog:
        a.set_xscale('log')
        a.set_yscale('log')
    a.grid(True)    
    a.set_xlabel(xcolname, size=20)
    a.set_ylabel(ycolname if ycolname else ycol, size=20)
    if colorbar:
        if adapt_ticks:
            keep_tick = np.ones_like(Re) == 1
            keep_tick[1:] = np.diff(Re)/Re[1:] > 0.05
            tks = Re[keep_tick]
        else:
            tks = None
        cb = f.colorbar(re_map, orientation='vertical', ticks=tks)
        cb.ax.set_ylabel("Reynold\'s number", size=20)
        cb.ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    else:
        a.legend(fontsize=10)
    a.tick_params(labelsize=15)
    return f


def plot_available_data(dat, ycol, ycolname=None, 
                        xcol='alpha', xcolname="Angle of attack (deg)",
                        ccol='Re', ccolname="Reynold\'s number",
                        label=None, fig=None, style=None, colorbar=False, adapt_ticks=False):
    if fig:
        f = fig
        a = fig.axes[0]
    else:
        f,a = plt.subplots(figsize=(10,8))
    
    label = label if label else ""
    
    c = dat[ccol].drop_duplicates().values
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    re_map = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    
    for r in c:
        e = dat[dat[ccol]==r]
        if not e[ycol].hasnans:
            a.plot(e[xcol], e[ycol], style if style else "-+", 
                    label=f"{label}{ccol} {r}", 
                    linewidth=1,
                    color=re_map.to_rgba(r))
    
    a.grid(True)    
    a.set_xlabel(xcolname, size=20)
    a.set_ylabel(ycolname if ycolname else ycol, size=20)
    if colorbar:
        if adapt_ticks:
            keep_tick = np.ones_like(c) == 1
            keep_tick[1:] = np.diff(c)/c[1:] > 0.05
            tks = c[keep_tick]
        else:
            tks = None
        cb = f.colorbar(re_map, orientation='vertical', ticks=tks)
        cb.ax.set_ylabel(ccolname, size=20)
        if ccolname == 'Re':
            cb.ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    else:
        a.legend(fontsize=10)
    a.tick_params(labelsize=15)
    f.tight_layout()
    return f

if __name__ == "__main__":
    e231, e231_cd, e231_cm= read_airfoil_data('data/e231.csv')
    e231.head()