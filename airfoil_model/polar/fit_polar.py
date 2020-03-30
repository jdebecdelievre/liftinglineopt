import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
torch.set_default_dtype(torch.float64)
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import datetime
import os
op = os.path
import click
import json
import pdb

def build_model(hidden_layers):
    layers = [
    nn.Linear(2, hidden_layers[0]),
    nn.Sigmoid()
    ] 
    for i in range(len(hidden_layers)-1):
        layers += [nn.Linear(hidden_layers[i], hidden_layers[i+1]),
        nn.Sigmoid()]
    return  nn.Sequential(*layers, nn.Linear(hidden_layers[-1], 1))

def load_data(filename, params):
    data = pd.read_csv(filename)
    data = data.loc[np.isnan(data.Cm)]

    inp = torch.tensor(data[["Cl", "Re"]].values)
    out = torch.tensor(data[["Cd"]].values)
    inp_mean = inp.mean(0)
    out_mean = out.mean(0)
    inp_std = inp.std(0)
    out_std = out.std(0)
    inp = (inp-inp_mean)/inp_std
    out = (out-out_mean)/out_std


    dataset = TensorDataset(inp, out)
    loader = DataLoader(dataset, 
                        batch_size=params['batch_size'], 
                        shuffle=False)
    
    return loader, dataset, inp, out, inp_mean, out_mean, inp_std, out_std, data

@click.command()
@click.option('-m','--model_restart', default=None, help='restart from model')
@click.option('-lr','--lr', default=None,  help='learning rate', type=click.FLOAT)
@click.option('-fs','--filename_suffix', default=None,  help='filename_suffix for model', type=click.STRING)
@click.option('-hl','--hidden_layers', default=None, help='hidden layers', type=click.STRING)
@click.option('-e','--epochs', default=None, help='number of epochs', type=click.INT)
def main(model_restart, filename_suffix, lr, hidden_layers, epochs):
    # Load params
    if model_restart:
        with open(op.join("models", model_restart+'.json'), "r") as f:
            params = json.load(f)
    else:
        params= dict(
            hidden_layers = [8],
            epochs = 100000,
            batch_size = 500,
            lr=1e-3
        )
    if lr:
        params['lr'] = lr
    if epochs:
        params['epochs'] = epochs
    if hidden_layers:
        hidden_layers = hidden_layers[1:-1] if hidden_layers[0]=='[' else hidden_layers
        hidden_layers = [int(l) for l in hidden_layers.split(',')]

    # Load data
    loader, dataset, inp, out, inp_mean, out_mean, inp_std, out_std, e231 = load_data("../data/e231.csv", params)
    writer = SummaryWriter(comment=("_" + filename_suffix) if filename_suffix else "")

    # Build model
    model = build_model(params["hidden_layers"])
    if model_restart:
        model.load_state_dict(torch.load(op.join("models", model_restart + ".mdl")))

    opt = torch.optim.Adam(model.parameters(),
                        lr=params['lr'],
                        weight_decay=0.001,
                        betas=(0.9, 0.999), 
                        eps=1e-08
    )

    # Train
    model.train()
    # plt.plot(e231.alpha, e231.Cl, '+-')
    f, a = plt.subplots()
    a.plot(inp.data[:,0], model(inp)[:,0].data, '+')
    a.plot(inp.data[:,0], out[:,0].data, '+')
    f.savefig("tmp.png")
    f_, a_ = plt.subplots()
    LL = []

    try:
        for e in range(params["epochs"]):
            L = 0
            ite = 0
            for i, o in loader:
                loss = (((model(i) - o))**2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                L += loss.data
                ite += i.shape[0]
            LL.append(L/ite)

            if e%10 ==0:
                writer.add_scalar('loss',L/ite, e)

            if e%100 == 0:
                a.clear()
                a.plot(inp.data[:,0], model(inp)[:,0].data, '+')
                a.plot(inp.data[:,0], out[:,0].data, '+')
                writer.add_figure("data_fit",f)

    except KeyboardInterrupt:
        pass

    # Save model
    model_file_name = os.path.join("models", os.path.basename(writer.logdir)+ ".mdl")
    torch.save(model.state_dict(), model_file_name)
    with open(model_file_name[:-4]+'.json', "w") as f:
        params["epochs"] = e
        json.dump(params, f)

if __name__ == "__main__":
    main()