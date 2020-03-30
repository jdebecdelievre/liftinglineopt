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
    return  nn.Sequential(*layers, nn.Linear(hidden_layers[-1], 3))

def load_data(filename, params):
    data = pd.read_csv(filename)
    data["has_cd"] = 0
    data["has_cm"] = 0
    data["has_cl"] = 1
    # data.loc[np.isnan(data.Cd), "has_cm"] = 1
    # data.loc[np.isnan(data.Cm), "has_cd"] = 1
    data = data.fillna(0)

    inp = torch.tensor(data[["alpha", "Re"]].values)
    out = torch.tensor(data[["Cl", "Cd", "Cm"]].values)
    inp_mean = inp.mean(0)
    out_mean = out.mean(0)
    inp_std = inp.std(0)
    out_std = out.std(0)
    msk = torch.tensor(data[["has_cl","has_cd", "has_cm"]].values)
    inp = (inp-inp_mean)/inp_std
    out = (out-out_mean)/out_std


    dataset = TensorDataset(inp, out, msk)
    loader = DataLoader(dataset, 
                        batch_size=params['batch_size'], 
                        shuffle=False)
    
    return loader, dataset, inp, out, msk, inp_mean, out_mean, inp_std, out_std, data

@click.command()
@click.option('-m','--model_restart', default=None, help='restart from model')
@click.option('-lr','--lr', default=None,  help='learning rate', type=click.FLOAT)
@click.option('-hl','--hidden_layers', default=None, help='hidden layers', type=click.STRING)
@click.option('-e','--epochs', default=None, help='number of epochs', type=click.INT)
def main(model_restart, lr, hidden_layers, epochs):
    # Load params
    if model_restart:
        with open(op.join("models", model_restart+'.json'), "r") as f:
            params = json.load(f)
    else:
        params= dict(
            hidden_layers = [4,4],
            epochs = 100000,
            batch_size = 500,
            lr=6e-4
        )
    if lr:
        params['lr'] = lr
    if epochs:
        params['epochs'] = epochs
    if hidden_layers:
        hidden_layers = hidden_layers[1:-1] if hidden_layers[0]=='[' else hidden_layers
        hidden_layers = [int(l) for l in hidden_layers.split(',')]

    # Load data
    loader, dataset, inp, out, msk, inp_mean, out_mean, inp_std, out_std, e231 = load_data("e231.csv", params)
    writer = SummaryWriter()

    # Build model
    model = build_model(params["hidden_layers"])
    if model_restart:
        model.load_state_dict(torch.load(op.join("models", model_restart + ".mdl")))

    # model = nn.Linear(2,3)
    opt = torch.optim.Adam(model.parameters(),
                        lr=params['lr'],
                        weight_decay=0.001,
                        betas=(0.9, 0.999), 
                        eps=1e-08
    )
                        # momentum=0.9,
                        # dampening=0.,
                        # nesterov=True)

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
            for i, o, m in loader:
                loss = (((model(i) - o)*m)**2).sum()
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
                # f.savefig("tmp.png")
                # a_.clear()
                # a_.loglog(LL)
                # f_.savefig(learning.png")
                # writer.add_figure(f_)

    except KeyboardInterrupt:
        pass

    # Save model
    model_file_name = os.path.join("models", os.path.basename(writer.logdir)+ ".mdl")
    torch.save(model.state_dict(), model_file_name)
    with open(model_file_name[:-4]+'.json', "w") as f:
        params["epochs"] = e
        json.dump(params, f)
    # import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()