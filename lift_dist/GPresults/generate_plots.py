import sys
sys.path.append('../..')
from run import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

def get_data(LD, folder):
    W = np.load(folder + '/w.npy', allow_pickle=True)
    Nw = len(W)
    print("weights", W)
    c   = np.zeros((Nw, LD.Ny))
    AR  = np.zeros(Nw)
    re  = np.zeros((Nw, LD.Ny))
    Re  = np.zeros(Nw)
    cl  = np.zeros((Nw, LD.Ny))
    CL  = np.zeros(Nw)
    cdp = np.zeros((Nw, LD.Ny))
    CDp = np.zeros(Nw)
    alpha_i  = np.zeros((Nw, LD.Ny))
    CDi = np.zeros(Nw)
    L_D = np.zeros(Nw)

    n = 0
    for i in range(Nw):
        try:
            x = np.load(folder + f"/sol_{i}.npy", allow_pickle=True)
            x = np.atleast_1d(x)[0]
            LD.W = W[i]
            c[i], AR[i], re[i], Re[i], cl[i], CL[i], cdp[i], CDp[i], alpha_i[i], CDi[i], L_D[i] = LD.metrics(x)
            n+=1
        except FileNotFoundError:
            pass

    if n < Nw:
        W   = W  [:n]
        c   = c  [:n]
        AR  = AR [:n]
        re  = re [:n]
        Re  = Re [:n]
        cl  = cl [:n]
        CL  = CL [:n]
        cdp = cdp[:n]
        CDp = CDp[:n]
        CDi = CDi[:n]
        L_D = L_D[:n]

    return W, c, AR, re, Re, cl, CL, cdp, CDp, alpha_i, CDi, L_D


def planshape(c, LD, Nw):
    ypts = np.zeros(LD.Ny * 4 - 1)
    ypts[:LD.Ny] = np.squeeze(LD.y_pts)
    ypts[LD.Ny:2*LD.Ny-1] = -np.flip(LD.y_pts.squeeze())[1:]
    ypts[2*LD.Ny-1:4*LD.Ny-2] = np.flip(ypts[:2*LD.Ny-1])
    ypts[-1] = ypts[0]

    chord = np.zeros((Nw, LD.Ny * 4 - 1))
    chord[:, :LD.Ny] = c/4
    chord[:, LD.Ny:2*LD.Ny-1] = np.flip(c/4,1)[:, 1:]
    chord[:, 2*LD.Ny-1:4*LD.Ny-2] = -3 * np.flip(chord[:, :2*LD.Ny-1],1)
    chord[:, -1] = chord[:, 0]
    return ypts, chord


def plot_val(ycol, xcol, ycolname=None, xcolname="Weight (N)",
            label=None, fig=None, style=None, colorbar=False, 
             adapt_ticks=False, color='k'):
    if fig:
        f = fig
        a = fig.axes[0]
    else:
        f,a = plt.subplots(figsize=(6,6))
    
    label = label if label else ""
    
    a.plot(xcol, ycol, '-o', markersize=5, linewidth=2, label=label, color=color)
    
    a.grid(True)    
    a.set_xlabel(xcolname, size=20)
    a.set_ylabel(ycolname if ycolname else '', size=20)
    a.tick_params(labelsize=15)
    f.tight_layout()
    return f


def plot_dist( ycol, ccol, ycolname=None, 
                xcol=LD.y_pts, xcolname="spanwise distance y",
                ccolname="Weight (N)",
                OYsym=False, ybounds=None,flagout=True,
                label=None, fig=None, style=None, 
                colorbar=False, adapt_ticks=False, plotoptions={}):
    
    options = {"linewidth":1}
    options.update(plotoptions)
    
    if fig:
        f = fig
        a = fig.axes[0]
    else:
        f,a = plt.subplots(figsize=(10,6))
    
    label = label if label else ""
    Nc = ycol.shape[0]
    norm = mpl.colors.Normalize(vmin=ccol.min(), vmax=ccol.max())
    ccol_map = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)

            
    if ybounds is not None:
        a.set_ylim(ybounds)
    
    if OYsym:
        ycol = np.hstack((ycol,np.flip(ycol[:, :-1],1)))
        xcol = np.vstack((xcol,-np.flip(xcol[:-1])))
        
    for i in range(Nc):
        
        linewidth = 1
        if ybounds is not None and flagout:
            if not np.logical_and((ycol[i] < ybounds[1]).all(),                          (ycol[i] > ybounds[0]).all()):
                linewidth = 8
        
        a.plot(xcol, ycol[i], style if style else "-+", 
                label=f"{label} {ccolname}: {ccol[i]:.2f}", 
                color=ccol_map.to_rgba(ccol[i]),
                  linewidth=linewidth)
    
    a.grid(True)    
    a.set_xlabel(xcolname, size=20)
    a.set_ylabel(ycolname if ycolname else "", size=20)
    if colorbar:
        if adapt_ticks:
            keep_tick = np.ones_like(ccol) == 1
            keep_tick[1:] = np.diff(ccol)/ccol[1:] > 0.05
            tks = ccol[keep_tick]
        else:
            tks = None
        cb = f.colorbar(ccol_map, orientation='vertical', ticks=tks)
        cb.ax.set_ylabel(ccolname, size=20)
        cb.ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    else:
        a.legend(fontsize=10)
    a.tick_params(labelsize=15)
    f.tight_layout()
    return f

parser = argparse.ArgumentParser()
parser.add_argument('--folder', '-f', type=str, default='data/',
                    help='data folder')
parser.add_argument('--savedir', '-s', type=str, default='images/',
                    help='saving folder')
parser.add_argument('--n1', '-n1', type=int, default=0,
                    help="initial weight index")
parser.add_argument('--n2', '-n2', type=int, default=int(1e5),
                    help="last weight index")

args = parser.parse_args()


if __name__ == "__main__":

    folder = args.folder
    savedir = args.savedir
    n1 = args.n1
    n2 = args.n2

    W, c, AR, re, Re, cl, CL, cdp, CDp, alpha_i, CDi, L_D = get_data(LD, folder)

    W = W[n1:n2]
    c = c[n1:n2]
    # A = A[n1:n2]
    # Cw_AR = Cw_AR[n1:n2]
    # A1 = A1[n1:n2]
    Re = Re[n1:n2]
    AR = AR[n1:n2]
    cdp = cdp[n1:n2]
    CDp = CDp[n1:n2]
    cl = cl[n1:n2]
    CL = CL[n1:n2]
    CDi = CDi[n1:n2]
    L_D = L_D[n1:n2]
    re = re[n1:n2]

    # W
    plot_val(ycol=W, xcol=np.arange(W.shape[0]), ycolname="Weight (N)")
    plt.savefig(savedir + 'W.png', bbox_inches='tight')
    plt.close()

    # L_D
    plot_val(ycol=L_D, xcol=W, ycolname="Lift to Drag Ratio")
    plt.savefig(savedir + 'L_D.png', bbox_inches='tight')
    plt.close()

    # AR
    plot_val(ycol=AR, xcol=W, ycolname="Aspect Ratio");
    plt.savefig(savedir + 'AR.png', bbox_inches='tight')
    plt.close()

    # Re
    plot_val(ycol=Re, xcol=W, ycolname="Re");
    plt.savefig(savedir + 'Re.png', bbox_inches='tight')
    plt.close()

    # CD
    plot_val(CDi + CDp, W, "$C_D$");
    plt.savefig(savedir + 'CD.png', bbox_inches='tight')
    plt.close()

    # CDi and
    plot_val(CDi, W, "$C_{D_i}$");
    plt.savefig(savedir + 'CDi.png', bbox_inches='tight')
    plt.close()

    # CDp
    plot_val(CDp, W, "$C_{D}$");
    plt.savefig(savedir + 'CDp.png', bbox_inches='tight')
    plt.close()

    # CL
    plot_val(CL, W, "$C_L$");
    plt.savefig(savedir + 'CL.png', bbox_inches='tight')
    plt.close()

    # Max/Min chord
    n1 = 0
    n2 = 1000000
    c_min = np.min(c, 1)
    plot_val(c_min, xcol=W, ycolname="Min Chord");
    plt.savefig(savedir + 'minChord.png', bbox_inches='tight')
    plt.close()

    c_max = np.max(c, 1)
    plot_val(c_max, xcol=W, ycolname="Max Chord");
    plt.savefig(savedir + 'maxChord.png', bbox_inches='tight')
    plt.close()

    # Max/Min Re
    Re_min = np.min(re, 1)
    plot_val(Re_min, xcol=W, ycolname="Min Re");
    plt.savefig(savedir + 'minRe.png', bbox_inches='tight')
    plt.close()

    Re_max = np.max(re, 1)
    plot_val(Re_max, xcol=W, ycolname="Max Re");
    plt.savefig(savedir + 'maxRe.png', bbox_inches='tight')
    plt.close()

    # Min Cl
    cl_min = np.min(cl[:,1:], 1)
    plot_val(cl_min, W, ycolname="Min cl");
    plt.savefig(savedir + 'minCl.png', bbox_inches='tight')
    plt.close()

    # Max CL
    cl_max = np.max(cl, 1)
    plot_val(cl_max, W, ycolname="Max cl");
    plt.savefig(savedir + 'maxCl.png', bbox_inches='tight')
    plt.close()

    # lift distr
    plot_dist((cl * c  * (AR/CL)[:, None]), ccol=W, OYsym=True, ycolname="normalized lift distribution", colorbar=True);
    plt.savefig(savedir + 'lift.png', bbox_inches='tight')
    plt.close()

    # lift distr
    # plot_dist(A, ccol=W, ycolname="A", xcol=np.arange(2, 2+LD.Na)+1, style='-o', colorbar=True);
    # plt.savefig(savedir + 'A.png', bbox_inches='tight')
    # plt.close()

    # cl distr
    plot_dist(cl, 
            ccol=W,
            ybounds=[-0.5, 1.2], 
            flagout=True,
            OYsym=True, 
            colorbar=True, 
            adapt_ticks=True,
            ycolname='$c_l$')
    plt.savefig(savedir + 'cldist.png', bbox_inches='tight')
    plt.close()

    # Re distr
    plot_dist(re, ccol=W, OYsym=True, ycolname="Re", colorbar=True)
    plt.savefig(savedir + 'redist.png', bbox_inches='tight')
    plt.close()

    # c distr
    plot_dist(c, ccol=W, OYsym=True, ycolname="cb/S", colorbar=True)
    plt.savefig(savedir  + 'c.png', bbox_inches='tight')
    plt.close()

    # Planshape
    planshape_abs, planshape_ord = planshape(c, LD, W.shape[0])
    plot_dist(planshape_ord, 
            ccol = W,
            xcol=planshape_abs,
            style='-',
            colorbar=True, 
            adapt_ticks=True)
    plt.title('Planform shape', size=20)
    plt.savefig(savedir + "planform.png")
    plt.close()

    # # Contour plot of 2D section fun
    # N = 500
    # ccl = np.linspace(np.min(CL), np.max(CL), N)
    # rre = np.linspace(np.min(Re), np.max(Re), N)
    # ccl, rre = np.meshgrid(ccl, rre)
    # cdp = LD.cd0_value(ccl.flatten(), rre.flatten())
    # ccl = ccl.reshape(N,N)
    # rre = rre.reshape(N,N)
    # cdp = cdp.reshape(N,N)


    # f = plot_val(Re, CL, xcolname='$C_L$', ycolname='Re')
    # a = f.axes[0]
    # a.grid(False)
    # cmap = a.contourf(ccl, rre, cdp, levels=int(N/5))
    # a.ticklabel_format(useOffset=False)

    # cb = f.colorbar(cmap)
    # cb.ax.set_ylabel("$C_{D_p}$", size=20)
    # cb.ax.tick_params(labelsize=15)

    # a.set_title("Contours of 2D Section Drag", size=15)

    # for n in range(5):
    #     i = (W.shape[0] // 5) * n
    #     a.annotate(f"W = {W[i]:.1f} N", (CL[i], Re[i]), color='w', size=10)
    # f.savefig(savedir + '2Dfun.png')



