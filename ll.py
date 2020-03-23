import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint


def lifting_line_helper(chord, alpha, CL_2d, al_camber, spanwise_loc, N_fourrier):

    # Spanwise loc goes from -b/2 to b/2, contains the root
    assert ((spanwise_loc[0] + spanwise_loc[-1]) < 1e-9); "spanwise_loc must go from -b/2 to b/2"
    assert ((spanwise_loc[1:] - spanwise_loc[:-1]) > 0).all(); "spanwise_loc must monotonically increase"
    assert 2*int(Nloc/2) == Nloc - 1, f"spanwise_loc must have an odd length, not {Nloc}"
    root = int((Nloc - 1)/2)
    assert np.abs(spanwise_loc[root]) < 1e-9
    spanwise_loc = np.squeeze(spanwise_loc)[:, None]


    # Size
    if type(chord) != np.array or chord.shape[0] == 1:
        chord = chord * np.ones(spanwise_loc.shape[0])
    else: 
        assert (chord.shape == spanwise_loc.shape)
    assert (chord >= 0.).all(); "chord size must be positive"

    if type(CL_2d) != np.array or CL_2d.shape[0] == 1:
        CL_2d = CL_2d * np.ones(spanwise_loc.shape[0])
    else: 
        assert (CL_2d.shape == spanwise_loc.shape)

    if type(alpha) != np.array or alpha.shape[0] == 1:
        alpha = alpha * np.ones(spanwise_loc.shape[0])
    else: 
        assert (alpha.shape == spanwise_loc.shape)
    
    if type(al_camber) != np.array or al_camber.shape[0] == 1:
        al_camber = al_camber * np.ones(spanwise_loc.shape[0])
    else: 
        assert (al_camber.shape == spanwise_loc.shape)
    chord = np.squeeze(chord)[:, None]
    alpha = np.squeeze(alpha)[:, None]
    al_camber = np.squeeze(al_camber)[:, None]
    CL_2d = np.squeeze(CL_2d)[:, None]

    return chord, alpha, CL_2d, al_camber, spanwise_loc, N_fourrier


def lifting_line_fourrier(chord, alpha, CL_2d, al_camber, spanwise_loc, N_fourrier):  
    """Most basic lifting line theory.
    
    Parameters
    ----------
    chord : np.array
        Array of size Nloc with the chord of each section
    alpha : np.array
        Array of size Nloc with the angle of attack of each section
    CL_2d : np.array
        Array of size Nloc (or 1) with the 2D lift curve slope of each section
    al_camber : np.array
        Array of size Nloc (or 1) with the 0 lift angle of attack (due to camber) of each section
    spanwise_loc : np.array
        Array of size Nloc with the spanwise location of the specified alpha and chord
    N_fourrier : int
        Number of Fourrier coefficients to consider when fitting CL
    
    Returns
    -------
    CL : np.float
        Lift coefficient of the wing
    CDi : np.float
        Induced drag coefficient of the wing
    eps : np.float
        Span efficiency of the wing
    cl_span : np.array
        Array of size Nloc with the lift coefficient of each section
    A : np.array
        Array of size N_fourrier with the coefficients of the Fourrier expansion of the lift coefficient spanwise distribution
    """

    # Process input
    chord, alpha, CL_2d, al_camber, spanwise_loc, N_fourrier = lifting_line_helper(chord, alpha, CL_2d, al_camber, spanwise_loc, N_fourrier)
    Nloc = chord.shape[0]
    root = int((Nloc - 1)/2)
    cs = chord[root] * CL_2d[root]
    b = spanwise_loc[-1]*2

    # Determine geometric features
    N_alpha = alpha.shape[0]
    area =  1/2 * ((chord[1:] + chord[:-1]).T @ 
        (spanwise_loc[1:]- spanwise_loc[:-1]))
    AR = b**2 / area
    print(f"area {area}, AR {AR}, root chord {chord[root]}")

    # Create influence matrices and solce system
    N_th = chord.shape[0] - 2 # tips not useful to determine A
    theta0 = np.arccos(spanwise_loc * 2/b)
    Nn = np.arange(1, N_fourrier+1)[:, None]
    S = np.sin(theta0 @ Nn.T)[1:-1]
    C = ((chord * CL_2d) / cs * (alpha - al_camber))[1:-1]
    Sp = (S * Nn.T)  * ((chord * CL_2d) / np.sin(theta0))[1:-1] / (4*b)
    A = np.linalg.lstsq(S + Sp, C,rcond=None)[0]

    # Compute non dim forces
    CL = cs * np.pi * b / 4 / area * A[0]
    import pdb; pdb.set_trace()

    eps = 1 / ((A*Nn).T @ A / A[0]**2)
    CDi = CL**2 / np.pi / AR / eps
    cl_span = cs * (S @ A) / chord[1:-1]
    return CL, CDi, eps, cl_span, A


def plot_planform(chord, spanwise_loc):
    x_le = chord/4
    x_te = -3*chord/4
    plt.plot(spanwise_loc, x_le)
    plt.plot(spanwise_loc, x_te)
    plt.axis('equal')
    plt.show()
    return


def nonlin_lifting_line_fourrier(chord, alpha, CL_2d, al_camber, spanwise_loc, N_fourrier):  
    """Most basic lifting line theory.
    
    Parameters
    ----------
    chord : np.array
        Array of size Nloc with the chord of each section
    alpha : np.array
        Array of size Nloc with the angle of attack of each section
    CL_2d : np.array
        Array of size Nloc (or 1) with the 2D lift curve slope of each section
    al_camber : np.array
        Array of size Nloc (or 1) with the 0 lift angle of attack (due to camber) of each section
    spanwise_loc : np.array
        Array of size Nloc with the spanwise location of the specified alpha and chord
    N_fourrier : int
        Number of Fourrier coefficients to consider when fitting CL
    
    Returns
    -------
    CL : np.float
        Lift coefficient of the wing
    CDi : np.float
        Induced drag coefficient of the wing
    eps : np.float
        Span efficiency of the wing
    cl_span : np.array
        Array of size Nloc with the lift coefficient of each section
    A : np.array
        Array of size N_fourrier with the coefficients of the Fourrier expansion of the lift coefficient spanwise distribution
    """

    # Process input
    Nloc = chord.shape[0]
    root = int((Nloc - 1)/2)
    cs = chord[root] * CL_2d[root]
    b = spanwise_loc[-1]*2

    # Determine geometric features
    N_alpha = alpha.shape[0]
    area =  1/2 * ((chord[1:] + chord[:-1]).T @ 
        (spanwise_loc[1:]- spanwise_loc[:-1]))
    AR = b**2 / area
    print(f"area {area}, AR {AR}, root chord {chord[root]}")

    # Create influence matrices and solce system
    N_th = chord.shape[0] - 2 # tips not useful to determine A
    theta0 = np.arccos(spanwise_loc * 2/b)
    Nn = np.arange(1, N_fourrier+1)[:, None]
    S = np.sin(theta0 @ Nn.T)[1:-1]
    C = ((chord * CL_2d) / cs * (alpha - al_camber))[1:-1]
    Sp = (S * Nn.T)  * ((chord * CL_2d) / np.sin(theta0))[1:-1] / (4*b)
    A = np.linalg.lstsq(S + Sp, C,rcond=None)[0]

    # Compute non dim forces
    CL = cs * np.pi * b / 4 / area * A[0]

    eps = 1 / ((A*Nn).T @ A / A[0]**2)
    CDi = CL**2 / np.pi / AR / eps
    cl_span = cs * (S @ A) / chord[1:-1]
    return CL, CDi, eps, cl_span, A




def lifting_line_disc(chord, alpha, CL_2d, al_camber, spanwise_loc):  
    """Most basic lifting line theory.
    
    Parameters
    ----------
    chord : np.array
        Array of size Nloc with the chord of each section
    alpha : np.array
        Array of size Nloc with the angle of attack of each section
    CL_2d : np.array
        Array of size Nloc (or 1) with the 2D lift curve slope of each section
    al_camber : np.array
        Array of size Nloc (or 1) with the 0 lift angle of attack (due to camber) of each section
    spanwise_loc : np.array
        Array of size Nloc with the spanwise location of the specified alpha and chord
    
    Returns
    -------
    CL : np.float
        Lift coefficient of the wing
    CDi : np.float
        Induced drag coefficient of the wing
    eps : np.float
        Span efficiency of the wing
    cl_span : np.array
        Array of size Nloc with the lift coefficient of each section
    """

    # Process input
    chord, alpha, CL_2d, al_camber, spanwise_loc, N_fourrier = lifting_line_helper(chord, alpha, CL_2d, al_camber, spanwise_loc, N_fourrier)
    Nloc = chord.shape[0]
    root = int((Nloc - 1)/2)
    cs = chord[root] * CL_2d[root]
    b = spanwise_loc[-1]*2

    # Determine geometric features
    N_alpha = alpha.shape[0]
    area =  1/2 * ((chord[1:] + chord[:-1]).T @ 
        (spanwise_loc[1:]- spanwise_loc[:-1]))
    AR = b**2 / area
    print(f"area {area}, AR {AR}, root chord {chord[root]}")

    # Define x
    # X = cl_1:Nloc, alpha_i_1:Nloc
    x0 = np.hstack(np.ones(Nloc), np.zeros(Nloc))
    al_i = np.zeros(Nloc)
    def alpha_i(x):
        cl = x[:Nloc]
        cf = 1 / (8 * np.pi)
        ccl = chord * cl
        for y in range(1, Nloc-1):
            ccl_ = ccl / (spanwise_loc[y] - spanwise_loc)
            al_i[y] = ccl[0] / (spanwise_loc[y]-b/2) - ccl[-1] / (spanwise_loc[y]+b/2) - (
                    (ccl_[1:] + ccl[:-1]).T @ (spanwise_loc[1:]- spanwise_loc[:-1])**2)
        al_i[0] = 
        al_i[-1] = 
    ncon = NonlinearConstraint()


    # Compute non dim forces
    CL = cs * np.pi * b / 4 / area * A[0]
    import pdb; pdb.set_trace()

    eps = 1 / ((A*Nn).T @ A / A[0]**2)
    CDi = CL**2 / np.pi / AR / eps
    cl_span = cs * (S @ A) / chord[1:-1]
    return CL, CDi, eps, cl_span, A


if __name__ == "__main__":
    V = 1
    rho = 1.225
    b = 1000
    cs  = 1
    # chord_fn = lambda y: cs * np.sqrt(1 - (y/b*2)**2)
    chord_fn = lambda y: np.ones(y.shape)*cs
    al_camber = 0
    CL_2d = 2 * np.pi
    Nloc = 101
    spanwise_loc = np.linspace(-b/2, b/2, Nloc)
    CL, CDi, eps, cl_span, A = lifting_line(
        alpha=np.array([0.1]),
        chord = chord_fn(spanwise_loc),
        spanwise_loc = spanwise_loc,
        CL_2d = CL_2d,
        al_camber = al_camber,
        N_fourrier=10
    )
    import pdb; pdb.set_trace()
    print(CL, CDi, eps)
