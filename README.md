# liftinglineopt
Lifting line computations, wrapped into an optimizer

- `airfoil_model`: Neural network fit of 2D wind tunnel data
- `LLopt`: joint solution of lifting line theory and optimization of the planshape (chord and twist distribution and span) as well as airspeed for maximum L/D
This creates a complicated optimization problem with nonlinear equality constraints. See `llopt/equations/equations.pdf`
- `lift_dist`: optimization of the lift and chord distribution, span and airspeed for maximmum L/D. The twist distribution is obtained afterwards with lifting line theory.