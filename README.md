# triangle-nonlocality-symmetric-subspace
Computational appendix for the work on exploring local hidden variable models in the fully symmetric subspace of the triangle network.

## Data and code for the NN-inspired Bell-type inequality
The data folder include w, delta_w pairs LHV-Net found for l=1,2. Recall the inequality is
$f_w(p) - f_w(p_elegant) <= delta_w$
$f_w(p) = w * s_{111}(p) - (1-w) \Delta_l(p)$

The python script ineq_finding_best_w_plots.py evaluates f_w(pabc) - f_w(p_elegant) for several distributions and
plots them together with the maximum of this expression as found by LHV-Net.

Moreover, with the help of the function f_w_minus_f_w_Elegant, we can evaluate
the inequality's main expression for any distribution pabc and any w, l (use verbose=True
to see the average values and penalties used in the inequality).
Together with the delta_w values, we can then check if the inequality is violated
for a given distribution pabc and w,l.


## Data for almost symmetric classical strategy with s_111 > 0.25
Include data and a python script to plot it.


## Mathematica code 
