# triangle-nonlocality-symmetric-subspace
Computational appendix for the work on exploring local hidden variable models in the fully symmetric subspace of the triangle network.

## Data and code for the NN-inspired Bell-type inequality
The ```data``` folder include w, delta_w pairs LHV-Net found for l=1,2. Recall the inequality is

$f_w(p) - f_w(p_{elegant}) \leq \delta_w$,

with:

$f_w(p) = w s_{111}(p) - (1-w) \Delta_l(p)$,

where $\Delta_l(p)$ is a penalty for being non-symmetric and $s_{111}$ captures the strength of (1,1,1)-type outcomes. The parameter $w$ balances these two. Details in the paper. 

The python script ```ineq_finding_best_w_plots.py``` evaluates $f_w(p) - f_w(p_{elegant})$ for several distributions and plots them together with the maximum of this expression as found by LHV-Net.
Moreover, with the help of the ```function f_w_minus_f_w_Elegant```, we can evaluate the inequality's main expression for any distribution pabc and any w, l (use verbose=True to see the average values and penalties used in the inequality). Together with the delta_w values, we can then check if the inequality is violated for a given distribution pabc and w,l.


## Data for almost symmetric classical strategy with s_111 > 0.25
The distribution can be found in ```data/probs_high111.csv```, in lexicographic order (0,0,0; 0,0,1; ... 3,3,3). np.reshape can reshape it into a (4,4,4) tensor if necessary.

The flags (forced to be determistic and discretized in a 100x100 grid), can be found in ```data/flags_high111.json```. An example python script for loading and plotting these flags, as well as analyzing the distribution, is provided (```flag_plotter_high111.py```)


## Mathematica code 
