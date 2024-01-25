import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

"""
This script evaluates f_w(pabc) - f_w(p_elegant) for several distributions and
plots them together with the maximum of this expression as found by LHV-Net.

Moreover, with the help of the function f_w_minus_f_w_Elegant, we can evaluate
the inequality's main expression for any distribution pabc and any w, l (use verbose=True
to see the average values and penalties used in the inequality).
Together with the delta_w values, we can then check if the inequality is violated
for a given distribution pabc and w,l.
"""

# These indices help in writing the inequality compactly 
samevalues = []
one_differents = []
all_differents = []
for o1 in range(4):
    samevalues.append((o1,o1,o1))
    for o2 in range(4):
        if o1!=o2:
            one_differents.append((o1,o1,o2))
            one_differents.append((o1,o2,o1))
            one_differents.append((o2,o1,o1))
            for o3 in range(4):
                if o1!=o3 and o2!=o3:
                    all_differents.append((o1,o2,o3))


# Load delta_w's and w's
ws_l1_coarse = np.loadtxt("./data/ineq-delta_w-vs-w/ws_l1_coarse.txt")
delta_ws_l1_coarse = np.loadtxt("./data/ineq-delta_w-vs-w/delta_ws_l1_coarse.txt")

ws_l1_fine = np.loadtxt("./data/ineq-delta_w-vs-w/ws_l1_fine.txt")
delta_ws_l1_fine = np.loadtxt("./data/ineq-delta_w-vs-w/delta_ws_l1_fine.txt")


ws_l2_coarse = np.loadtxt("./data/ineq-delta_w-vs-w/ws_l2_coarse.txt")
delta_ws_l2_coarse = np.loadtxt("./data/ineq-delta_w-vs-w/delta_ws_l2_coarse.txt")

ws_l2_fine = np.loadtxt("./data/ineq-delta_w-vs-w/ws_l2_fine.txt")
delta_ws_l2_fine = np.loadtxt("./data/ineq-delta_w-vs-w/delta_ws_l2_fine.txt")

# Define important distributions
## 'Squares' distribution (non-symmetric, but high s_111)
p_squares = np.zeros((4,4,4))
p_squares[0,0,0] = 1/8
p_squares[1,1,1] = 1/8
p_squares[2,2,2] = 1/8
p_squares[3,3,3] = 1/8
p_squares[0,3,2] = 1/8
p_squares[1,2,3] = 1/8
p_squares[2,1,0] = 1/8
p_squares[3,0,1] = 1/8

## Aways outputing 0,0,0 distribution
p_1 = np.zeros((4,4,4))
p_1[0,0,0] = 1

## One of the s_111 = 0.25, symmetric distributions
p_111_is_025 = np.zeros((4,4,4))
for index_tuple in samevalues:
    p_111_is_025[index_tuple] = 0.25/4
for index_tuple in one_differents:
    p_111_is_025[index_tuple] = 0
for index_tuple in all_differents:
    p_111_is_025[index_tuple] = 0.75/24

## s_111 > 0.25, almost symmetric distribution from LHV-Net
p_high_s_111 = np.loadtxt("./data/probs_high111.csv")
p_high_s_111 = np.reshape(p_high_s_111,(4,4,4))

## Assert all these are normalized to 1
assert(np.sum(p_squares)==1), "p_squares not normalized. It sums to {}".format(np.sum(p_squares))
assert(np.sum(p_1)==1), "p_1 not normalized. It sums to {}".format(np.sum(p_1))
assert(np.sum(p_111_is_025)==1), "p_111_is_025 not normalized. It sums to {}".format(np.sum(p_111_is_025))
assert(np.isclose(np.sum(p_high_s_111),1)), "p_high_s_111 not normalized. It sums to {}".format(np.sum(p_high_s_111))

# Define the function use in the inequality
def f_w_minus_f_w_Elegant(pabc, l, w, verbose=False):
    """ Evaluates f_w(pabc) - f_w(p_elegant). Should be less than -delta_w for all local models. If more, then NL."""
    # Make sure probability normalized
    assert(np.isclose(np.sum(pabc),1)), "pabc not normalized. It sums to {}".format(np.sum(pabc))

    # Create permutation penalty term
    M_111 = np.mean(np.stack([pabc[index_tuple] for index_tuple in samevalues],axis=0),axis=0)
    M_112 = np.mean(np.stack([pabc[index_tuple] for index_tuple in one_differents],axis=0),axis=0)
    M_123 = np.mean(np.stack([pabc[index_tuple] for index_tuple in all_differents],axis=0),axis=0)

    Delta_111 = np.sum(np.stack([np.abs(M_111 - pabc[index_tuple])**l for index_tuple in samevalues],axis=0),axis=0)
    Delta_112 = np.sum(np.stack([np.abs(M_112 - pabc[index_tuple])**l for index_tuple in one_differents],axis=0),axis=0)
    Delta_123 = np.sum(np.stack([np.abs(M_123 - pabc[index_tuple])**l for index_tuple in all_differents],axis=0),axis=0)

    # s_111 part:
    s_111 = pabc[0,0,0] + pabc[1,1,1] + pabc[2,2,2] + pabc[3,3,3]

    # asymmetry penalty:
    Delta = Delta_111 + Delta_112 + Delta_123

    expression = w*s_111 - (1-w)* Delta - w*(100/256)

    if verbose:
        print("Evaluating inequality for l={}, w={}".format(l,w))
        print("Averages:")
        print("M_111:",M_111)
        print("M_112:",M_112)
        print("M_123:",M_123)
        print()
        print("Penalties:")
        print("Delta_111:",Delta_111)
        print("Delta_112:",Delta_112)
        print("Delta_123:",Delta_123)
        print()
        print("s_111:",s_111)
        print("Delta:", Delta)
    
    return expression

# For each ws dataset, calculate f_w_minus_f_w_Elegant for each w for each distribution
p_squares_values_l2_coarse = np.array([f_w_minus_f_w_Elegant(p_squares, l=2, w=w) for w in ws_l2_coarse])
p_squares_values_l2_fine = np.array([f_w_minus_f_w_Elegant(p_squares, l=2, w=w) for w in ws_l2_fine])
p_squares_values_l1_coarse = np.array([f_w_minus_f_w_Elegant(p_squares, l=1, w=w) for w in ws_l1_coarse])
p_squares_values_l1_fine = np.array([f_w_minus_f_w_Elegant(p_squares, l=1, w=w) for w in ws_l1_fine])

p_1_values_l2_coarse = np.array([f_w_minus_f_w_Elegant(p_1, l=2, w=w) for w in ws_l2_coarse])
p_1_values_l2_fine = np.array([f_w_minus_f_w_Elegant(p_1, l=2, w=w) for w in ws_l2_fine])
p_1_values_l1_coarse = np.array([f_w_minus_f_w_Elegant(p_1, l=1, w=w) for w in ws_l1_coarse])
p_1_values_l1_fine = np.array([f_w_minus_f_w_Elegant(p_1, l=1, w=w) for w in ws_l1_fine])

p_111_is_025_values_l2_coarse = np.array([f_w_minus_f_w_Elegant(p_111_is_025, l=2, w=w) for w in ws_l2_coarse])
p_111_is_025_values_l2_fine = np.array([f_w_minus_f_w_Elegant(p_111_is_025, l=2, w=w) for w in ws_l2_fine])
p_111_is_025_values_l1_coarse = np.array([f_w_minus_f_w_Elegant(p_111_is_025, l=1, w=w) for w in ws_l1_coarse])
p_111_is_025_values_l1_fine = np.array([f_w_minus_f_w_Elegant(p_111_is_025, l=1, w=w) for w in ws_l1_fine])

p_high_s_111_values_l2_coarse = np.array([f_w_minus_f_w_Elegant(p_high_s_111, l=2, w=w) for w in ws_l2_coarse])
p_high_s_111_values_l2_fine = np.array([f_w_minus_f_w_Elegant(p_high_s_111, l=2, w=w) for w in ws_l2_fine])
p_high_s_111_values_l1_coarse = np.array([f_w_minus_f_w_Elegant(p_high_s_111, l=1, w=w) for w in ws_l1_coarse])
p_high_s_111_values_l1_fine = np.array([f_w_minus_f_w_Elegant(p_high_s_111, l=1, w=w) for w in ws_l1_fine])

# Plot the results
# l=2
optimal_w = 0.1613
plt.figure(figsize=(5, 4))
ax = plt.subplot(111)

ax.axhline(0,color='grey')

ax.plot(ws_l2_coarse,-1*delta_ws_l2_coarse,'+-',label='LHV-Net',color='red',ms=7)
ax.plot(ws_l2_coarse,p_111_is_025_values_l2_coarse, label=r'$s_{111}$=0.25',linestyle='--',alpha=0.8, color='#4d0180')
ax.plot(ws_l2_coarse,p_high_s_111_values_l2_coarse, label=r'$s_{111}\approx 0.289$',linestyle='-.',alpha=0.8, color='#005719')
ax.plot(ws_l2_coarse,p_squares_values_l2_coarse, label='Squares',linestyle=':',alpha=0.8, color='#4d0180')
ax.plot(ws_l2_coarse,p_1_values_l2_coarse, label=r'$p(111)=1$',dashes=[2.5,6],alpha=0.8, color = '#4d0180')
ax.axvline(optimal_w)

ax.set_ylim(bottom=-0.2)

## Inset for l=2
axins = inset_axes(ax, width=1.6, height=1.3, loc='lower left',
                                bbox_to_anchor=(0.13,0.32,.4,.4), bbox_transform=ax.transAxes)
axins.axhline(y=0,color='grey')
axins.plot(ws_l2_fine,-1*delta_ws_l2_fine,'+-',label='LHV-Net',color='red',ms=7)
axins.plot(ws_l2_fine,p_111_is_025_values_l2_fine, label=r'$s_{111}$=0.25',linestyle='--',alpha=0.8, color='#4d0180')
axins.plot(ws_l2_fine,p_high_s_111_values_l2_fine, label=r'$s_{111}\approx 0.289$',linestyle='-.',alpha=0.8, color='#005719')
axins.plot(ws_l2_fine,p_squares_values_l2_fine, label='Squares',linestyle=':',alpha=0.8, color='#4d0180')
plt.axvline(optimal_w)

axins.axis(ymin=-0.03)
axins.axis(ymax = 0.02)

ax.legend(loc = 'upper right')
ax.set_xlabel(r'$w$')
ax.set_ylabel(r'$f_w(p)-f_w(p_E)$')
ax.set_title(r'$l=2$')

plt.savefig("./plots/ineq_l2.png", bbox_inches="tight", dpi=300)


# l=1
optimal_w = 0.6784
plt.figure(figsize=(5, 4))
ax = plt.subplot(111)

ax.axhline(0,color='grey')

ax.plot(ws_l1_coarse,-1*delta_ws_l1_coarse,'+-',label='LHV-Net',color='red',ms=7)
ax.plot(ws_l1_coarse,p_111_is_025_values_l1_coarse, label=r'$s_{111}$=0.25',linestyle='--',alpha=0.8, color='#4d0180')
ax.plot(ws_l1_coarse,p_high_s_111_values_l1_coarse, label=r'$s_{111}\approx 0.289$',linestyle='-.',alpha=0.8, color='#005719')
ax.plot(ws_l1_coarse,p_1_values_l1_coarse, label=r'$p(111)=1$',dashes=[2.5,6],alpha=0.8, color = '#4d0180')
ax.axvline(optimal_w)

ax.set_ylim(bottom=-0.2)

## Inset for l=1
axins = inset_axes(ax, width=1.6, height=1.3, loc='lower left',
                                bbox_to_anchor=(0.1,0.3,.4,.4), bbox_transform=ax.transAxes)
axins.axhline(y=0,color='grey')
axins.plot(ws_l1_fine,-1*delta_ws_l1_fine,'+-',label='LHV-Net',color='red',ms=7)
axins.plot(ws_l1_fine,p_111_is_025_values_l1_fine, label=r'$s_{111}$=0.25',linestyle='--',alpha=0.8, color='#4d0180')
axins.plot(ws_l1_fine,p_high_s_111_values_l1_fine, label=r'$s_{111}\approx 0.289$',linestyle='-.',alpha=0.8, color='#005719')
axins.plot(ws_l1_fine,p_1_values_l1_fine, label=r'$p(111)=1$',dashes=[2.5,6],alpha=0.8, color = '#4d0180')
axins.axvline(optimal_w)

axins.axis(ymin=-0.15)

ax.legend(loc = 'upper right')
ax.set_xlabel(r'$w$')
ax.set_ylabel(r'$f_w(p)-f_w(p_E)$')
ax.set_title(r'$l=1$')

plt.savefig("./plots/ineq_l1.png", bbox_inches="tight", dpi=300)
