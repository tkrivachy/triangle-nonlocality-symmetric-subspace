# Author: Victor Gitton
#
# This python script generates the largest analytical triangle-local region
# of the fully symmetric distributions with 4 outcomes per party.
#
# Run as: python3 plot_full_inner_approx.py

import matplotlib.pyplot as plt
import numpy as np

scatter_xs = []
scatter_ys = []

def get_xy(s111,s112):
    s123 = 1. - s111 - s112
    x = 0.5*s111 + s123
    y = np.sqrt(3)/2 * s111
    return (x,y)

def scatter_point(s111,s112):
    x,y = get_xy(s111,s112)
    scatter_xs.append(x)
    scatter_ys.append(y)

# Draw main family of local distributions (no current required)
for r in np.arange(0.,1.000,0.02):
    for eta in np.arange(0.,1.000,0.02):
        for nu in np.arange(0.,0.9999,0.02):
			
            q = (1.-r)/3. + (nu/(1.-nu)) * (4.*eta - 1.)/3.

            if q <= 0.5 and q >= 0.:
                s111 = (  r*(1. - nu)          + eta*nu    ) / 4.
                s112 = 3.*( (1. - nu)*(1. - r) + 3.*eta*nu ) / 4.
                scatter_point(s111,s112)

# Draw little anti-correlated line + currents
for r in np.arange(0.,1.0001,0.05):
    s111 = r/48.
    s112 = (12. - 3.*r) / 48.

    scatter_point(s111,s112)

    # Draw the currents

    # l controls the q0_left/q0_right line
    # l = 0 means q0_left
    # l = 1 means q0_right
    for l in np.arange(0.,1.,0.05):
        q0_111 = 0.25 * (1. - l) + 0.0 * l
        q0_112 = 0.75 * (1. - l) + 0.5 * l

        for eps in np.arange(0.,1.,0.01):
            s111_prime = (1.-eps)**3 * s111 + ( 3. * eps*(1.-eps) + 0.75 * eps**3 ) * q0_111 + 0.25 * (eps**3) * 4. / 64.
            s112_prime = (1.-eps)**3 * s112 + ( 3. * eps*(1.-eps) + 0.75 * eps**3 ) * q0_112 + 0.25 * (eps**3) * 36. / 64.

            scatter_point(s111_prime,s112_prime)
	
# Tamas plot style
fig, axs = plt.subplots(1, 1, figsize=(14,8),dpi=150)
axs.set_aspect('equal', 'box')

plt.scatter(scatter_xs,scatter_ys,s=1)

# Tamas plot style continued
plt.plot([0,0.5,1,0],[0,np.sqrt(3)/2,0,0],alpha=0.2,c='k')
x_id,y_id = get_xy(1/16,9/16)
x_el,y_el = get_xy(25/64,9/64)
plt.scatter(x_id,y_id,label='Identity distr.',marker='*',color='r')
plt.scatter(x_el,y_el,label='Elegant distr.',marker='o',color='r')
plt.axhline(y=np.sqrt(3)/2 * 1/4,label=r'$s_{111} = 1/4$', c='m')
plt.axhline(y=np.sqrt(3)/2 * 1/2,label=r'$s_{111} = 1/2$ (Finner)', c='b')

def elegant_vis(vis):
    """ Recreating the elegant distribution with visibility v (vis) in each singlet. See https://www.nature.com/articles/s41534-020-00305-x for details."""
    p = np.array([1/256 *(4+9 *vis+9 *vis**2+3 *vis**3),1/256 *(4+vis-3 *vis**2-vis**3),1/256 *(4+vis-3 *vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+9*vis+9*vis**2+3*vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+9*vis+9*vis**2+3*vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+9*vis+9*vis**2+3*vis**3)])
    return p
vises = np.linspace(0,1,1000)
elegant_vises = np.array([elegant_vis(vis) for vis in vises])
# critical visibility is around 0.8
crit_index_vis = int(vises.shape[0]*0.8)
crit_elegant_distr_vis = elegant_vis(crit_index_vis/vises.shape[0])
crit_elegant_distr_xy = get_xy(4*crit_elegant_distr_vis[0],36*crit_elegant_distr_vis[1])
elegant_vises_xys = np.array([get_xy(4*elegant_vis[0],36*elegant_vis[1]) for elegant_vis in elegant_vises])

plt.plot(elegant_vises_xys[:,0],elegant_vises_xys[:,1],label='Noisy Elegant (vis.)',c='r', alpha=0.2)
plt.scatter(crit_elegant_distr_xy[0],crit_elegant_distr_xy[1],label='Critical visibility of 0.8',marker='s',color='r',s=40)

plt.legend()

plt.title("The largest analytically local region of the symmetric supbspace that we found")
fs = 14
plt.text(0.5-0.02, np.sqrt(3)/2-0.07, r'$s_{111}$', fontsize=fs)
plt.text(0-0.04,0.07 , r'$s_{112}$', fontsize=fs)
plt.text(1-0.02, 0.07, r'$s_{123}$', fontsize=fs)
plt.savefig("./plots/full_inner_approx.png",bbox_inches='tight')

