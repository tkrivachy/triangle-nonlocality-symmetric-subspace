"""
This script loads and plots the lowest Euclidean distances that LHV-Net found for 260 points in the symmetric subspace.
Color scales are standardized to be from 0 to 0.1 in the viridis colorbar. (Everything above 0.1 is set to 0.1).
In essence, low distances (dark blue) are in or close to the local set, while brighter colors (yellow) are further away.
For card=4, the Elegant distribution is also plotted. Moreover, its noisy version is plotted as well (visibility at sources), which is 
conjectured to have a crticial visibility of 0.8. (see details in https://www.nature.com/articles/s41534-020-00305-x)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.set_printoptions(linewidth=200)

# Set cardinality (3,4,5 or 6)
card = 6

# Load data
all_coordinates_lin = np.loadtxt("./data/symmetric_subspace_maps/sym_subspace_map_card_"+str(card)+".csv",delimiter=',',skiprows=1)

print("Data loaded with shape:", all_coordinates_lin.shape)

# Define colorbar extent to be from 0 to 0.1 in viridis colorbar
minima = 0
maxima = 0.1

norm = plt.Normalize(vmin=minima, vmax=maxima)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)

# Let's set up the symmetric subspace parameters.
# In general lambdas are the weights in the simplex. l1 = lambda_111; l2  = lambda_112; l3 = lambda_123. NOTATION: In the paper we used s_111, s_112, s_123.
# They are the sums of the probabilities that are all the same, e.g. lambda_111 = sum_i p(iii)
# The conversion between lambdas and probabilities (for 4 outcomes) is: lambda_111 = 4*p_111; lambda_112 = 36*p_112; lambda_123 = 24*p_123
# For plotting we need x and y coordinates, so we use those as well.

# Extremal points of the triangle in plot.
x111 = 0.5
y111 = np.sqrt(3)/2
x112 = 0
y112 = 0
x123 = 1
y123 = 0

# indices that are all the same in symmetric subspace, i.e. 111, 112, and 123-type indices grouped together.
all_samevalues = []
all_one_differents = []
all_all_differents = []
for i in range(7):
    samevalues = []
    one_differents = []
    all_differents = []
    for o1 in range(i):
        samevalues.append((o1,o1,o1))
        for o2 in range(i):
            if o1!=o2:
                one_differents.append((o1,o1,o2))
                one_differents.append((o1,o2,o1))
                one_differents.append((o2,o1,o1))
                for o3 in range(i):
                    if o1!=o3 and o2!=o3:
                        all_differents.append((o1,o2,o3))
    all_samevalues.append(samevalues)
    all_one_differents.append(one_differents)
    all_all_differents.append(all_differents)

samevalues = all_samevalues[card]
one_differents = all_one_differents[card]
all_differents = all_all_differents[card]

def p_from_lambdas(l1,l2,l3=None, card=4):
    """ Get probability distribution from lambdas, which are the weights in the simplex.
    l1 = lambda_111; l2  = lambda_112; l3 = lambda_123. card is cardinality of output of each party """
    l3 = 1-l1-l2
    if l3>=-1e-3:
        p = np.ones((card,card,card))
        for index in samevalues:
            p[index] = l1/card
        for index in one_differents:
            p[index] = l2/(card*(card-1)*3)
        for index in all_differents:
            p[index] = l3/(card*(card-1)*(card-2))
    else:
        # Return something stupid if I'm outside the simplex.
        p= np.ones((card,card,card))*(-1)
    return p

def p_to_xy(p111,p112,p123,card=4):
    """ Convert probabilities to x and y coordinates used in figure."""
    x = (p111*card*x111 + p112*(card*(card-1)*3)*x112 + p123*x123*(card*(card-1)*(card-2)))
    y = (p111*card*y111 + p112*(card*(card-1)*3)*y112 + p123*y123*(card*(card-1)*(card-2)))
    return [x,y]

def lambdas_to_xy(lambdas):
    """ Convert lambdas to x and y coordinates used in figure."""
    l111,l112,l123 = lambdas
    x = (l111*x111 + l112*x112 + l123*x123)
    y = (l111*y111 +  l112*y112 +  l123*y123)
    return [x,y]

xy_coordinates_lin = np.array([lambdas_to_xy(all_coordinates_lin[i,:][0:3]) for i in range(all_coordinates_lin.shape[0])])

xs = xy_coordinates_lin[:,0]
ys = xy_coordinates_lin[:,1]
colors = all_coordinates_lin[:,-1]

x_id,y_id = p_to_xy(1/card**3,1/card**3,1/card**3,card=card)
if card==4:
    x_el,y_el = p_to_xy(25/256,1/256,5/256)

if card==4:
    def elegant_vis(vis):
        """ Recreating the elegant distribution with visibility v (vis) in each singlet. See https://www.nature.com/articles/s41534-020-00305-x for details."""
        p = np.array([1/256 *(4+9 *vis+9 *vis**2+3 *vis**3),1/256 *(4+vis-3 *vis**2-vis**3),1/256 *(4+vis-3 *vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+9*vis+9*vis**2+3*vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+9*vis+9*vis**2+3*vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4-3*vis+3*vis**2+vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+vis-3*vis**2-vis**3),1/256 *(4+9*vis+9*vis**2+3*vis**3)])
        return p
    vises = np.linspace(0,1,1000)
    elegant_vises = np.array([elegant_vis(vis) for vis in vises])
    # critical visibility is around 0.8
    crit_index_vis = int(vises.shape[0]*0.8)
        
    print("Crit vis:", crit_index_vis/vises.shape[0])
    crit_elegant_distr_vis = elegant_vis(crit_index_vis/vises.shape[0])
    crit_elegant_distr_xy = p_to_xy(crit_elegant_distr_vis[0],crit_elegant_distr_vis[1],crit_elegant_distr_vis[6], card=card)
    elegant_vises_xys = np.array([p_to_xy(elegant_vis[0],elegant_vis[1],elegant_vis[6], card=card) for elegant_vis in elegant_vises])


# Plot stuff 
markersize = 425
fig, axs = plt.subplots(1, 1, figsize=(14,8),dpi=150)
axs.set_aspect('equal', 'box')

plt.plot([0,0.5,1,0],[0,np.sqrt(3)/2,0,0],alpha=0.2,c='k')

sc = plt.scatter(xs,ys, marker='h', s=markersize, c=colors, cmap='viridis', norm=norm)
plt.scatter(x_id,y_id,label='Identity distr.',marker='*',color='r')
if card==4:
    plt.scatter(x_el,y_el,label='Elegant distr.',marker='o',color='r')

plt.axhline(y=np.sqrt(3)/2 * 1/card,label=r'$s_{111} = 1/N$', c='m')
plt.axhline(y=np.sqrt(3)/2 * 1/np.sqrt(card),label=r'$s_{111} = 1/\sqrt{N}$ (Finner)', c='b')

if card ==4:
    plt.plot(elegant_vises_xys[:,0],elegant_vises_xys[:,1],label='Noisy Elegant (vis.)',c='r', alpha=0.2)
    plt.scatter(crit_elegant_distr_xy[0],crit_elegant_distr_xy[1],label='Critical visibility of 0.8',marker='s',color='r',s=40)

plt.legend()
plt.colorbar(sc)

plt.title("Symmetric subspace map for outcome cardinality "+str(card))
fs = 14
plt.text(0.5-0.02, np.sqrt(3)/2-0.07, r'$s_{111}$', fontsize=fs)
plt.text(0-0.04,0.07 , r'$s_{112}$', fontsize=fs)
plt.text(1-0.02, 0.07, r'$s_{123}$', fontsize=fs)
#plt.show()
# if triangle_lattice is in basebase then remove it from the name
plt.savefig("./plots/symmetric_subspace_map_card_"+str(card)+".png",bbox_inches='tight')
print("Plotting finished succesfully. Saved to ./plots/symmetric_subspace_map_card_"+str(card)+".png")