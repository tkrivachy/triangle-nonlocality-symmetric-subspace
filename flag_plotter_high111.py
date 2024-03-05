import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json

# Original distribution from LHV-Net
p_LHVNet = np.loadtxt('./data/probs_LHVNet_high111.csv')
p_LHVNet = p_LHVNet.reshape((4,4,4))

print("Loaded LHV-Net distribution with shape:", p_LHVNet.shape)
print()

# Load deterministic response functions (flags) from ./data/flags_high111.json
with open('./data/flags_high111.json') as f:
    det_flags = json.load(f)

print("Alice flags shape:  ", np.array(det_flags["Alice"]).shape)
print("Bob flags shape:    ", np.array(det_flags["Bob"]).shape)
print("Charlie flags shape:", np.array(det_flags["Charlie"]).shape)


# Define helpful functions
def det_flags_to_distribution(deterministic_flags):
    """ Assumes all flags have same shape."""
    flag_a = np.array(deterministic_flags["Alice"]).astype('int')
    flag_b = np.array(deterministic_flags["Bob"]).astype('int').T
    flag_c = np.array(deterministic_flags["Charlie"]).astype('int')
    grid_resolution = flag_a.shape[0]

    p = np.zeros((4,4,4))
    for alpha in range(grid_resolution):
        for beta in range(grid_resolution):
            for gamma in range(grid_resolution):
                p[flag_a[beta,gamma],flag_b[gamma,alpha],flag_c[alpha,beta]]+=1
    p=p/(grid_resolution**3)
    return p

# Define colormap to be used
colors = ['red','green','blue','orange']
n_bins = 4  # Discretizes the interpolation into bins
cmap_name = 'my_list'
colormap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

def plot_det_flags(det_flags,pabc, p_elegant=None, p_111_is_025=None):
    """ Plots the deterministic flags and the distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(12,13))
    plt.subplot(2,2,1)
    plt.imshow(np.array(det_flags["Alice"]).T,cmap=colormap)
    #plt.gca().invert_yaxis()
    plt.title('Response of Alice to her inputs.')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\gamma$')

    plt.subplot(2,2,2)
    plt.imshow(np.array(det_flags["Bob"]).T,cmap=colormap)
    #plt.gca().invert_yaxis()
    plt.title('Response of Bob to his inputs.')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\gamma$')

    plt.subplot(2,2,3)
    plt.imshow(np.array(det_flags["Charlie"]),cmap=colormap)
    #plt.gca().invert_yaxis()
    plt.title('Response of Charlie to his inputs.')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\alpha$')

    plt.subplot(2,2,4)
    try:
        plt.plot(p_elegant.flatten(),'o',color='green',alpha=0.55,markersize=5,label = 'Elegant')
    except:
        pass
    try:
        plt.plot(p_111_is_025.flatten(),'s',color='orange',alpha=0.85,markersize=5,label=r'$s_{{111}} = 0.25$ (analytic)')
    except:
        pass
    plt.plot(p_from_detflags.flatten(),'rx',alpha = 0.85,markersize=5,label = 'LHV-Net det. approx.')
    plt.plot(p_LHVNet.flatten(),'b+',alpha = 0.85,markersize=5,label = 'LHV-Net original')
    plt.xlabel('outcome')
    plt.ylabel('probability of outcome')
    plt.ylim(ymin=-0.002)
    plt.legend(loc=(0.45,0.27))


    s_111 = pabc[0,0,0] + pabc[1,1,1] + pabc[2,2,2] + pabc[3,3,3]

    l=2

    M_111 = np.mean(np.stack([pabc[index_tuple] for index_tuple in samevalues],axis=0),axis=0)
    M_112 = np.mean(np.stack([pabc[index_tuple] for index_tuple in one_differents],axis=0),axis=0)
    M_123 = np.mean(np.stack([pabc[index_tuple] for index_tuple in all_differents],axis=0),axis=0)

    Delta_111_2 = np.sum(np.stack([np.abs(M_111 - pabc[index_tuple])**l for index_tuple in samevalues],axis=0),axis=0)
    Delta_112_2 = np.sum(np.stack([np.abs(M_112 - pabc[index_tuple])**l for index_tuple in one_differents],axis=0),axis=0)
    Delta_123_2 = np.sum(np.stack([np.abs(M_123 - pabc[index_tuple])**l for index_tuple in all_differents],axis=0),axis=0)

    l=1
    Delta_111_1 = np.sum(np.stack([np.abs(M_111 - pabc[index_tuple])**l for index_tuple in samevalues],axis=0),axis=0)
    Delta_112_1 = np.sum(np.stack([np.abs(M_112 - pabc[index_tuple])**l for index_tuple in one_differents],axis=0),axis=0)
    Delta_123_1 = np.sum(np.stack([np.abs(M_123 - pabc[index_tuple])**l for index_tuple in all_differents],axis=0),axis=0)

    Delta_1 = Delta_111_1 + Delta_112_1 + Delta_123_1
    Delta_2 = Delta_111_2 + Delta_112_2 + Delta_123_2

    fig.suptitle(r"Response functions for LHV-Net distribution's det. approx. with $s_{{111}}\approx{}$.".format(s_111)+"\n"+r"$\Delta_1 = {}$".format(Delta_1) + "\n" + r"$\Delta_2 = {}$".format(Delta_2) , fontsize = 14)
    plt.savefig('./plots/detflag_high111.png', bbox_inches="tight",dpi=300)



# If you would like to plot Elegant distribution and the s_111 = 0.25 distribution as well
    
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

p_elegant = np.zeros((4,4,4))
for o in samevalues:
    p_elegant[o] = 25/256
for o in one_differents:
    p_elegant[o] = 1/256
for o in all_differents:
    p_elegant[o] = 5/256

## One of the s_111 = 0.25, symmetric distributions
p_111_is_025 = np.zeros((4,4,4))
for index_tuple in samevalues:
    p_111_is_025[index_tuple] = 0.25/4
for index_tuple in one_differents:
    p_111_is_025[index_tuple] = 1/80
for index_tuple in all_differents:
    p_111_is_025[index_tuple] = 1/80

p_from_detflags = det_flags_to_distribution(det_flags)
plot_det_flags = plot_det_flags(det_flags,p_from_detflags,p_elegant=p_elegant,p_111_is_025=p_111_is_025)


def get_statistics_of_distribution(name, pabc):
    # Means of values
    M_111 = np.mean(np.stack([pabc[index_tuple] for index_tuple in samevalues],axis=0),axis=0)
    M_112 = np.mean(np.stack([pabc[index_tuple] for index_tuple in one_differents],axis=0),axis=0)
    M_123 = np.mean(np.stack([pabc[index_tuple] for index_tuple in all_differents],axis=0),axis=0)

    # Standard deviations of values
    std_111 = np.std(np.stack([pabc[index_tuple] for index_tuple in samevalues],axis=0),axis=0)
    std_112 = np.std(np.stack([pabc[index_tuple] for index_tuple in one_differents],axis=0),axis=0)
    std_123 = np.std(np.stack([pabc[index_tuple] for index_tuple in all_differents],axis=0),axis=0)

    # Max difference between values and means
    max_diff_111 = np.max(np.stack([np.abs(pabc[index_tuple] - M_111) for index_tuple in samevalues],axis=0),axis=0)
    max_diff_112 = np.max(np.stack([np.abs(pabc[index_tuple] - M_112) for index_tuple in one_differents],axis=0),axis=0)
    max_diff_123 = np.max(np.stack([np.abs(pabc[index_tuple] - M_123) for index_tuple in all_differents],axis=0),axis=0)

    min_111 = np.min(np.stack([pabc[index_tuple] for index_tuple in samevalues],axis=0),axis=0)
    min_112 = np.min(np.stack([pabc[index_tuple] for index_tuple in one_differents],axis=0),axis=0)
    min_123 = np.min(np.stack([pabc[index_tuple] for index_tuple in all_differents],axis=0),axis=0)

    max_111 = np.max(np.stack([pabc[index_tuple] for index_tuple in samevalues],axis=0),axis=0)
    max_112 = np.max(np.stack([pabc[index_tuple] for index_tuple in one_differents],axis=0),axis=0)
    max_123 = np.max(np.stack([pabc[index_tuple] for index_tuple in all_differents],axis=0),axis=0)

    l=2

    Delta_111_2 = np.sum(np.stack([np.abs(M_111 - pabc[index_tuple])**l for index_tuple in samevalues],axis=0),axis=0)
    Delta_112_2 = np.sum(np.stack([np.abs(M_112 - pabc[index_tuple])**l for index_tuple in one_differents],axis=0),axis=0)
    Delta_123_2 = np.sum(np.stack([np.abs(M_123 - pabc[index_tuple])**l for index_tuple in all_differents],axis=0),axis=0)

    l=1
    Delta_111_1 = np.sum(np.stack([np.abs(M_111 - pabc[index_tuple])**l for index_tuple in samevalues],axis=0),axis=0)
    Delta_112_1 = np.sum(np.stack([np.abs(M_112 - pabc[index_tuple])**l for index_tuple in one_differents],axis=0),axis=0)
    Delta_123_1 = np.sum(np.stack([np.abs(M_123 - pabc[index_tuple])**l for index_tuple in all_differents],axis=0),axis=0)

    Delta_1 = Delta_111_1 + Delta_112_1 + Delta_123_1
    Delta_2 = Delta_111_2 + Delta_112_2 + Delta_123_2

    print()
    print("--------------------------------------------------")
    print("Statistics for",name,"distribution")
    print()
    print("Means:")
    print("M_111:",M_111)
    print("M_112:",M_112)
    print("M_123:",M_123)
    print()
    print("Standard deviations:")
    print("std_111:",std_111)
    print("std_112:",std_112)
    print("std_123:",std_123)
    print()
    print("Max differences:")
    print("max_diff_111:",max_diff_111)
    print("max_diff_112:",max_diff_112)
    print("max_diff_123:",max_diff_123)
    print()
    print("Min:")
    print("min_111:",min_111)
    print("min_112:",min_112)
    print("min_123:",min_123)
    print()
    print("Max:")
    print("max_111:",max_111)
    print("max_112:",max_112)
    print("max_123:",max_123)
    print()
    print("Delta (l=1):",Delta_1)
    print("Delta_111_1:",Delta_111_1)
    print("Delta_112_1:",Delta_112_1)
    print("Delta_123_1:",Delta_123_1)
    print()
    print("Delta (l=2):",Delta_2)
    print("Delta_111_2:",Delta_111_2)
    print("Delta_112_2:",Delta_112_2)
    print("Delta_123_2:",Delta_123_2)


    print("--------------------------------------------------")
    print()

get_statistics_of_distribution("original distribution of LHV-Net", p_LHVNet)
get_statistics_of_distribution("deterimistic approximation of LHV-Net", p_from_detflags)


# Just the distribution plots:
plt.clf()
plt.figure(figsize=(5, 4))
try:
    plt.plot(p_elegant.flatten(),'o',color='green',alpha=0.55,markersize=5,label = 'Elegant')
except:
    pass
try:
    plt.plot(p_111_is_025.flatten(),'s',color='orange',alpha=0.85,markersize=5,label=r'$s_{{111}} = 0.25$ (analytic)')
except:
    pass
plt.plot(p_from_detflags.flatten(),'rx',alpha = 0.85,markersize=5,label = 'LHV-Net det. approx.')
plt.plot(p_LHVNet.flatten(),'b+',alpha = 0.85,markersize=5,label = 'LHV-Net original')
plt.xlabel('outcome')
plt.ylabel('probability of outcome')
plt.ylim(ymin=-0.002)
plt.legend(loc=(0.45,0.27))

plt.savefig('./plots/distribution_high111.png', bbox_inches="tight",dpi=300)