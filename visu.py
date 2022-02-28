import numpy as np 
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def heatmap_vizualisation(data, candidate_score, feature_name, best_value, delta_best_value):
    values = list(data)

    vmin_value = np.min(values)
    vmax_value = np.max(values)

    colors = []

    # approx best_values    
    
    best_value_approx = best_value
    fac = 1
     
    if best_value < 1 : 
        while best_value_approx < 10 : 
            fac *= 10
            best_value_approx = best_value_approx * 10
        
        delta_best_value_approx = np.round(delta_best_value * fac)

    else :
        while best_value_approx > 10 : 
            fac *= 10
            best_value_approx = best_value_approx / 10

        delta_best_value_approx = np.round(delta_best_value / fac)

    best_value_approx = np.round(best_value_approx)

    
    # define cmap
    for i in range(int(max(best_value_approx - delta_best_value_approx - 2, 0))):
        colors.append('r')

    for i in range(int(max(best_value_approx - delta_best_value_approx - 2, 0)), int(min(best_value_approx - delta_best_value_approx + 1, best_value_approx - 1))):
        colors.append('y')

    if best_value_approx - delta_best_value_approx + 1 < best_value_approx - 1 :
        for i in range(int(best_value_approx - delta_best_value_approx + 1), int(best_value_approx + delta_best_value_approx - 1)) :
            colors.append('g')

    else : 
        for i in range(int(best_value_approx - 1), int(best_value_approx + 1)) :
            colors.append('g')

    print(colors)

    for i in range(int(max(best_value_approx + delta_best_value_approx - 1, best_value_approx + 1)), int(min(best_value_approx + delta_best_value_approx + 2, 10))):
        colors.append('y')

    for i in range(int(min(best_value_approx + delta_best_value_approx - 1, 10)), 10):
        colors.append('r')

        
    print(colors)

    n_bins = 1000*vmax_value


    cmap_value = LinearSegmentedColormap.from_list('RYGYR', colors, N=n_bins)

    center_value = best_value
    cbar_value = False

    title_value = feature_name

    fig, ax = plt.subplots(1,1, figsize=(20,2))
    sb.heatmap([values], vmin = vmin_value, vmax = vmax_value, cmap = cmap_value, cbar = cbar_value, center = center_value, xticklabels = False, yticklabels = False, ax = ax)

    ax.set_title(title_value, color='w', size=16)

    x = [candidate_score, candidate_score]
    y = [0, 1]

    ax.plot(x,y, linewidth = 5, color = 'black');

    fig.show()