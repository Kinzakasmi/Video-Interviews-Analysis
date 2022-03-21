import numpy as np 
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def heatmap_vizualisation(candidate_score, feature_name, best_value=500, delta_best_value=200, vmin=0, vmax=1000):
    vmax_approx = vmax
    vmin_approx = vmin
    fac = 1

    while vmax_approx - vmin_approx <= 1000 :
        vmin_approx = vmin_approx*10
        vmax_approx = vmax_approx*10
        fac = fac*10

    best_value_approx = best_value*fac
    delta_best_value_approx = delta_best_value*fac
    candidate_score_approx = candidate_score*fac

    values = [i for i in range(vmin_approx, vmax_approx)]

    # define cmap
    n_val  = len(values)
    colors = np.zeros((n_val, 4))

    i_green  = min(best_value_approx + delta_best_value_approx, np.max(values)) - vmin_approx
    i_red    = max(best_value_approx - delta_best_value_approx, vmin_approx) - vmin_approx
    i_yellow = best_value_approx - vmin_approx

    colors[0,:] = [1,0,0,1]

    for i in range(1, n_val) :
        if i < i_red : 
            colors[i,:] = [1, 1 + (i - i_red) / i_red, 0, 1]

        elif i < i_yellow :
            colors[i,:] = [(i_yellow - i) / (i_yellow - i_red), 1, 0, 1]

        elif i < i_green :
            colors[i,:] = [1 - (i_green - i) / (i_green - i_yellow), 1, 0, 1]

        else :
            colors[i,:] = [1, (n_val - i) / (n_val - i_green), 0, 1]


    cmap_value = ListedColormap(colors)

    # plot figures 
    cbar_value = False
    title_value = feature_name

    fig, ax = plt.subplots(1,1, figsize=(20,2), facecolor='white')

    sb.heatmap([values], cmap = cmap_value, cbar = cbar_value, xticklabels = False, yticklabels = False, ax = ax)

    ax.set_title(title_value, color='black', size=20)
    
    x = [candidate_score_approx - vmin_approx, candidate_score_approx - vmin_approx]
    y = [0, 1]

    ax.plot(x,y, linewidth = 5, color = 'black');

    plt.text(0 - 20, 1.2, str(vmin), color = 'black', size=16)
    plt.text(candidate_score_approx - vmin_approx - 20, 1.2, str(candidate_score), color = 'black', size=16)
    plt.text(n_val - 20, 1.2, str(vmax), color = 'black', size=16)

    fig.show()

def boxplot_viz(feats):
    import ipywidgets as widgets
    from ipywidgets import Layout

    email = widgets.Select(
        options = feats.index.tolist(),
        description='Interview',
        disabled=False,
        layout = Layout(width='50%', height='80px', display='flex')
    )

    question = widgets.Select(
        options = feats.question.tolist(),
        description='Question',
        disabled=False,
        layout = Layout(width='50%', height='80px', display='flex')
    )


    variable = widgets.Select(
        options = feats.columns.tolist(),
        description='Variable',
        disabled=False,
        layout = Layout(width='50%', height='80px', display='flex')
    )

    import matplotlib.pyplot as plt
    import seaborn as sns

    def print_boxplot(email,variable):
        var = round(feats.loc[email,variable],3)

        sns.set_theme(style="whitegrid")
        ax = sns.boxplot(y=variable, data=feats)
        ax.axhline(var,c='r')

        plt.text(-0.3, var*1.005, var, horizontalalignment='left', size='small', color='red', weight='normal')

    widgets.interactive(print_boxplot,email=email,variable=variable)
