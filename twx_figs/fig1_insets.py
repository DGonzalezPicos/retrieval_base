
import numpy as np
import matplotlib.pyplot as plt

import numpy as np

import matplotlib.pyplot as plt

def create_insets(inset_regions, residuals_plot=False):

    # Create a figure with a specific size ratio
    if residuals_plot:
        fig, (ax, ax_res) = plt.subplots(2,1, figsize=(10,4), sharex=True, gridspec_kw={'height_ratios':[5,1], 'hspace':0.0})
    else:
        fig, ax = plt.subplots(1,1, figsize=(10,4))

    

    def add_inset(ax, x1, x2, y1, y2, width, height, x_inset, y_inset):
        # do not show connecting lines
        axins = ax.inset_axes(
            bounds=(x_inset, y_inset, width, height),
            # transform=ax.transData,
            xlim=(x1, x2),
            ylim=(y1, y2),
            # xticklabels=[],
            # yticklabels=[]
            
        )
        # ax.indicate_inset_zoom(axins)
        
        return axins
    offset = 0.34 if residuals_plot else 0.0
    width, height = 0.295, 0.7
    x_inset, y_inset = 0.0, -0.7 - offset
    axins = []
    for r, region in enumerate(inset_regions):
        x1, x2 = region[0]
        y1, y2 = region[1]
        axins.append(add_inset(ax, x1, x2, y1, y2, width, height, x_inset, y_inset))
        x_inset += width + 0.06

    if residuals_plot:
        return fig, (ax, ax_res), axins
    return fig, ax, axins
    
if __name__ == '__main__':
    import retrieval_base.auxiliary_functions as af
    path = af.get_path(return_pathlib=True)
    
    inset_regions = [[(0.2, 1.0), (0.1, 0.4)],
                     [(7.0, 7.5), (0.3, 0.6)],
                     [(9.0, 9.5), (0.1, 0.2)]]
    
    fig, (ax, ax_res), axins = create_insets(inset_regions, residuals_plot=True)
    
    x = np.linspace(0, 10, 1000)
    y = 0.5*np.sin(x) + np.random.normal(0, 0.05, 1000)
    ax.plot(x, y, 'k-', linewidth=1)

    # Save the figure
    plt.savefig(path / 'twx_figs/fig1_three_insets.pdf', bbox_inches='tight', dpi=300)
    print('Saved figure to', path / 'twx_figs/fig1_three_insets.pdf')
