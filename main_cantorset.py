from mps import MPS, data1D_to_vector_base3, approximate_data
from cantor_set import discrete_cantor_set
import numpy as np
from MPS_plot import save_plot_MPS
# R = 7
# eps = 1e-15
# k = 1000000000000
# d = 3
# y = discrete_cantor_set(R,d**R)
# typeplot = 2
# asquantum = data1D_to_vector_bits(y, R, True)
# asquantum = asquantum / np.linalg.norm(asquantum)
# mps_data = MPS(d, R, asquantum, k, eps)[typeplot]
# #save plop
# dirpath = '/Users/gabinleroy/Desktop'
# save_plot_MPS(mps_data, dirpath, R, d, k, eps, typeplot)
#c = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])

import matplotlib.pyplot as plt

def plot_mps_iteration(ax, x, y, label, title,ylabel,i):
    if i == 0:
        op = 'o'
    if i == 1:
        op ='x'
    ax.plot(x, y,marker=op ,label=label)
    #ax.set_title(title)
    #ax.set_xlabel("car")
    # Add text annotations to each subplot
    ax.grid('True')
    #Add label to matrix plot

    


def main():
    #cantor parameters
    q = 3
    Rs = [12,13]
    d_space = 1
    
        
    #MPS parameter
    k = int(1e20)
    typeplot = 1
    directory = 'cantor_data'
    epsilons = [0.1,1e-10,1e-20,1e-30]
    

    
    #Number of subplots per row
    subplots_per_row = 2
    number_rows = 2

    #Create a figure and subplots
    fig, axs = plt.subplots(number_rows, subplots_per_row, sharex='all')
    fig.supxlabel(f'Bond',fontsize='xx-large')
    #fig.supylabel(f'$s_i$',fontsize='xx-large')
    cmap = plt.cm.get_cmap('tab10')
    #----------------------------------------
    #for some iterations of cantor set, compute typeplot for different epsilon and plot all on same figure 
    for (i,typeplot) in enumerate(['Bonddimension','EE']):

        if typeplot == 'Bonddimension':
            ylabel = "Bond dim."
        if typeplot == 'EE':
            ylabel = "EE"
        
        for (j,R) in enumerate(Rs):
            L = d_space * R
            y = discrete_cantor_set(R, q ** R)
            y = data1D_to_vector_base3(y, R, True)
            y = y / np.linalg.norm(y)
            
            for epsilon in epsilons:
                MPS_result = MPS(q, L, y, k, epsilon)
                y_plot = MPS_result[i+1]
                plot_mps_iteration(axs[i,j], range(len(y_plot)), y_plot, f'$\epsilon$ = {epsilon}', f'local dim. ={q} , k={k} , R = {R}',ylabel,i)
                
            # Save the figure for each R
            #fig.savefig(f'{directory}/{ylabel}_plot_locald{q}_rank{L}_k{k}_eps{epsilon}.pdf',format='pdf')

    handles, labels = axs[-1,-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.11,0.92), fontsize='xx-large', ncol=len(labels))  
    # Add text annotations to each subplot
    #ax[1,0].text(7, 7.5, f'{component}', va='center', ha='center', rotation='horizontal', fontsize='xx-large')    
    for ax in axs.flat:
        ax.tick_params(axis='both', which='both', labelsize='xx-large')
    plt.grid('True')   
    plt.show()
    # q = 3
    # R = 13
    # L = R
    # y = discrete_cantor_set(R, q ** R)
    # y = data1D_to_vector_base3(y, R, True)
    # y = y / np.linalg.norm(y)
    # MPS(q, L, y, int(1e20), 0.1,{},True)
   
if __name__ == "__main__":
    main()
