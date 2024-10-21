import numpy as np
import matplotlib.pyplot as plt
import random
import glob
import os
import math
from general_functions import extract_from_file_name, file_as_list

def plot_txt_file(directory_path, file_prefix, file_suffix, title_parameters, xlabel, ylabel, output_fig_directory, extract_preffix, extract_suffix, epsilon_color_mapping,compteur,wantlogscale):
    '''
    goes over all file with given prefix 'file_prefix'=str and given suffix 'file_suffix'= str, that are located in directory with path 'directory_path' =str, and
    plot the data of each file in a same figure,
    saves the figure with name depending on 'title_parameters' in directory with path 'output_fig_directory' = str
    xlabel and ylabel are str
    title_parameter = dictionnary
    '''

    # Use glob to find files with the specified prefix and suffix in the directory
    txt_files = glob.glob(os.path.join(directory_path, f'{file_prefix}*{file_suffix}'))
    
    if not txt_files:
        print(f"No matching files found with the name including: {file_prefix}")
        print("or ")
        print(f"No matching files found with the name including: {file_suffix}")
        return

    # Initialize a plot
    plt.figure(figsize=(10, 6))
    

    # Iterate over each file and plot its data
    for txt_file in txt_files:
        epsilon = extract_from_file_name(txt_file, extract_preffix, extract_suffix)
        legendoverl = extract_from_file_name(txt_file,'_a0_A','_Plottedq')
        # Extract data from the file (modify this part based on your data format)
        with open(txt_file, 'r') as file:
            data = [float(line.strip())+1e-20 for line in file]
        # Plot the data with assigned color
        color = epsilon_color_mapping.get(epsilon)
        print(epsilon)
        plt.plot(data, marker='o', linestyle='-', color=color, label=legendoverl)

    if ylabel == 'EE':
        random_data_name = f'random_EE__locald2_rank16_eps{epsilon}.txt'
    if ylabel =='Bond dim.':
        random_data_name = f'random_bondd__locald2_rank16_eps{epsilon}.txt'

    random_data = file_as_list(f'random/{random_data_name}')
    plt.plot(random_data, marker='x', linestyle = '-', color='black', label='Random')
    title = ', '.join([
    f'{key}:{format(value, ".3e")}' if isinstance(value, (int, float)) else f'{key}:{value}'
    for key, value in list(title_parameters.items())[1:]
])

    filename = '_'.join([f'{key}{value}' for key, value in title_parameters.items()])
    # Add labels, title, legend, etc. (modify as needed)
    plt.xlabel(xlabel,fontsize='xx-large')
    plt.ylabel(ylabel,fontsize='xx-large')
    plt.title(title,fontsize='large')
    if compteur==0:
        plt.legend(fontsize='xx-large')
    plt.xticks(fontsize='xx-large')
    plt.yticks(fontsize='xx-large')
    if wantlogscale:
        plt.yscale('log')  # Set y-axis to log scale
    plt.grid('True')
    #plt.xlim(9,10)
    #plt.ylim(-2,2)
    # Save the figure in PDF format
    output_fig_path = os.path.join(output_fig_directory, filename+'.pdf')
    plt.savefig(output_fig_path, format='pdf')

    # Display the plot (optional)
    plt.show()

#-------------------------------------------------------------to generate EE and bondd plot where random and SG overlap are ploted, T by T
# epsilons = [0.002]
# overlap_fct = ['xx','yy','zz']
# alpha = 'z'
# k = 100000000000000000000
# a = 0

# for typeplot in ['EE','Bonddimension']:
#     for wantlogscale in [True,False]: 



#         if typeplot == 'Bonddimension':
#             ylabel = 'Bond dim.'
#             if wantlogscale:
#                 par_name = 'logs_random_bondd'
#             else:
#                 par_name = 'random_bondd'
#         if typeplot == 'EE':
#             ylabel = 'EE'
#             if wantlogscale:
#                 par_name = 'logs_random_EE'
#             else:
#                 par_name = 'random_EE'
#         #------------------

#         extract_prefix='eps'
#         extract_sufix = '_alpha'+alpha
#         xlabel = 'bond'

#         #Choose a color map
#         color_map = plt.cm.get_cmap('tab10')
#         #Generate 5 distinct colors
#         epsilon_color_mapping = {'0.0275':color_map(0), '0.0325':color_map(1), '0.0375':color_map(2), '0.022500000000000003':color_map(3), '0.030000000000000002':color_map(4), '0.034999999999999996':color_map(5),'1e-05':color_map(6)
#                                 ,'0.002':color_map(7), '0.03':color_map(8), '0.01':color_map(9)}


#         for epsilon in epsilons:
#             for ove in overlap_fct:
#                 compteur= 0
#                 for T in [2100]:
#                     directory_path = '/Users/gabinleroy/Desktop/spinglass_python/workspace_spinglass/tota_n_data/n287t'+str(T)
#                     file_prefix = f'datafromplotof{typeplot}_locald2_rank16_k100000000000000000000_eps{epsilon}'
#                     #for alphai in [1, 58, 115, 172, 229, 286]:
#                     file_sufix = f'Plottedq{ove}.txt'
#                     title_parameters = {par_name:'_', 'local_dim':2, 'rank':16, 'k':k, 'a':a,'Plotted':'q'+ove, 'T':float("0."+str(T)), 'eps':epsilon}
#                     plot_txt_file(directory_path, file_prefix, file_sufix, title_parameters, xlabel, ylabel, directory_path, extract_prefix, extract_sufix, epsilon_color_mapping, compteur, wantlogscale)
#                     compteur+=1

#go over all temperature, and plot selected file on same plot, with legend indicating T
Ts = [1500,1945,1960,2100]
#EE or Bonddimension
typeplot = 'EE'
epsilons = [0.002,0.03]
alpha = 'z'
alphai = '150'
A1 = 0
A2 = 1
components = ['xx','yy','zz']
L=16

#Number of subplots per row
subplots_per_row = len(components)
number_rows = len(epsilons)
#Create a figure and subplots
fig, axs = plt.subplots(number_rows, subplots_per_row, sharex='all')
fig.supxlabel(f'bond',fontsize='xx-large')
x = np.arange(L-1)

if typeplot == 'Bonddimension':
    marker = 'o'
    fig.supylabel(f'Bond dim.',fontisze='xx-large')
if typeplot == 'EE':
    marker = 'x'
    fig.supylabel(f'EE', fontsize = 'xx-large')

for i,epsilon in enumerate(epsilons):
    for j,component in enumerate(components):
        #index (i,j) of plot
        for T in Ts:
            Tfloat = float("0."+str(T))
            file_name = f'datafromplotof{typeplot}_locald2_rank16_k100000000000000000000_eps{epsilon}_alpha{alpha}_alphai{alphai}_Lx286_Ly286_T{Tfloat}_a0_A{A1}&{A2}_Plottedq{component}'
            file_path = f'tota_n_data/n287t{T}/'+file_name+'.txt'
            data = file_as_list(file_path, '\t')
            axs[i,j].plot(data, marker=marker, label=f'$T=0.{T}$')
            #axs[i,j].legend(fontsize='large')
            axs[i,j].grid('True')
            #Add label to matrix plot
            axs[i, 0].text(-3.0, 3.5, f'$\epsilon={epsilon}$', va='center', ha='center', rotation='vertical', fontsize='xx-large')
            axs[0, j].text(7, 7.5, f'{component}', va='center', ha='center', rotation='horizontal', fontsize='xx-large')
            #axs[i,j].set_yscale('log')
            axs[i,j].set_xlim(5,10)
            axs[i,j].set_ylim(4,7)
# Add common legend below the figure

handles, labels = axs[-1,-1].get_legend_handles_labels()
fig.legend(handles, labels, loc=(0.1,0.05),fontsize='xx-large', ncol=len(labels))
fig.suptitle(r'$d=2$, $L=16$, $k=1 \times 10^{20}$, $A_1={0}, A_2={1}$, $\alpha=z$, $\alpha_i=150$', fontsize='xx-large')


# Set larger tick labels on all axes
for ax in axs.flat:
    ax.tick_params(axis='both', which='both', labelsize='xx-large')
plt.show()


