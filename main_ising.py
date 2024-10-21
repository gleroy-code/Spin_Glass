from ising import load_ising_data, totalEE, find_plateau_value
import numpy as np
import matplotlib.pyplot as plt
import math
from mps import image_to_vector_bits, MPS
from scipy.signal import find_peaks
from general_functions import add_to_txt
def main():
    print('o')
# total EE - data does not exist -----------------------------------------------------------------------------------------------------
    # Ts = [1.9, 1.91, 1.92, 1.93, 1.94, 1.95, 1.96, 1.97, 1.98, 1.99, 
    # 2.0, 2.01, 2.02, 2.03, 2.04, 2.05, 2.06, 2.07, 2.08, 2.09, 
    # 2.1, 2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 2.18, 2.19, 
    # 2.2, 2.205, 2.21, 2.22, 2.225, 2.23, 2.235, 2.24, 2.245, 
    # 2.25, 2.255, 2.257, 2.257, 2.26, 2.265, 2.267, 2.27, 2.273, 
    # 2.275, 2.276, 2.278, 2.28, 2.281, 2.282, 2.283, 2.285, 2.287, 
    # 2.288, 2.29, 2.295, 2.297, 2.3, 2.305, 2.310 ,2.32, 2.33, 2.34, 
    # 2.35, 2.36, 2.37, 2.38, 2.39, 2.4, 2.41, 2.42, 2.43, 2.44, 2.45, 
    # 2.46, 2.47, 2.48, 2.49, 2.5, 2.51, 2.52, 2.53, 2.54, 2.55, 2.56, 
    # 2.57, 2.58, 2.59, 2.6, 2.61, 2.62, 2.63, 2.64, 2.65, 2.66, 2.67, 2.68, 2.69, 2.7]
    # ising_arrays = load_ising_data(Ts)





    # # Truncation parameters
    # k = 20000000
    # epsilons = [0.000000001, 0.01, 0.1, 0.3]

    # # Critical temperature
    # Tc = 2.2691

    # # Define colors for each epsilon value
    # colors = ['b', 'r', 'c', 'm']

    # # List of R values
    # R_values = [4]
    # #R_values = [4,4,4,4]

    # # Number of subplots per row
    # plots_per_row = 2

    # # Calculate the number of rows needed
    # num_rows = len(R_values) // plots_per_row
    # if len(R_values) % plots_per_row != 0:
    #     num_rows += 1

    # # Create subplots
    # fig, axs = plt.subplots(num_rows, plots_per_row, sharex='all')

    # # Flatten the axs array to handle both cases (1 row or multiple rows)
    # axs = axs.flatten()

    # for i, R in enumerate(R_values):
    #     # Plotting the data for each epsilon value with a different color
    #     for epsilon, color in zip(epsilons, colors):
    #         tot = totalEE(R, k, epsilon, ising_arrays)
    #         axs[i].scatter(Ts, tot, marker='.', color=color, label=f'$\epsilon={epsilon}$')

    #         # Save the data to a text file
    #         data = np.column_stack((Ts, tot))
    #         filename = f"R_{R}_epsilon_{epsilon}_data.txt"
    #         np.savetxt(filename, data, header='ts tot', comments='')

    #     # Adding labels, legend, and grid
    #     axs[i].set_xlabel('$T/J$',fontsize='xx-large')
    #     axs[i].set_ylabel('$S(T)$',fontsize='xx-large')
    #     #axs[i].set_title(f'Total EE for Ising square lattice, $2^R\\times2^R$, $R={R}$')
    #     axs[i].set_title(f'$R={R}$',fontsize='xx-large')
    #     axs[i].grid(True)

    #     # Add a vertical dashed line at the critical temperature with a legend
    #     axs[i].axvline(x=Tc, linestyle='--', color='k', label=f'$T_c={Tc}$')
    #     #axs[i].legend()

    # handles, labels = axs[-1].get_legend_handles_labels()
    # fig.legend(handles, labels, loc=(0.11,0.92),fontsize='xx-large', ncol=len(labels))
    # #fig.suptitle(r'$d=2$, $L=16$, $k=1 \times 10^{20}$, $A_1={0}, A_2={1}$, $\alpha=z$, $\alpha_i=150$', fontsize='xx-large')
        
    # # Adjust layout and save the plot
    # plt.tight_layout()

    # # Create a filename indicating totalEE, k, and different values of epsilon
    # filename = f"totalEE_k={k}_epsilons={'_'.join(map(str, epsilons))}.pdf"

    # # Set larger tick labels on all axes
    # for ax in axs.flat:
    #     ax.tick_params(axis='both', which='both', labelsize='xx-large')
        
    # # Save the figure
    # plt.savefig(filename)


    # # Show the plot
    # plt.show()
    

# total EE - data exist -----------------------------------------------------------------------------------------------------
    # Ts = [1.9, 1.91, 1.92, 1.93, 1.94, 1.95, 1.96, 1.97, 1.98, 1.99, 
    # 2.0, 2.01, 2.02, 2.03, 2.04, 2.05, 2.06, 2.07, 2.08, 2.09, 
    # 2.1, 2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 2.18, 2.19, 
    # 2.2, 2.205, 2.21, 2.22, 2.225, 2.23, 2.235, 2.24, 2.245, 
    # 2.25, 2.255, 2.257,2.257, 2.26, 2.265, 2.267, 2.27, 2.273, 
    # 2.275, 2.276, 2.278, 2.28, 2.281, 2.282, 2.283, 2.285, 2.287, 
    # 2.288, 2.29, 2.295, 2.297, 2.3, 2.305, 2.310 ,2.32, 2.33, 2.34, 
    # 2.35, 2.36, 2.37, 2.38, 2.39, 2.4, 2.41, 2.42, 2.43, 2.44, 2.45, 
    # 2.46, 2.47, 2.48, 2.49, 2.5, 2.51, 2.52, 2.53, 2.54, 2.55, 2.56, 
    # 2.57, 2.58, 2.59, 2.6, 2.61, 2.62, 2.63, 2.64, 2.65, 2.66, 2.67, 2.68, 2.69, 2.7]

    # # Truncation parameters
    # k = 20000000
    # #epsilons = [0.000000001, 0.01, 0.1, 0.3]
    # epsilons = [0.000000001]

    # # Critical temperature
    # Tc = 2.2691

    # # Define colors for each epsilon value
    # colors = ['b', 'r', 'c', 'm']

    # # List of R values
    # R_values = [4, 6, 7, 9]
    # #R_values = [4,4,4,4]

    # # Number of subplots per row
    # plots_per_row = 2

    # # Calculate the number of rows needed
    # num_rows = len(R_values) // plots_per_row
    # if len(R_values) % plots_per_row != 0:
    #     num_rows += 1

    # # Create subplots
    # fig, axs = plt.subplots(num_rows, plots_per_row, sharex='all')

    # infl_point = []

    # # Flatten the axs array to handle both cases (1 row or multiple rows)
    # axs = axs.flatten()
    # fig.supxlabel(f'$T/J$',fontsize='xx-large')  
    # fig.supylabel('$S(T)$',fontsize='xx-large')
    # for i, R in enumerate(R_values):
    #     # Plotting the data for each epsilon value with a different color
    #     for epsilon, color in zip(epsilons, colors):
    #         file_dir = '/Users/gabinleroy/Desktop/spinglass_python/ising_data/total_EE/Rfiles1minus/'
    #         file_name = f'R_{R}_epsilon_{epsilon}_data.txt'
    #         full_name = file_dir+file_name
    #         tot = np.loadtxt(full_name, usecols=(1,), skiprows=1)
    #         axs[i].scatter(Ts, tot, marker='.', color=color, label=f'$\epsilon={epsilon}$')

    #         #extract inflection point
        
    #         # Compute the first derivative of y_values with respect to Ts
    #         dy_dT = np.gradient(tot, Ts)
    #         # Compute the second derivative of y_values with respect to Ts
    #         d2y_dT2 = np.gradient(dy_dT, Ts)
    #         indices = np.argmax(abs(d2y_dT2) > 19000)
    #         infl_point.append(Ts[indices])
    #         plt.figure()
    #         plt.plot(Ts,d2y_dT2,label=f'R{R}, eps{epsilon}')
    #         plt.legend()

    #     # Adding labels, legend, and grid
    #     #axs[i].set_xlabel('$T/J$',fontsize='xx-large')
    #     #axs[i].set_ylabel('$S(T)$',fontsize='xx-large')
    #     #axs[i].set_title(f'Total EE for Ising square lattice, $2^R\\times2^R$, $R={R}$')
    #     axs[i].set_title(f'$R={R}$',fontsize='xx-large')
    #     axs[i].grid(True)

    #     # Add a vertical dashed line at the critical temperature with a legend
    #     axs[i].axvline(x=Tc, linestyle='--', color='k', label=f'$T_c={Tc}$')
    #     #axs[i].legend()

    # handles, labels = axs[-1].get_legend_handles_labels()
    # fig.legend(handles, labels, loc=(0.11,0.92),fontsize='xx-large', ncol=len(labels))
    # #fig.suptitle(r'$d=2$, $L=16$, $k=1 \times 10^{20}$, $A_1={0}, A_2={1}$, $\alpha=z$, $\alpha_i=150$', fontsize='xx-large')
        
    # # Adjust layout and save the plot
    # plt.tight_layout()

    # # Create a filename indicating totalEE, k, and different values of epsilon
    # filename = f"totalEE_k={k}_epsilons={'_'.join(map(str, epsilons))}.pdf"

    # # Set larger tick labels on all axes
    # for ax in axs.flat:
    #     ax.tick_params(axis='both', which='both', labelsize='xx-large')
        
    # # Save the figure
    # plt.savefig(filename)
    # print(infl_point)

    # # Show the plot
    # plt.show()





    

#mean EE sev T sev eps ----------------------------------------------------------------------
    # T_values = [
    #     1.9, 2.35, 2.7,
    #     2.265
    #     ]
    # ising_arrays = load_ising_data(T_values)

    # # lattice size
    # R = 9
    # # rank of TTF
    # L = 2 * R
    # d = 2
    # # Truncation parameters
    # k = 20000000
    # epsilons = [0.000000001, 0.01, 0.1]

    # # Critical temperature
    # Tc = 2.2691
    # x_values = [i for i in range(0, L - 1)]  # L-1 bond dimensions 

    # # Define colors and markers for each epsilon value
    # colors = ['b', 'r', 'c']
    # markers = ['x', 's', 'D']
    # marker_sizes = [12, 4, 2]  # Set the size for 'x' marker to 30, others to 20

    # # Number of subplots per row
    # plots_per_row = 2 

    # # Calculate the number of rows needed
    # num_rows = len(T_values) // plots_per_row
    # if len(T_values) % plots_per_row != 0:
    #     num_rows += 1

    # # Calculate the size of each subplot to fit A4 page
    # fig_width, fig_height = 8.27, 11.69  # A4 dimensions in inches
    # subplot_size = min(fig_width / plots_per_row, fig_height / num_rows)

    # # Increase the size of each subplot
    # subplot_size *= 2

    # # Create subplots
    # fig, axs = plt.subplots(num_rows, plots_per_row, sharex='all')
    # fig.supxlabel(f'$i$',fontsize='xx-large')
    # # Flatten the axs array to handle both cases (1 row or multiple rows)
    # axs = axs.flatten()

    # for i, T in enumerate(T_values):
    #     print('for T='+str(T))
    #     # Plotting the data for each epsilon value with a different color and marker
    #     for j, (epsilon, color) in enumerate(zip(epsilons, colors)):
    #         EE_tempor = []
    #         for snap in ising_arrays[i][1::]:
    #             y = image_to_vector_bits(snap, R, True)
    #             y = y / np.linalg.norm(y)
    #             EE = MPS(d, L, y, k, epsilon)[2]
    #             EE_tempor.append(EE)
            
    #         array_of_lists = np.array(EE_tempor)
    #         mean_list = np.mean(array_of_lists, axis=0)
    #         mean_EE = mean_list.tolist()
    #         print('mean EE = ')
    #         print(mean_EE)
        
    #         marker_size = marker_sizes[j] if markers[j] == 'x' else marker_sizes[-1]
            
    #         # Use plot instead of scatter to connect points with lines
    #         axs[i].plot(x_values, mean_EE, marker=markers[j], markersize=marker_size, linestyle='-', color=color, label=f'$\epsilon={epsilon}$')

    #         # Save the data to a text file
    #         #data = np.column_stack((x_values, EE))
    #         #filename = f"T={T}_epsilon={epsilon}_k={k}.txt"
    #         #np.savetxt(filename, data, header='car meanEE', comments='')

    #     # Adding labels, set y-axis to log scale, and grid
    #     axs[i].set_xlabel('bond', fontsize='xx-large')
    #     axs[i].set_ylabel('Mean EE', fontsize='xx-large')
    #     axs[i].set_title(f'$T={T}$', fontsize='xx-large')
    #     #axs[i].set_yscale('log')
    #     axs[i].grid(True)
    # for ax in axs.flat:
    #     ax.tick_params(axis='both', which='both', labelsize='xx-large')
    # # Add a legend below all the subplots
    # handles, labels = axs[-1].get_legend_handles_labels()
    # fig.legend(handles, labels, loc=(0.11,0.92),fontsize='xx-large', ncol=len(labels))

    # # Adjust layout and save the plot as SVG (Scalable Vector Graphics)
    # plt.tight_layout()

    # # Create a filename indicating totalEE, k, and different values of epsilon
    # filename = f"meanEE(car)_k={k}_epsilons={'_'.join(map(str, epsilons))}.pdf"

    # # Save the figure in vectorial format (SVG)
    # plt.savefig(filename, format='pdf', bbox_inches='tight')

    # # Show the plot
    # plt.show()

# #position of plateau of mean EE for several epsilon as a function of T 
#     T_values = [2.270]
#     ising_arrays = load_ising_data(T_values)

#     # lattice size
#     R = 9
#     # rank of TTF
#     L = 2 * R
#     d = 2
#     # Truncation parameters
#     k = int(1e20)
#     epsilons = [1e-100]

#     # Critical temperature
#     Tc = 2.2691
#     x_values = [i for i in range(0, L - 1)]  # L-1 bond dimensions 

#     # Define colors and markers for each epsilon value
#     #colors = ['b', 'r', 'c']
#     #markers = ['x', 's', 'D']
#     #marker_sizes = [12, 4, 2]  # Set the size for 'x' marker to 30, others to 20
#     fig, axs = plt.subplots(1, 2, sharex='all')
#     fig.supxlabel(f'bond',fontsize='xx-large')
    
#     # Create subplots

#     # Lists to store plateau values and corresponding temperatures

#     for j, epsilon in enumerate(epsilons):
        
#         # Plotting the data for each epsilon value with a different color and marker
#         for i, T in enumerate(T_values):
#             print('for T='+str(T))
#             EE_tempor = []
#             #bondd_tempor = []
#             for snap in ising_arrays[i][1::]:
                
#                 y = image_to_vector_bits(snap, R, True)
#                 y = y / np.linalg.norm(y)
#                 mps = MPS(d, L, y, k, epsilon)
                
#                 EE = mps[2]
#                 EE_tempor.append(EE)
#                 #bondd = mps[1]
#                 #bondd_tempor.append(bondd)
                
            
#             array_of_lists = np.array(EE_tempor)
#             mean_list = np.mean(array_of_lists, axis=0)
#             max_index = np.argmax(mean_list) 
#             mean_EE = mean_list.tolist()
#             #array_of_lists2 = np.array(bondd_tempor)
#             #mean_list2 = np.mean(array_of_lists2, axis=0)
#             #mean_bondd = mean_list2.tolist()
#             print('mean EE = ')
#             print(mean_EE)

            

#             ## Use plot instead of scatter to connect points with lines
#             #axs[0].plot(x_values, mean_EE, marker='x', linestyle='-', label=f'$T={T}$')
#             #axs[1].plot(x_values, mean_bondd, marker='x', linestyle='-', label=f'$T={T}$')
#             #plt.legend()
#             ## Save the data to a text file
#             data = np.column_stack((x_values, EE))
#             filename = f"meanEE/meanEE_T{T}_epsilon{epsilon}_k{k}.txt"
#             np.savetxt(filename, data, header='bond meanEE', comments='')
#             add_to_txt([max_index],'maxindex_EE.txt', row_by_row=False)
            
       

#         ## Adding labels, set y-axis to log scale, and grid
        
#         #axs[0].set_ylabel('Mean EE', fontsize='xx-large')
#         #axs[1].set_ylabel('Mean Bond dim.', fontsize='xx-large')
#         #axs[i].set_yscale('log')
#         #for ax in axs.flat:
#         #    ax.tick_params(axis='both', which='both', labelsize='xx-large')
#         #axs[0].grid(True)
#         #axs[1].grid(True)
    
   

#     ## Adjust layout and save the plot as SVG (Scalable Vector Graphics)
#     #plt.tight_layout()

#     ## Create a filename indicating totalEE, k, and different values of epsilon
#     #filename = f"meanEE(car)_k={k}_epsilons={'_'.join(map(str, epsilons))}.pdf"

#     ## Save the figure in vectorial format (SVG)
#     #plt.savefig(filename, format='pdf', bbox_inches='tight')

#     ## Show the plot
#     #plt.show()
    
    
#plot position of max of EE as a function of T-------------------------------------------------------------
# T_values = [1.9, 1.91, 1.92, 1.93, 1.94, 1.95, 1.96, 1.97, 1.98, 1.99, 
# 2.0, 2.01, 2.02, 2.03, 2.04, 2.05, 2.06, 2.07, 2.08, 2.09, 
# 2.1, 2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 2.18, 2.19, 
# 2.2, 2.205, 2.21, 2.22, 2.225, 2.23, 2.235, 2.24, 2.245, 
# 2.25, 2.255, 2.257, 2.26, 2.265, 2.267, 2.27, 2.273, 
# 2.275, 2.276, 2.278, 2.28, 2.281, 2.282, 2.283, 2.285, 2.287, 
# 2.288, 2.29, 2.295, 2.297, 2.3, 2.305, 2.310 ,2.32, 2.33, 2.34, 
# 2.35, 2.36, 2.37, 2.38, 2.39, 2.4, 2.41, 2.42, 2.43, 2.44, 2.45, 
# 2.46, 2.47, 2.48, 2.49, 2.5, 2.51, 2.52, 2.53, 2.54, 2.55, 2.56, 
# 2.57, 2.58, 2.59, 2.6, 2.61, 2.62, 2.63, 2.64, 2.65, 2.66, 2.67, 2.68, 2.69, 2.7]

# maxpos = [8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 7,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8,
# 8]
    
# plt.figure()
# plt.scatter(T_values,maxpos,marker='o',color='black')
# plt.xlabel('T/J', fontsize='xx-large')
# plt.ylabel('Position of max EE', fontsize='xx-large')
# plt.xticks(fontsize='xx-large')
# plt.yticks(fontsize='xx-large')
# plt.grid('True')

# plt.axvline(x=T_values[40], color='mediumspringgreen', linestyle='--',label=f'T={T_values[40]}')
# plt.axvline(x=T_values[77], color='orange', linestyle='--', label=f'T={T_values[77]}')
# plt.legend(fontsize='xx-large')

# plt.show()



#position of plateau of mean EE for several epsilon as a function of T 
T_values = [2.270]
ising_arrays = load_ising_data(T_values)

# lattice size
R = 9
# rank of TTF
L = 2 * R
d = 2
# Truncation parameters
k = int(1e20)
epsilons = [1e-100]

# Critical temperature
Tc = 2.2691
x_values = [i for i in range(0, L - 1)]  # L-1 bond dimensions 

# Define colors and markers for each epsilon value
#colors = ['b', 'r', 'c']
#markers = ['x', 's', 'D']
#marker_sizes = [12, 4, 2]  # Set the size for 'x' marker to 30, others to 20
plt.figure()

# Create subplots

# Lists to store plateau values and corresponding temperatures

for j, epsilon in enumerate(epsilons):
    
    # Plotting the data for each epsilon value with a different color and marker
    for i, T in enumerate(T_values):
        print('for T='+str(T))
        EE_tempor = []
        #bondd_tempor = []
        for (p,snap) in enumerate(ising_arrays[i][1::][:10]):
            
            y = image_to_vector_bits(snap, R, True)
            y = y / np.linalg.norm(y)
            mps = MPS(d, L, y, k, epsilon)
            
            EE = mps[2]
            EE_tempor.append(EE)
            ## Use plot instead of scatter to connect points with lines
            plt.plot(x_values, EE, marker='.', linestyle='-',label=f'i={p}')
            plt.xlabel('bond', fontsize='xx-large')
            plt.ylabel('EE', fontsize='xx-large')
            plt.xticks(fontsize = 'xx-large')
            plt.yticks(fontsize = 'xx-large')
            plt.grid('True')
            plt.legend()
            #bondd = mps[1]
            #bondd_tempor.append(bondd)
            
        
        array_of_lists = np.array(EE_tempor)
        mean_list = np.mean(array_of_lists, axis=0)
        max_index = np.argmax(mean_list) 
        mean_EE = mean_list.tolist()
        #array_of_lists2 = np.array(bondd_tempor)
        #mean_list2 = np.mean(array_of_lists2, axis=0)
        #mean_bondd = mean_list2.tolist()
        print('mean EE = ')
        print(mean_EE)


    

    ## Adding labels, set y-axis to log scale, and grid
    
    #axs[0].set_ylabel('Mean EE', fontsize='xx-large')
    #axs[1].set_ylabel('Mean Bond dim.', fontsize='xx-large')
    #axs[i].set_yscale('log')
    #for ax in axs.flat:
    #    ax.tick_params(axis='both', which='both', labelsize='xx-large')
    #axs[0].grid(True)
    #axs[1].grid(True)



## Adjust layout and save the plot as SVG (Scalable Vector Graphics)
#plt.tight_layout()

## Create a filename indicating totalEE, k, and different values of epsilon
#filename = f"meanEE(car)_k={k}_epsilons={'_'.join(map(str, epsilons))}.pdf"

## Save the figure in vectorial format (SVG)
#plt.savefig(filename, format='pdf', bbox_inches='tight')

## Show the plot
plt.show()


if __name__ == "__main__":
    main()