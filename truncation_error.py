import numpy as np
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
import time
from mps import approximate_data, MPS, image_to_vector_bits
import math
from general_functions import save_list_as_txt, add_to_txt,file_as_list
from MPS_several_eps import MPS_sev_eps
# ---------------------------------------------------------------------------------------------------------------
#compute the relative error induced by truncation, fixed eps, varying k
# d = 2
# L_eps_list = [[10,0.05],[10,1e-100],[15,0.01],[15,1e-100]]
# fig, axs = plt.subplots(2, 2)

# # Create a 2x2 subplot layout
# fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# for i, (L, epsilon) in enumerate(L_eps_list):
    
#     k = 10**1000 
#     errors = np.array([])
#     c = np.random.uniform(0, 1, size=(1, d**L))
#     norm_c = np.linalg.norm(c)
#     c = c/norm_c
#     D = max(MPS(d,L,c,k,epsilon)[1])+10
#     errors = np.array([])

#     for j in range(d, D):
#         errors = np.append(errors, np.linalg.norm(c - approximate_data(d, L, c, j, epsilon)))

#     # Determine the subplot location
#     row = i // 2
#     col = i % 2

#     # Plot the scatter plot in the corresponding subplot
#     axs[row, col].scatter(range(d, D), errors, marker='o', color='blue', label='Data Points')

#     # Add labels and a title to each subplot
#     axs[row, col].set_xlabel(f'truncation parameter $k$')
#     axs[row, col].set_ylabel(f'$e_{{\psi}}(k, \epsilon)$')
#     axs[row, col].set_title('L = {}, d = {}, $\epsilon$ = {}'.format(L, d, epsilon))

#     # Add a grid
#     axs[row, col].grid(True)

# # Adjust layout for better spacing
# plt.tight_layout()

# # Save the figure if needed
# plt.savefig('all_subplots.pdf')

# # Show the plots
# plt.show()

# ---------------------------------------------------------------------------------------------------------------
#compute the relative error induced by truncation, fixed k, several L, if error data does not exist
# d = 2
# k = 10*10000
# L_k_list = [[8,k],[10,k],[12,k],[14,k]]
# eps = np.linspace(1e-100, 1, 40)

# # Create a 2x2 subplot layout
# fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex='all',sharey='all')
# fig.supxlabel(f'$\epsilon$',fontsize='xx-large')
# fig.supylabel(f'$e(k,\epsilon)$',fontsize='xx-large')

# for i, (L, k) in enumerate(L_k_list):
    
#     R = int(L/2)
#     image_name = '5.3.01.tiff'
#     image_data = image_to_vector_bits(image_name,R,False)
#     image_data = image_data/np.linalg.norm(image_data)
#     error_image = np.array([])

#     L = 2*R
#     errors = np.array([])
#     c = np.random.uniform(0, 1, size=(1, d**L))
#     norm_c = np.linalg.norm(c)
#     c = c/norm_c
   
#     errors = np.array([])

#     for epsilon in eps:
#         y1 = MPS(d,L,c,k,epsilon)
#         y2 = MPS(d,L,image_data,k,epsilon)
#         errors = np.append(errors, np.linalg.norm(c - approximate_data(d, L, y1, k, epsilon)))
#         error_image = np.append(error_image, np.linalg.norm(image_data - approximate_data(d, L, y2, k, epsilon)))

#     # Determine the subplot location
#     row = i // 2
#     col = i % 2

#     # Plot the scatter plot in the corresponding subplot
#     axs[row, col].plot(eps, errors, marker='D', color='c', label=f'$\sim U(0,1)$, $L={L}$')
#     axs[row, col].plot(eps, error_image, marker='s', color='purple', label=f'Image, $L={L}$')
#     axs[row,col].legend(fontsize='xx-large')
#     # Set the minimum y-value to 0
#     axs[row, col].set_ylim(bottom=0)
#     # Add labels and a title to each subplot
#     #axs[row, col].set_xlabel(f'truncation parameter $\epsilon$')
#     #axs[row, col].set_ylabel(f'$e_{{\psi}}(k, \epsilon)$')
#     #axs[row, col].set_title(f'$L={L}$, $d={d}$, $k=+\infty$')
#     # Add a grid
#     axs[row, col].grid(True)

# #adjust font size of axis tick labels
# for ax in axs.flatten():
#     ax.tick_params(axis='both', labelsize='xx-large')

# # Adjust layout for better spacing
# plt.tight_layout()

# # Save the figure if needed
# plt.savefig('all_subplotspa.pdf')

# #limit
# plt.ylim(-0.03, 1.03)
# fig.suptitle(f'Truncation error, $d={d}$, $k={k:.3e}$ ',fontsize='xx-large')
# # Show the plots
# plt.show()

#compute the relative error induced by truncation, fixed k, several L, if error data exists--------------------------
# d = 2
# k = 100000000000000000000
# L = 16


# eps =  np.linspace(0.0025,0.5,40)

# # Create a 2x2 subplot layout
# fig, axs = plt.subplots(1, 1, figsize=(10, 8), sharex='all',sharey='all')
# fig.supxlabel(f'$\epsilon$',fontsize='xx-large')
# fig.supylabel(f'$e(k,\epsilon)$',fontsize='xx-large')



# error_image = np.array([])  
# errors = np.array([])

# for epsilon in eps:
#     print(epsilon)
#     random_data_name = f'errors/random_error_locald2_rank{L}_k{k}_eps{epsilon}.txt'
#     data_name = f'errors/error_locald2_rank{L}_k{k}_eps{epsilon}_alphaz_alphai150_Lx286_Ly286_T0.0.15_a0_A01_PlottedSxx.txt'
#     errors = np.append(errors,file_as_list(random_data_name,' ')[1])
#     print("bla")
#     print(file_as_list(random_data_name,' ')[1])
#     error_image = np.append(error_image,file_as_list(data_name,' ')[1])
#     print(errors)
    
# # Plot the scatter plot in the corresponding subplot
# plt.plot(eps, errors, marker='D', color='c', label=f'$\sim N(0,1)$, $L={L}$')
# plt.plot(eps, error_image, marker='s', color='purple', label=f'SG overlap, $L={L}$')
# plt.legend(fontsize='xx-large')
# # Add labels and a title to each subplot
# #axs[row, col].set_xlabel(f'truncation parameter $\epsilon$')
# #axs[row, col].set_ylabel(f'$e_{{\psi}}(k, \epsilon)$')
# #axs[row, col].set_title(f'$L={L}$, $d={d}$, $k=+\infty$')
# # Add a grid
# plt.grid(True)


# # Adjust layout for better spacing
# plt.tight_layout()
# plt.xticks(fontsize='xx-large')
# plt.yticks(fontsize='xx-large')
# # Save the figure if needed
# plt.savefig('all_subplotspa.pdf')

# fig.suptitle(f'Truncation error, $d={d}$, $k={k:.3e}$ ',fontsize='xx-large')
# # Show the plots
# plt.show()


# shared axis EE or bondd for different set of epsilons
# ---------------------------------------------------------------------------------------------------------------
# d = 2
# L = 22
# c = np.random.uniform(0, 1, size=(1, d**L))
# norm_c = np.linalg.norm(c)
# c = c/norm_c
# x = np.arange(L-1)
# k = 10**1000

# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# fig.suptitle(f'Random tensor with entries $\sim U(0,1)$, $d=${d}, $L=${L}', fontsize='x-large')

# # ax1------
# epsilons = np.linspace(0.001, 0.02, 2)
# for eps in epsilons:
#     y = MPS(d, L, c, k, eps)
#     ax1.plot(x, y[2], marker='o', label=f'$\epsilon ={eps:.2e}$')

# eps = 1e-100
# y = MPS(d, L, c, k, eps)
# ax1.plot(x, y[2], marker='o', color='black', label=f'$\epsilon ={eps}$')

# # Plot horizontal dashed line at y = d^(L/2)
# #ax1.axhline(y=d**(L/2), linestyle='--', color='black', label=f'$d^{{L/2}}$')

# ax1.set_ylabel('EE', fontsize='x-large')
# ax1.tick_params(axis='both', which='major', labelsize='medium')
# ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='medium')
# ax1.set_yscale('log')
# ax1.grid(True)  # Enable grid
# ax1.tick_params(axis='both', which='major', labelsize='x-large')  # Increase tick label size

# # ax2------
# epsilons = np.linspace(0.01, 0.03, 2)
# for eps in epsilons:
#     y = MPS(d, L, c, k, eps)
#     ax2.plot(x, y[2], marker='o', label=f'$\epsilon ={eps:.2e}$')

# ax2.set_xlabel('car', fontsize='x-large')
# ax2.set_ylabel('EE', fontsize='x-large')
# ax2.tick_params(axis='both', which='major', labelsize='medium')
# ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='medium')
# ax2.set_yscale('log')
# ax2.grid(True)  # Enable grid
# ax2.tick_params(axis='both', which='major', labelsize='x-large')  # Increase tick label size

# plt.tight_layout()

# # Save and show the final plot
# filename = 'log_EE_random'
# #plt.savefig('/Users/gabinleroy/Desktop/master_thesis_report/' + filename + '_interactive.pdf', format='pdf')
# plt.show()

#plot EE and compared with max EE = log bondd, on same plot
# ---------------------------------------------------------------------------------------------------------------
# d = 2
# L = 9
# c = np.random.uniform(0, 1, size=(1, d**L))
# norm_c = np.linalg.norm(c)
# c = c/norm_c
# x = np.arange(L-1)
# k = 1000000000000
# fig, (ax1) = plt.subplots(1)
# epsilon = 1e-100
# x = np.arange(L-1)
# bondd, EE = MPS(d,L,c,k,epsilon)[1], MPS(d,L,c,k,epsilon)[2] 
# #print(bondd)
# logbondd = []
# for el in bondd:
#     logbondd.append(math.log2(el))
# print(logbondd)
# print(EE)
# ax1.plot(x, logbondd, marker='o', color='black', label='log Bond dim.')
# ax1.plot(x, EE, marker='x', color='magenta', label='EE')
# ax1.plot(x, bondd, marker='o', color='blue', label='Bond dim.')
# ax1.set_title(f'Random $\psi$ $\sim U(0,1)$, $d=${d}, $L=${L}, $\epsilon={epsilon}$',fontsize='xx-large')
# ax1.grid(True)  # Enable grid
# ax1.tick_params(axis='both', which='major', labelsize='xx-large')  # Increase tick label size
# ax1.set_ylabel('', fontsize='xx-large')
# ax1.set_xlabel('car', fontsize='xx-large')
# ax1.legend(fontsize='xx-large')  # Use fontsize to set the legend font size
# plt.show()

#plot EEor bond for unif random data in [0,1], in context of spin glass
# ---------------------------------------------------------------------------------------------------------------
# R = 7
# d = 2
# L = 3*R
# c = np.random.uniform(0, 1, size=(1, d**L))
# norm_c = np.linalg.norm(c)
# c = c/norm_c
# x = np.arange(L-1)
# k = int(1e20)
# directory = 'random'
# #fig, (ax1) = plt.subplots(1)
# x = np.arange(L-1)
# epsilons = [1e-05]
# for epsilon in epsilons:
#     mps = MPS(d,L,c,k,epsilon)
#     bondd = mps[1]
#     EE = mps[2]
#     title_par_bondd = {'random_bondd':'_','locald':d, 'rank':L, 'eps':epsilon}
#     title_par_EE = {'random_EE':'_','locald':d, 'rank':L, 'eps':epsilon}
#     save_list_as_txt(bondd, directory, title_par_bondd, '\t')
#     save_list_as_txt(EE, directory, title_par_EE, '\t')
#ax1.plot(x, logbondd, marker='o', color='black', label='log Bond dim.')
#ax1.plot(x, EE, marker='x', color='magenta', label='EE')
#ax1.plot(x, bondd, marker='o', color='blue', label='Bond dim.')
#ax1.set_title(f'Random $\psi$ $\sim U(0,1)$, $d=${d}, $L=${L}, $\epsilon={epsilon}$',fontsize='xx-large')
#ax1.grid(True)  # Enable grid
#ax1.tick_params(axis='both', which='major', labelsize='xx-large')  # Increase tick label size
#ax1.set_ylabel('EE', fontsize='xx-large')
#ax1.set_xlabel('car', fontsize='xx-large')
#ax1.legend(fontsize='xx-large')  # Use fontsize to set the legend font size
#plt.show()
    
#save EE or bond for uniform. point on unit sphere
# ---------------------------------------------------------------------------------------------------------------
#dimention of space where spin glass live
# D = 3
# R = 8
# d = 2
# L = 2*R

# vec = np.random.randn(D, d**L)
# vec /= np.linalg.norm(vec, axis=0)

# zi = vec[2]
# zi = zi/np.linalg.norm(zi)

# x = np.arange(L-1)
# k = int(1e20)
# directory = 'random'

# epsilons = np.linspace(0.0025,0.5,40)
# for epsilon in epsilons:
#     mps = MPS(d,L,zi,k,epsilon)
#     #bondd = mps[1]
#     #EE = mps[2]
#     #title_par_bondd = {'random_bondd':'_','locald':d, 'rank':L, 'eps':epsilon}
#     #title_par_EE = {'random_EE':'_','locald':d, 'rank':L, 'eps':epsilon}
#     #save_list_as_txt(bondd, directory, title_par_bondd, '\t')
#     #save_list_as_txt(EE, directory, title_par_EE, '\t')
    
#     #compute error and save
#     approx_data = approximate_data(2,(2*R),mps,k,epsilon)
#     error =  np.linalg.norm(zi - approx_data)
#     full_file_name = f'errors/random_error_locald2_rank{2*R}_k{k}_eps{epsilon}.txt'
#     add_to_txt(np.array([epsilon, error]),  full_file_name, True)

#distribution of singular value for random unif data
# d = 2
# L = 9
# c = np.random.uniform(0, 1, size=(1, d**L))
# norm_c = np.linalg.norm(c)
# c = c/norm_c
# x = np.arange(L-1)
# k = 1000000000000
# add_par = {}
# MPS_sev_eps(d,L,c,k,[0.02,0.2],add_par,True)


