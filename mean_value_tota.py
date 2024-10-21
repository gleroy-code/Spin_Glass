from general_functions import calculate_mean_and_save
import os

#for snapshots ---------------------------------------------------------------------------------------------------

def EE_fixed_T_eps_k_alpha_alphai_S_a_meanA(directory_path, alpha, alphai, eps, k, a, S):
    '''
    goes over all file in n287tT, which have file name 'datafromplotofEE_locald2_rank16_k' + str(k) + '_eps' + str(eps) + '_alphaz_alphai100_Lx286_Ly286_T0.21_a' + str(a) + '_A'+value of A + '_PlottedS' + S + '.txt'
    directpry_path  = path of folder n287tT
    returns mean value over all replicas = thermal average
    '''
    T=float("0."+os.path.basename(directory_path)[5:])
    file_prefix = 'datafromplotofEE_locald2_rank16_k' + str(k) + '_eps' + str(eps) + '_alpha'+str(alpha)+'_alphai'+str(alphai)+'_Lx286_Ly286_T'+str(T)+'_a' + str(a)
    file_sufix = 'PlottedS'+S+'.txt'
    filename = file_prefix + '_meanA_' + file_sufix
    output_file_directory = directory_path
    calculate_mean_and_save(directory_path, file_prefix, file_sufix, output_file_directory, filename)

def bondd_fixed_T_eps_k_alpha_alphai_S_a_meanA(directory_path, alpha, alphai, eps, k, a, S):
    '''
    goes over all file in n287tT, which have file name 'datafromplotofBonddimension_locald2_rank16_k' + str(k) + '_eps' + str(eps) + '_alphaz_alphai100_Lx286_Ly286_T0.21_a' + str(a) + '_A'+value of A + '_PlottedS' + S + '.txt'
    directpry_path  = path of folder n287tT
    returns mean value over all replicas = thermal average
    '''
    T=float("0."+os.path.basename(directory_path)[5:])
    file_prefix = 'datafromplotofBonddimension_locald2_rank16_k' + str(k) + '_eps' + str(eps) + '_alpha'+str(alpha)+'_alphai'+str(alphai)+'_Lx286_Ly286_T'+str(T)+'_a' + str(a)
    file_sufix = 'PlottedS'+S+'.txt'
    filename = file_prefix + '_meanA_' + file_sufix
    output_file_directory = directory_path
    calculate_mean_and_save(directory_path, file_prefix, file_sufix, output_file_directory, filename)

#Example usage:
# alpha = 'z'
# k = 100000000
# a = 0
# S = 'z'
# for T in [1500, 1945, 1960, 2100]:
#     directory_path = '/Users/gabinleroy/Desktop/spinglass_python/workspace_spinglass/tota_n_data/n287t'+str(T)
#     for alphai in [1, 58, 115, 172, 229, 286]:
#         for eps in [1e-100, 1e-09, 0.01, 0.05, 0.07, 0.08]:
#             bondd_fixed_T_eps_k_alpha_alphai_S_a_meanA(directory_path, alpha, alphai, eps, k, a, S)

#for 3D data, with cut=129---------------------------------------------------------------------------------------------------

def EE_fixed_T_eps_k_alpha_alphai_S_a_meanA_3D(directory_path, eps, k, a, S):
    '''
    goes over all file in n287tT, which have file name 'datafromplotofEE_locald2_rank21_k' + str(k) + '_eps' + str(eps) + '_cut129_Lx286_Ly286_TXXXX_a' + str(a) + '_A'+value of A + '_PlottedS' + S + '.txt'
    directpry_path  = path of folder n287tT
    returns mean value over all replicas = thermal average
    '''
    T=float("0."+os.path.basename(directory_path)[5:])
    file_prefix = 'datafromplotofEE_locald2_rank21_k' + str(k) + '_eps' + str(eps) + '_cut129_Lx286_Ly286_T'+str(T)+'_a' + str(a)
    file_sufix = 'PlottedS'+S+'.txt'
    filename = file_prefix + '_meanA_' + file_sufix
    output_file_directory = directory_path
    calculate_mean_and_save(directory_path, file_prefix, file_sufix, output_file_directory, filename)

def bondd_fixed_T_eps_k_alpha_alphai_S_a_meanA_3D(directory_path, eps, k, a, S):
    '''
    goes over all file in n287tT, which have file name 'datafromplotofEE_locald2_rank21_k' + str(k) + '_eps' + str(eps) + '_cut129_Lx286_Ly286_TXXXX_a' + str(a) + '_A'+value of A + '_PlottedS' + S + '.txt'
    directpry_path  = path of folder n287tT
    returns mean value over all replicas = thermal average
    '''
    T=float("0."+os.path.basename(directory_path)[5:])
    file_prefix = 'datafromplotofBonddimension_locald2_rank21_k100000000000000000000' + '_eps' + str(eps) + '_cut129_Lx286_Ly286_T'+str(T)+'_a' + str(a)
    file_sufix = 'PlottedS'+S+'.txt'
    filename = file_prefix + '_meanA_' + file_sufix
    output_file_directory = directory_path
    calculate_mean_and_save(directory_path, file_prefix, file_sufix, output_file_directory, filename)


#Example usage:
# alpha = 'z'
# k = 100000000000000000000
# a = 0
# S = 'z'
# for T in [1500,1945,1960,2100]:
#     directory_path = '/Users/gabinleroy/Desktop/spinglass_python/workspace_spinglass/tota_n_data3D/n287t'+str(T)
#     for eps in [1e-05]:
#         EE_fixed_T_eps_k_alpha_alphai_S_a_meanA_3D(directory_path, eps, k, a, S)
    
   