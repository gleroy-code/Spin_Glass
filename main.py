import numpy as np
from reading_data import manipulate_file
from reading_tota_data import MPS_on_totaslice, MPS_on_totacube


def main():
    print("Executing main function ------------------------------------ :")
    
    #for snapshots ---------------------------------------------------------------------------------------------------
    #if want user to enter parameter, decomment this ------------------------------------------
    # Get user parameters
    #alpha = input("Enter alpha : ")
    #alphai = int(input("Enter alphai : "))
    #k = int(input("Enter k : "))
    #eps = float(input("Enter eps : "))
    #typeplot = int(input("Enter typeplot of MPS : "))
    #separator = '  '
    #component = input("Enter component of spin : ")
    #par = [alpha,alphai,[k,eps],typeplot, separator, component]
    #get main directory path
    #main_dir = input("Enter directory of tota N. data : ")

    #if you want to enter parameters in code directly, decomment this --------------------------
    epsilons = [0.002]
    alpha = 'x'
    # Generate 6 evenly spaced integer points between 1 and 286
    #alphais = np.linspace(1, 286, 6, dtype=int)
    alphais = [150]
    k = int(1e20)
    typeplot = 1
    component = 'y'
    main_dir = '/Users/gabinleroy/Desktop/spinglass_python/workspace_spinglass/test_data2'
    for eps in epsilons:
        for alphai in alphais:
            par=[alpha,int(alphai),[k,eps],typeplot, '  ', component]
            #execute function
            manipulate_file(main_dir, MPS_on_totaslice, par)
            #save parameters in parameters.txt
            #parameter_file_path = '/Users/gabinleroy/Desktop/spinglass_python/workspace_spinglass/parameters.txt'
            #add_to_txt( ['','alpha = '+str(alpha), 'alphai = '+str(alphai), 'k = '+str(k), 'eps = '+str(eps), 'typeplot = '+str(typeplot), 'component = '+component], parameter_file_path)




    #for 3D data ---------------------------------------------------------------------------------------------------
    #if you want to enter parameters in code directly, decomment this
    #epsilons = [0.0001, 0.001, 0.005, 0.007, 0.01, 0.2]
    #epsilons = np.linspace(0.01, 0.2, 2)
    # epsilons = [0.00003]
    # k = int(1e20)
    # cut = 10
    # typeplot = 1
    # component = 'z'
    # main_dir = '/Users/gabinleroy/Desktop/spinglass_python/workspace_spinglass/tota_n_data3D'
    # for eps in epsilons:
    #     print(eps)
    #     par=[cut,[k,eps],typeplot, '  ', component]
    #     #execute function
    #     manipulate_file(main_dir, MPS_on_totacube, par)
    #     #save parameters in parameters.txt
    #     #parameter_file_path = '/Users/gabinleroy/Desktop/spinglass_python/workspace_spinglass/parameters.txt'
    #     #add_to_txt( ['','alpha = '+str(alpha), 'alphai = '+str(alphai), 'k = '+str(k), 'eps = '+str(eps), 'typeplot = '+str(typeplot), 'compone$

if __name__ == "__main__":
    main()

    