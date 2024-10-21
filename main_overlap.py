import numpy as np
from reading_data import manipulate_file
from reading_tota_data import MPS_on_totaslice,create_overlap,overlap_function



def main():
    print("Executing main function ------------------------------------ :")
     
    #to generate data of raw snapshots--------------------------------
    # epsilons = [0.002,0.01,0.03]
    # alpha = 'z'
    # alphais = [150]
    # k = int(1e20)
    # typeplot = 1
    # main_dir = '/Users/gabinleroy/Desktop/spinglass_python/workspace_spinglass/tota_n_data'
    # components = ['x','y','z']

    # for component in components:
    #     for eps in epsilons:
    #         for alphai in alphais:
    #             par=[alpha,int(alphai),[k,eps],typeplot, '  ',component]
    #             #execute function
    #             manipulate_file(main_dir, MPS_on_totaslice, par)
    


    #to generate the overlap, assuming raw data snapshot file already exist in n287txxx repertories  ----------------------------------
    # directory_path = '/Users/gabinleroy/Desktop/spinglass_python/workspace_spinglass/tota_n_data'
    # pairsA = [[0,1]]
    # pairscomponent = [['x','x'],['y','y'],['z','z']]
    # a = 0
    # T = 2100
    # alpha = 'z'
    # alphai=150

    # for A in pairsA:
    #     for comp in pairscomponent:

    #         A_value1 = A[0]
    #         A_value2 = A[1]
    #         component1 = comp[0]
    #         component2 = comp[1]
    #         result_file_path = create_overlap(directory_path, T,alpha,alphai,a,component1,component2, A_value1, A_value2, overlap_function)


    #to work on overlap - generate MPS plots---------------------------------------
    # epsilons = np.linspace(0.0025,0.5,40)
    # alpha = 'z'
    # alphais = [150]
    # k = int(1e20)
    # typeplot = 1
    # component = True

    # main_dir = '/Users/gabinleroy/Desktop/spinglass_python/workspace_spinglass/tota_n_data'
    # for eps in epsilons:
    #     for alphai in alphais:
    #         par=[alpha,int(alphai),[k,eps],typeplot, '  ',component]
    #         #execute function
    #         manipulate_file(main_dir, MPS_on_totaslice, par)
           

if __name__ == "__main__":
    main()

    