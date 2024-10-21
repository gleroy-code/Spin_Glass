import os

def collect_filenames_onedirectory(main_directory):
    '''
    given a main_directory, read the name of all files in it. If the main directory contains folder, it will output
    also '.DS_Store', this is mac that procudes hidden folder
    '''
    
    L = []
    
    for filename in os.listdir(main_directory):
        L.append(filename)

    return L

import os
def print_file_info(dirpath,filename,par):
    '''
    context : file in a certain directory
    prints filename and its path and also the name of the directory where it is in
    '''
    
    print("")
    print('dirpath ='+str(dirpath))
    print('dirname ='+os.path.basename(dirpath))
    print('filename='+str(filename))

import os

def manipulate_file(main_directory,g=print_file_info, gpar=[]):
    '''
    goes over all directories and subdirectories of a main_directory, and read all the filenames and their path,
    and it applies a function g on these files
    g is a function g(dirpath,filename,gpar) where gpar=list of additional parameters
    by default, this function just goes over all possible files and print their directory, filename, and dirname
    '''
    

    # Iterate over all directories and subdirectories using os.walk
    for dirpath, dirnames, filenames in os.walk(main_directory):
        for filename in filenames:
            #print_file_info(dirpath,filename,gpar)
            g(dirpath,filename,gpar)

    

#Example usage:
#main_directory = "/Users/gabinleroy/Desktop/spinglass_python"
#all_files = manipulate_file(main_directory, MPS_on_totaslice, ['x',0,[200000,0.001],1])
