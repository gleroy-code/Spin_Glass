import numpy as np
import matplotlib.pyplot as plt
import random
import glob
import os
import math

def create_2Dscatter_plot(y, title_parameters, axis_parameters, directory_path):
    """
    Create a hexbin plot with 'parameters' in title and filename and save it to the specified directory.

    Parameters:
    - y: 2D array of data. Row index will correspond to xaxis and column index to yaxis
    - title_parameters: Dictionary of parameters for title
    - axis_parameters: Dictionary of parameters for axis
    - directory_path: Directory path to save the plot
    

    Returns:
    - None
    """
    # Create a figure
    #fig, ax = plt.subplots()

    # Create flattened indices for x and y
    x_indices, y_indices = np.meshgrid(range(y.shape[0]), range(y.shape[1]))

    # Create a hexbin plot with colors corresponding to the values in the array
    #hb = ax.hexbin(x_indices.flatten(), y_indices.flatten(), C=y.flatten(), cmap='viridis', gridsize=50)
    plt.matshow(y,cmap='gnuplot')

    # Set the title with parameters
    title = ', '.join([f'{key}={value}' for key, value in title_parameters.items()])
    #fig.suptitle(f'{title}')
    plt.title(f'{title}')

    #set legend on axis
    plt.xlabel(axis_parameters.get('xaxis'))
    plt.ylabel(axis_parameters.get('yaxis'))
     
    # Add a colorbar
    cbar = plt.colorbar(shrink=0.7)
    #cbar.set_label(f'$S_z$')

    # Construct the file name with parameters and the correct file format
    file_name = 'hexbin_plot_' + '_'.join([f'{key}{value}' for key, value in title_parameters.items()]) + '.pdf'

    # Save the figure to the specified directory with the constructed file name
    plt.savefig(directory_path + '/' + file_name, format='pdf', bbox_inches='tight', dpi=1500)

    #Show the plot (optional)
    #plt.show()

# Example usage:
#directory_path = '/Users/gabinleroy/Desktop/spinglass_python'
#parameters = {'x1': 1.0, 'x2': 2.0, 'x3': 3.0}  # Replace with your actual parameters
#y = np.random.rand(200, 200)  # Increase the size to demonstrate avoiding overlap

#create_2Dscatter_plot(y, parameters, directory_path)



def generate_list_of_random_floats(n, a, b):
    '''generates a list of n elements, where each element is uniformly drawn between a and b 
    '''
    return [random.uniform(a, b) for _ in range(n)]

def create_file_unifrandom(directory, n_rows, n_cols, a, b, c):
    '''
    create a .txt file 'n_rows' rows and 'n_cols' cols, in the directory "directory", where each element is uniformly drawn in [a,b], lines are separated \n, cols by c
    '''
    filename = f"n_rows{n_rows}_n_cols{n_cols}_uniformly_in_[{a},{b}].txt"
    full_file_path = directory+"/"+filename
        
    # Oen the file
    try:
        with open(full_file_path, 'w') as file:
          for _ in range(n_rows):
            floats = generate_list_of_random_floats(n_cols, a, b)
            line = c.join(map(str, floats))
            file.write(line + "\n")
    except FileNotFoundError:
        print(f"File '{filename}' not found in directory '{directory}'")
    except Exception as e:
        print(f"Error opening file: {e}")

#example use of create_file_unifrandom
#L = 115
#a = 1.0
#b = 2.0
#col_separator = ','
#n_rows = (L)*(L)*(L+1)
#n_cols = 3
#directory = '/Users/gabinleroy/Desktop/spinglass_python/test_data/n115t2900'
#create_file_unifrandom(directory,n_rows,n_cols, a, b,col_separator)

#example use of create_2Dscatter_plot()
#M = np.ones((5,15))
#title_parameters = {'Test':1}
#axis_parameters ={'xaxis':'un', 'yaxis':'deux'}
#dir = '/Users/gabinleroy/Desktop/spinglass_python/test_data'
#create_2Dscatter_plot(M, title_parameters, axis_parameters, dir) 
        
def save_2Darray_as_txt(y,directory,title_parameters,col_separator):
    '''
    saves y = 2D np.array in a .txt file where each line = row of array
    title_parameters = dictionnary, used for title of the .txt file
    '''
    # Create the title for the text file using parameters from the dictionary
    title = "_".join([f"{key}{value}" for key, value in title_parameters.items()])
    filename = f"{title}.txt"
    filepath = os.path.join(directory, filename)

    try:
        # Save the 2D array to the text file
        np.savetxt(filepath, y, delimiter=col_separator, fmt='%g')
        print(f"Array saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving array to {filename}: {e}")

def save_list_as_txt(y,directory,title_parameters,col_separator):
    """
    Saves a list 'lst' to a .txt file in the specified 'directory'.
    If 'lst' is 1D, each element is written on a new line. If 'lst' is 2D,
    each line in the file corresponds to a sublist of the list.
    'title_parameters' is a dictionary used to create the title of the .txt file.
    """

    # Create the title for the text file using parameters from the dictionary
    title = "_".join([f"{key}{value}" for key, value in title_parameters.items()])
    filename = f"{title}.txt"
    filepath = os.path.join(directory, filename)

    try:
        # Save the list to the text file
        with open(filepath, 'w') as file:
            if all(isinstance(item, list) for item in y):
                for sublist in y:
                    file.write(col_separator.join(map(str, sublist)) + '\n')
            else:
                file.write('\n'.join(map(str, y)) + '\n')
        
        print(f"List saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving list to {filename}: {e}")

def add_to_txt(y,directory_of_file, row_by_row=False):
    '''
    parameters :- y is a list
                - row_by_row = boolean
                -directory__of_file is a str, the path of a file in which you want to add the element of the list. The element are added either 
                    - y[i] -> one row, y[i+1] -> the next row and same col if row_by_row = False
                    - y[i] -> one col, y[i+1] -> the next col and same row if row_by_row = True    
    '''
    try:
        with open(directory_of_file, 'a') as file:
            if row_by_row:
                # Add elements row-wise
                for element in y:
                    file.write(str(element) + ' ')
                file.write('\n')
            else:
                # Add elements column-wise
                for element in y:
                    file.write(str(element) + '\n')
    except Exception as e:
        print(f"Error: {e}")
    

# Example usage of add_to_txt()
#directory_path = '/Users/gabinleroy/Desktop/spinglass_python/workspace_spinglass/ooo.txt'
#data_to_add = [1.00000,24,23,2]
#Adding elements row-wise
#add_to_txt(data_to_add, directory_path,True)

def extract_from_file_name(txt_file, extract_prefix, extract_suffix):
    '''
    given a string txt_file, it extract all characters between extract_prefix and extract suffix.
    For example, gabinelmartin output 'el' if extract_prefix = 'gabin' and extract_suffix = 'martin'
    '''
    start_index = txt_file.find(extract_prefix)
    end_index = txt_file.find(extract_suffix)

    if start_index != -1 and end_index != -1 and start_index < end_index:
        # Extract characters between prefix and suffix
        extracted_text = txt_file[start_index + len(extract_prefix):end_index]
        return extracted_text
    else:
        # Prefix or suffix not found or in the wrong order
        return ""
#example use of extract_from_file_name
#txt_file = 'alphas_eps0.01_alphai100.txt'
#extract_prefix = 'eps'
#extract_suffix = '_alpha'
#result = extract_from_file_name(txt_file, extract_prefix, extract_suffix)
#print(result)  # Output: 'el'

def plot_txt_file(directory_path, file_prefix, file_suffix, title_parameters, xlabel, ylabel, output_fig_directory, extract_preffix, extract_suffix, epsilon_color_mapping,compteur):
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
    
    L = {'0.0275':False, '0.0325':False, '0.0375':False, '0.022500000000000003':False, '0.030000000000000002':False, '0.034999999999999996':False, '1e-05':False}


    # Iterate over each file and plot its data
    for txt_file in txt_files:
        epsilon = extract_from_file_name(txt_file, extract_preffix, extract_suffix)
        print(epsilon)
        # Extract data from the file (modify this part based on your data format)
        with open(txt_file, 'r') as file:
            data = [float(line.strip())+1e-20 for line in file]

        # Plot the data with assigned color
        color = epsilon_color_mapping.get(epsilon)
        if L.get(epsilon)==False:
            plt.plot(data, marker='o', linestyle='-', color=color, label='SG')
            L[epsilon]=True
        else:
            plt.plot(data, marker='o', linestyle='-', color=color)
    random_data = [2,
4,
8,
16,
32,
64,
128,
256,
469,
390,
271,
188,
136,
100,
64,
32,
16,
8,
4,
2]

    plt.plot(random_data, marker='x', linestyle = '-', color='black', label='Random')
    title = ', '.join([f'{key}:{format(value, ".3e")}' if isinstance(value, (int, float)) else f'{key}:{value}' for key, value in title_parameters.items()])
    filename = '_'.join([f'{key}{value}' for key, value in title_parameters.items()])
    # Add labels, title, legend, etc. (modify as needed)
    plt.xlabel(xlabel,fontsize='xx-large')
    plt.ylabel(ylabel,fontsize='xx-large')
    plt.title(title,fontsize='large')
    if compteur==0:
        plt.legend(fontsize='xx-large')
    plt.xticks(fontsize='xx-large')
    plt.yticks(fontsize='xx-large')
    #plt.yscale('log')  # Set y-axis to log scale
    plt.grid('True')
    #plt.xlim(9,10)
    #plt.ylim(-2,2)
   # Save the figure in PDF format
    output_fig_path = os.path.join(output_fig_directory, filename+'.pdf')
    plt.savefig(output_fig_path, format='pdf')

    # Display the plot (optional)
    plt.show()

#Example usage of plot_txt_file, if you want one plot per slice parameter
# alpha = 'z'
# k = 100000000000000000000
# a = 0
# S = 'z'
# file_prefix = 'datafromplotofBonddimension_locald2_rank21_k100000000000000000000_eps0.0325'
# xlabel = 'car'
# ylabel = 'Bond dim.'
# extract_prefix='eps'
# extract_sufix = '_cut'
# # Choose a color map
# color_map = plt.cm.get_cmap('tab10')
# # Generate 5 distinct colors
# epsilon_color_mapping = {'0.0275':color_map(0), '0.0325':color_map(1), '0.0375':color_map(2), '0.022500000000000003':color_map(3), '0.030000000000000002':color_map(4), '0.034999999999999996':color_map(5),'1e-05':color_map(6)}

# compteur= 0
# for T in [1500,1945,1960,2100]:
#     directory_path = '/Users/gabinleroy/Desktop/spinglass_python/workspace_spinglass/tota_n_data3D/n287t'+str(T)
#     #for alphai in [1, 58, 115, 172, 229, 286]:
#     file_sufix = '_cut129_Lx286_Ly286_T'+str(float("0."+str(T)))+'_a0_meanA_PlottedSz.txt'
#     title_parameters = {'local_dim':2, 'rank':21, 'k':k, 'a':a,  'mean':'A', 'PlottedS':S, 'T':float("0."+str(T)), 'eps':0.0325
#                       }
#     plot_txt_file(directory_path, file_prefix, file_sufix, title_parameters, xlabel, ylabel, directory_path, extract_prefix, extract_sufix, epsilon_color_mapping, compteur)
#     compteur+=1

#same, if you want to superpose all the slices:
# alpha = 'z'
# k = 100000000000000000000
# a = 0
# S = 'z'
# file_prefix = 'datafromplotofBonddimension_locald2'
# xlabel = 'car'
# ylabel = 'mean over A of Bond dimension '
# extract_prefix='eps'
# extract_sufix = '_alpha'
# # Choose a color map
# color_map = plt.cm.get_cmap('tab10')
# # Generate 5 distinct colors
# epsilon_color_mapping = {'0.0001':color_map(0), '0.001':color_map(1), '0.005':color_map(2), '0.007':color_map(3), '0.01':color_map(4), '0.2':color_map(5)}

# for T in [1500, 1945, 1960, 2100]:
#     file_sufix = str(float("0."+str(T)))+'_a0_meanA_PlottedSz.txt'
#     directory_path = '/Users/gabinleroy/Desktop/spinglass_python/workspace_spinglass/tota_n_data/n287t'+str(T)
#     title_parameters = {'logMean over A ':'EE', 'locald':2, 'rank':16, 'k':k, 'alpha':alpha, 'alphai':'6eventlyspaced', 'a':a,  'mean':'A', 'PlottedS':S, 'T':float("0."+str(T))}
#     plot_txt_file(directory_path, file_prefix, file_sufix, title_parameters, xlabel, ylabel, directory_path, extract_prefix, extract_sufix, epsilon_color_mapping)       


def calculate_mean_and_save(directory_path, file_prefix, file_suffix,  output_file_directory, filename):
    '''
    writes a file containg as filename the title_parameters, saves it in 'output_file_directory'
    this file is such that each row = mean value of the same row over all file containg 'file_prefix' in their name, located in 'directory path'
    '''

    # Use glob to find files with the specified suffix in the directory
    txt_files = glob.glob(os.path.join(directory_path, f'{file_prefix}*{file_suffix}'))

    if not txt_files:
        print(f"No matching files found with the name including : "+file_prefix)
        return    

    # List to store mean values for each row
    mean_values = []

    # Iterate over the rows in each file
    for row_number in range(get_number_of_rows(txt_files[0])):
        # List to store values for the current row across all files
        row_values = []

        # Iterate over each file
        for txt_file in txt_files:
            # Open the file and read the value at the current row
            with open(txt_file, 'r') as file:
                lines = file.readlines()
                row_values.append(float(lines[row_number].strip()))

        # Calculate the mean for the current row and append to mean_values
        mean_value = sum(row_values) / len(row_values)
        mean_values.append(mean_value)

    # Save mean values in a new file with name containing parameters
   
    
    output_file_path = os.path.join(output_file_directory, filename)
    with open(output_file_path, 'w') as output_file:
        for mean_value in mean_values:
            output_file.write(f"{mean_value}\n")

def get_number_of_rows(file_path):
    # Function to get the number of rows in a file
    with open(file_path, 'r') as f:
        return len(f.readlines())

def neumann(x):
   # Ensure x is positive for the logarith
    #print('neumann on' + str(x))
    if x<10e-15:
        return 0
    else:
        result = -x * math.log2(x)
        #print(result)
        return result  
    
def random_number_plots(directory,m, xlabel, ylabel, title, a, b, filename):
    '''
    plots a r uniformly distributed numbers in [a,b], in decreasing order
    '''
    L = generate_list_of_random_floats(m-10,a,b)
    for i in range(0,10):
        L.append(0)
    L = [1/10]*10
    #L = [1]
    for i in range(0,m-10):
        L.append(0)
    L.sort(reverse=True)  # Sort L in decreasing order
    x = np.arange(len(L))
    plt.figure()
    plt.plot(x,L, marker='o', color = 'b')
    plt.xlabel(xlabel,fontsize='xx-large')
    plt.ylabel(ylabel,fontsize='xx-large')
    plt.title(title,fontsize='xx-large')
    plt.grid(axis='x', linestyle='--')
    
    # Set x-axis ticks and grid lines only at the points specified by x
    plt.xticks(np.arange(0, len(L), step=4), fontsize='xx-large')
    plt.yticks(fontsize='xx-large')

    plt.grid('True')
    #plt.savefig(directory+'/'+ filename+'.pdf', format = 'pdf' )
    plt.show()

# directory = '/Users/gabinleroy/Desktop/spinglass_python/workspace_spinglass/'
# filename = 'D_20_numbers'
# m=20
# a=0
# b=1
# title = f'$r=10$, $m$={m}, $s_i^2=1/r$'
# xlabel = f'$i$'
# ylabel = f'$s_i^2$'
#random_number_plots(directory,20, xlabel,ylabel,title,a,b,filename)



def generate_random_floats():
    return [random.uniform(0.0, 1.0) for _ in range(3)]

def create_testfile(Lx, Ly, Lz):
    with open("/Users/gabinleroy/Desktop/spinglass_python/workspace_spinglass/test_data/n005t1500/test.txt", "w") as file:
        for _ in range((Lx+1) * (Ly+1) * (Lz+1)):
            floats = generate_random_floats()
            line = ",".join(map(str, floats))
            file.write(line + "\n")

#example use of create_testfile
# Replace Lx, Ly, Lz with your desired values
#=5
#Lx = L-1
#Ly = L-1
#Lz = L

#create_testfile(Lx, Ly, Lz)
            
def file_as_list(file_path, delimiter='\t'):
    '''
    Reads a file and converts its contents into a NumPy array.

    Parameters:
    - file_path (str): The path to the file.
    - delimiter (str): The delimiter used to separate columns in each line. Default is space (' ').

    Returns:
    - numpy.ndarray: A NumPy array containing the values from the file.
    '''
    try:
        with open(file_path, 'r') as file:
            file_array = np.genfromtxt(file, delimiter=delimiter)
        return file_array
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage--------------------------------------------------------:
# c = 'random_error_locald2_rank16_k100000000000000000000_eps0.05144736842105263.txt'
# file_path = f'errors/{c}'
# lines_list = file_as_list(file_path, ' ')
# print(lines_list[1])