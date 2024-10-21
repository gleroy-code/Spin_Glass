import os
import numpy as np
import matplotlib.pyplot as plt

from mps import find_R_from_snapshotsize, MPS, image_to_vector_bits, data3D_to_vector_bits, approximate_data, from_approximatedata_to_original_snapshot
from MPS_plot import save_plot_MPS
from general_functions import create_2Dscatter_plot, save_2Darray_as_txt, add_to_txt

import sys

def mapping_tota(x,y,z,Lx,Ly,Lz):
    '''
    context : consider a cubic lattice where x =0,...,Lx ; y=0,...,Ly ; z=0,...,Lz
              total number of points = (Lx+1)*(Ly+1)*(Lz+1)
    output : for a given coordinate (x,y,z) in this cubic lattice, returns the corresponding row in tota mapping
     '''
    if 0 <= x <= Lx and 0 <= y <= Ly and 0 <= z <= Lz:
        return int(x + y*(Lx+1) + z*(Lx+1)*(Ly+1))
    else:
        print('You ask me to find ')


def site_data(S, component):
    '''
    given [Sx,Sy,Sz], return the component 'component' of the spin
    component must be string in {‘x','y','z'} 
    '''
    dic = {'x':0, 'y':1, 'z':2}
    if type(component)!=str:
        raise TypeError('Invalid type for component. Must be string')
    if component not in {'x','y','z'}:
        raise ValueError('Invalid value for component. Must be in (x,y,z)')
    else:
        return S[dic.get(component)]


def snapshot(filename,directory,alpha,alphai,Lx,Ly,Lz,separator, component):
    '''
    context : cubic lattice of size (Lx+1)*(Ly+1)*(Lz+1) where coordinates start from 0
    output : slice of the cubic lattice. Slice = matrix, where each point is a function of the data in 'filename' 
    associated to that point. 
    '''
    #check type of lattice dimension
    if type(Lx) != int or type(Ly)!= int or type(Lz)!=int:
        raise TypeError("Invalid type for Lx or Ly or Lz. Should be integer")

    #check type and value for alpha
    if type(alpha)!=str: 
        raise TypeError("Invalid type for alpha. Alpha must be a string")
    if alpha not in {'x', 'y', 'z'}:
        raise ValueError("Invalid value for alpha. Should be one of {'x', 'y', 'z'}")
    
    #check if alphai has correct dimension and correct type
    lattice_dim = {'x':Lx, 'y':Ly, 'z':Lz}
    if type(alphai)!= int:
        raise TypeError("Invalid type for alphai. Alpha must be integer")
    if alphai<0 or alphai>lattice_dim.get(alpha):
        raise ValueError("Invalid value of alphai. Should be integer and in [O,Lalpha]")


    # Construct the full file path
    full_file_path = directory + '/' + filename

    # Open the file
    try:
        with open(full_file_path, 'r') as file:
            content = file.readlines()
           
            if alpha == 'x': 
                M = np.zeros((Lz+1,Ly+1))
                for i in range(0,Lz+1):
                    for j in range(0,Ly+1):
                        line = mapping_tota(alphai,j,i,Lx,Ly,Lz)
                        # Convert each substring to a float using map and float, .strip() to remove leading and trailing whitespaces 
                        float_values = list(map(float, content[line].strip().split(separator)))
                        M[i,j]=site_data(float_values, component)

            if alpha == 'y':
                M = np.zeros((Lz+1,Lx+1))
                for i in range(0,Lz+1):
                    for j in range(0,Lx+1):
                        line = mapping_tota(j,alphai,i,Lx,Ly,Lz)
                        # Convert each substring to a float using map and float, .strip() to remove leading and trailing whitespaces 
                        float_values = list(map(float, content[line].strip().split(separator)))
                        M[i,j]=site_data(float_values, component)

            if alpha == 'z':
                M = np.zeros((Ly+1,Lx+1))
                for i in range(0,Ly+1):
                    for j in range(0,Lx+1):
                        line = mapping_tota(j,i,alphai,Lx,Ly,Lz)
                        # Convert each substring to a float using map and float, .strip() to remove leading and trailing whitespaces 
                        float_values = list(map(float, content[line].strip().split(separator)))
                        M[i,j]=site_data(float_values, component)
                        
    except FileNotFoundError:
        print(f"File '{filename}' not found in directory '{directory}'")
    except Exception as e:
        print(f"Error opening file: {e}")
        
    return M
                
# Example usage:
#directory_path = '/Users/gabinleroy/Desktop/spinglass_python/n100t1100'
#file_name = 'snap.000.000.000011220.txt'
#snapshot(file_name,directory_path,'x',2,99,99,100)
#print(snapshot(file_name,directory_path,'x',2,99,99,100).shape)


def cube(filename,directory,cut,Lx,Ly,Lz, separator, component):
    
    # Construct the full file path
    full_file_path = directory + '/' + filename
    
    if cut<= min(Lx,Ly,Lz):
        # Open the file
        try:
            #  with open(full_file_path, 'r') as file:
            #      content = file.readlines()
            #      # Initialize M using NumPy
            #      M = np.zeros((cut + 1, cut + 1, cut + 1))
            #      for k in range(0,cut+1):
            #         for j in range(0,cut+1):
            #              for i in range(0,cut+1):
            #                  line = mapping_tota(i,j,k,Lx,Ly,Lz)
            #                  # Extract float values using NumPy array indexing and conversion
            #                  float_values = list(map(float, content[line].strip().split(separator)))
            #                  # Populate M using array indexing
            #                  M[i, j, k] = site_data(float_values, component)

            
            #Initialize M using NumPy
            M = np.zeros((cut + 1, cut + 1, cut + 1))
            
            #convert component to number : x->0, y->1, z->2
            if component == 'x':
                component = 0
            if component == 'y':
                component = 1
            if component =='z':
                component = 2

            #print('cube function ---------------------------------------------------------')
            #print('')
            #print("we load the column "+str(component)+' of file :')
            #print(full_file_path)
            #print('A = ')
            A = np.loadtxt(full_file_path)[:,component]     
            #print(A)
            #print('')
            #print("we loop over all coordinates in the cube [0,cut]^3, where cut = "+str(cut))
            #print("")
            for k in range(0,cut+1):
                #print('')
                #print("----- z = "+str(k))
                for j in range(0,cut+1):
                    #print("y = "+str(j))
                    inf = mapping_tota(0,j,k,Lx,Ly,Lz)
                    #print(" x from "+str(0)+' to '+str(cut))
                    #print("corresponding lines in tota file from : "+str(inf)+ ' to '+str(inf+cut+1))
                    M[:, j, k] = A[inf:inf+cut+1]                   
            

                        
        except FileNotFoundError:
            print(f"File '{filename}' not found in directory '{directory}'")
        except Exception as e:
            print(f"Error opening file: {e}")
       # print('cumaaaaaaaaaaaaaaa')
        #for z in range(0,cut+1):
        #    for y in range(0,cut+1):
        #        for x in range(0,cut+1):
        #            print(M[x,y,z])

        return M
        

    else:
        raise ValueError("Invalid range for cut. Should be in lattice size.")

#------------------------------------------------------------------------------------------------------------
# i, j = np.meshgrid(np.arange(3), np.arange(2),indexing='ij')
# print(i)
# print(j)
# print('i+j') 
# print(i+j)
# print("")
# def f(i, j):
#     return i

# A = np.zeros((2,2))
# A = f(i,j)
# print(A)
# print('ijk')
    
#i,j,k = np.meshgrid(np.arange(4),np.arange(3),np.arange(2),indexing='ij')
# print("i = ")
# print(i)
# print("j=")
# print(j)
# print("k=")
# print(k)

#L = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#def p(i,j,k):
#    return L[(i*2+j+k).astype]
#print(p(i,j,k))
#------------------------------------------------------------------------------------------------------------

def read_tota_filename(dirpath, filename, wantoverlap = False):
    '''
    extract L,T,a,A from a filename by Tota in a given directory
    if directory name is not of the correct tota form or filename not of the correct tota form, then outputs none
    '''
    #read name of the directory from directory path
    dirname = os.path.basename(dirpath)

    if wantoverlap == False:
        #check if dirname and filename has tota format 
        if dirname[5:].isdigit() and dirname[1:4].isdigit() and dirname.startswith("n") and filename.startswith("snap."):
            #read info from dirname
            T = float("0."+dirname[5:])
            L = float(dirname[1:4])
            #read info from filename
            a = filename[7]
            A = filename[11]
            result = [L,T,a,A]
            return result
        else:
            return None
        
    else:
        #check if dirname and filename has tota format 
        if dirname.startswith("n287t") and filename.startswith("overlap_alpha"):
            #read info from dirname
            T = float("0."+dirname[5:])
            L = float(dirname[1:4])
            #read info from filename
            a = '0'
            A2 = filename[-12]
            A1 = filename[-13]
            A = [A1,A2]
            component1 = filename[-2]
            component2 = filename[-1]
            component = [component1,component2]
            result = [L,T,a,A,component]
            return result
        else:
            return None       

def MPS_on_totaslice(dirpath, filename, par):
    '''
    explain what it does 
    parameters : alpha = string in {'x','y','z'}
                 alphai = integer 
                 k = integer
                 eps = float
                 typeplot = integer in {1,2,3}
    '''
    if type(par[5]) == str :
        alpha = par[0]
        alphai = par[1]
        k = par[2][0]
        eps = par[2][1]
        typeplot = int(par[3])
        separator = par[4]
        component = par[5]
        
        #for axis parameters of create_2Dscatter_plot
        if alpha == 'x':
            axis_parameters = {'xaxis': 'z', 'yaxis':'y'}
        if alpha == 'y':
            axis_parameters = {'xaxis': 'z', 'yaxis':'x'}
        if alpha == 'z':
            axis_parameters = {'xaxis': 'y', 'yaxis':'x'}
        
        #if we are in a tota snapshot file
        if read_tota_filename(dirpath,filename) != None:
            #dimensions of cubic lattice
            L = int(read_tota_filename(dirpath,filename)[0])
            Lx = L-1
            Ly = L-1
            Lz = L
            #temperature
            T = read_tota_filename(dirpath,filename)[1]
            #random configuration index
            a = read_tota_filename(dirpath,filename)[2]
            #replica index
            A = read_tota_filename(dirpath,filename)[3]
            #extract snapshot
            y = snapshot(filename,dirpath,alpha,alphai,Lx,Ly,Lz, separator, component)
            R = find_R_from_snapshotsize(y.shape[0],y.shape[1])
            print('shape of slice is ')
            print(y.shape)
            print('R chosen')
            print(R)
            #dictionnary of parameters for plots title
            plot_par = {'alpha':alpha, 'alphai':alphai, 'Lx':Lx, 'Ly':Ly, 'T':T, 'a':a, 'A':A, 'Plotted':'S'+component}
            #save raw data as .txt
            save_2Darray_as_txt(y,dirpath,plot_par,separator)
            #create a 2D colorp lot of the matrix slice y
            create_2Dscatter_plot(y, plot_par, axis_parameters, dirpath)
            #apply mps to associated normalized quantum state
            print("")
            print("")
            print('-----------------------------------------------------------------------')
            print("For snapshot with parameters : "+str(plot_par))
            asquantum = image_to_vector_bits(y, R, True)
            norme = np.linalg.norm(asquantum)
            asquantum = asquantum / norme
            data = MPS(2, (2*R), asquantum, k, eps, plot_par)
            mps_data = data[typeplot]
            mps_data2 = data[typeplot+1]
            #save plot from MPS
            save_plot_MPS(mps_data, dirpath, 2*R, 2, k, eps, typeplot, plot_par,True)
            save_plot_MPS(mps_data2, dirpath, 2*R, 2, k, eps, typeplot+1, plot_par,True)
            #compressed data
            approx_data = approximate_data(2,(2*R),data,k,eps)
            o = norme*from_approximatedata_to_original_snapshot(approx_data,R)
            plot_par =  {'alpha':alpha, 'alphai':alphai, 'Lx':Lx, 'Ly':Ly, 'T':T, 'a':a, 'A':A, 'Plotted':'S'+component, 'eps':eps}
            create_2Dscatter_plot(o, plot_par, axis_parameters, dirpath)
            #compute error and save
            error =  np.linalg.norm(asquantum - approx_data)
            print(error)
            full_file_name = f'error_locald2_rank{2*R}_k{k}_eps{eps}_alpha{alpha}_alphai{alphai}_Lx{Lx}_Ly{Ly}_T0.{T}_a{a}_A{A}_PlottedS{component}.txt'
            directory_error = 'errors/'+full_file_name 
            add_to_txt(np.array([eps, error]),  directory_error, True)
    else:
        alpha = par[0]
        alphai = par[1]
        k = par[2][0]
        eps = par[2][1]
        typeplot = int(par[3])
        separator = par[4]
        

        #for axis parameters of create_2Dscatter_plot
        if alpha == 'x':
            axis_parameters = {'xaxis': 'z', 'yaxis':'y'}
        if alpha == 'y':
            axis_parameters = {'xaxis': 'z', 'yaxis':'x'}
        if alpha == 'z':
            axis_parameters = {'xaxis': 'y', 'yaxis':'x'}
        
        #if we are in a tota snapshot file
        if read_tota_filename(dirpath,filename,True) != None:
            #dimensions of cubic lattice
            L = int(read_tota_filename(dirpath,filename,True)[0])
            Lx = L-1
            Ly = L-1
            Lz = L
            #temperature
            T = read_tota_filename(dirpath,filename,True)[1]
            #random configuration index
            a = read_tota_filename(dirpath,filename,True)[2]
            #replica index, now is [A1,A2]
            A = read_tota_filename(dirpath,filename,True)[3]
            #component
            component = read_tota_filename(dirpath,filename,True)[4]

            #extract snapshot
            filenameoverl = os.path.join(dirpath, filename)
            y = np.loadtxt(filenameoverl) 
            R = find_R_from_snapshotsize(y.shape[0],y.shape[1])
            print('shape of slice is ')
            print(y.shape)
            print('R chosen')
            print(R)
            #dictionnary of parameters for plots title
            plot_par = {'alpha':alpha, 'alphai':alphai, 'Lx':Lx, 'Ly':Ly, 'T':T, 'a':a, 'A':A[0]+'&'+A[1], 'Plotted':'q'+component[0]+component[1]}
            #create a 2D colorp lot of the matrix slice y
            create_2Dscatter_plot(y, plot_par, axis_parameters, dirpath)
            #apply mps to associated normalized quantum state
            print("")
            print("")
            print('-----------------------------------------------------------------------')
            print("For overlappsnap with parameters : "+str(plot_par))
            asquantum = image_to_vector_bits(y, R, True)
            norme = np.linalg.norm(asquantum)
            asquantum = asquantum / norme
            data = MPS(2, (2*R), asquantum, k, eps, plot_par)
            mps_data = data[typeplot]
            mps_data2 = data[typeplot+1]
            #save plot from MPS
            save_plot_MPS(mps_data, dirpath, 2*R, 2, k, eps, typeplot, plot_par,True)
            save_plot_MPS(mps_data2, dirpath, 2*R, 2, k, eps, typeplot+1, plot_par,True)
            #compressed data
            approx_data = approximate_data(2,(2*R),data,k,eps)
            o = norme*from_approximatedata_to_original_snapshot(approx_data,R)
            plot_par =  {'alpha':alpha, 'alphai':alphai, 'Lx':Lx, 'Ly':Ly, 'T':T, 'a':a, 'A':A[0]+'&'+A[1], 'Plotted':'q'+component[0]+component[1], 'eps':eps}
            create_2Dscatter_plot(o, plot_par, axis_parameters, dirpath)
            #compute error and save
            error =  np.linalg.norm(asquantum - approx_data)
            print(error)
            full_file_name = f'errors/error_locald2_rank{2*R}_k{k}_eps{eps}_alpha{alpha}_alphai{alphai}_Lx{Lx}_Ly{Ly}_T0.{T}_a{a}_A{A[0]}{A[1]}_PlottedS{component[0]}{component[1]}.txt'
            add_to_txt(np.array([eps, error]),  full_file_name, True)

   
def MPS_on_totacube(dirpath, filename, par):   
    '''
    explain what it does 
    parameters : alpha = string in {'x','y','z'}
                 alphai = integer 
                 k = integer
                 eps = float
                 typeplot = integer in {1,2,3}
    '''
    cut = par[0]
    k = par[1][0]
    eps = par[1][1]
    typeplot = int(par[2])
    separator = par[3]
    component = par[4]
    
    #if we are in a tota snapshot file
    if read_tota_filename(dirpath,filename) != None:
        
        #dimensions of cubic lattice
        L = int(read_tota_filename(dirpath,filename)[0])
        Lx = L-1
        Ly = L-1
        Lz = L
        #temperature
        T = read_tota_filename(dirpath,filename)[1]
        #random configuration index
        a = read_tota_filename(dirpath,filename)[2]
        #replica index
        A = read_tota_filename(dirpath,filename)[3]
        #extract snapshot
        print('-----------------------------------------------------------------------')
        y = cube(filename,dirpath,cut,Lx,Ly,Lz, separator, component)
        #print(y)
        R = find_R_from_snapshotsize(y.shape[0],y.shape[1],y.shape[2])
        #dictionnary of parameters for plots title
        plot_par = {'cut':cut, 'Lx':Lx, 'Ly':Ly, 'T':T, 'a':a, 'A':A, 'Plotted':'S'+component}
        
        print('')
        print('')
        print("For cube with parameters : "+str(plot_par))
        asquantum = data3D_to_vector_bits(y, R, True)
        asquantum = asquantum / np.linalg.norm(asquantum)
        data = MPS(2, (3*R), asquantum, k, eps, plot_par)
        
        #compute error and save
        #error =  np.linalg.norm(asquantum - approximate_data(2, 3*R, data, k, eps))
        #print(error)
        #full_file_name = f'error_locald2_rank{3*R}_k{k}_cut{cut}_Lx{Lx}_Ly{Ly}_T0.{T}_a{a}_A{A}_PlottedS{component}.txt'
        #directory_error = 'errors/+'+full_file_name 
        #add_to_txt(np.array([eps, error]), directory_error, True)
        
        mps_data = data[typeplot]
        mps_data2 = data[typeplot+1]
        #save plop
        save_plot_MPS(mps_data, dirpath, 3*R, 2, k, eps, typeplot, plot_par,True)
        save_plot_MPS(mps_data2, dirpath, 3*R, 2, k, eps, typeplot+1, plot_par,True)

import os
import numpy as np

def create_overlap(directory,T, alpha, alphai,a,component1, component2,A_value1, A_value2, g_function):
    '''
    directory = where n287txxx files are
    T = 2100
    alpha = string
    alphai = integer
    a = int
    make overlapp snap_component1_A1 with snap_component2_A2 
    '''
    subdir = directory+f'/n287t{T}'
    output_filename = f'overlap_alpha{alpha}_alphai{alphai}_Lx{286}_Ly{286}_T{float("0."+str(T))}_a{a}_A{A_value1}{A_value2}_PlottedS{component1}{component2}'
    
    # Find relevant files in the directory    
    file_pattern1 = f'alpha{alpha}_alphai{alphai}_Lx286_Ly286_T{float("0."+str(T))}_a0_A{A_value1}_PlottedS{component1}'
    file_pattern2 = f'alpha{alpha}_alphai{alphai}_Lx286_Ly286_T{float("0."+str(T))}_a0_A{A_value2}_PlottedS{component2}'

    file_list1 = [file for file in os.listdir(subdir) if file.startswith(file_pattern1)]
    file_list2 = [file for file in os.listdir(subdir) if file.startswith(file_pattern2)]

    # Ensure both files are found
    if not file_list1 or not file_list2:
        raise ValueError(f"Files not found for A={A_value1} and component = {component1} or A={A_value2} and component = {component2}")

    # Read data from the files
    file_path1 = os.path.join(subdir, file_list1[0])
    file_path2 = os.path.join(subdir, file_list2[0])

    data1 = np.loadtxt(file_path1)
    data2 = np.loadtxt(file_path2)

    # Perform the desired function g on the data
    result = g_function(data1, data2)

    # Save the result to a text file in the same directory
    result_file_path = os.path.join(subdir, output_filename)
    np.savetxt(result_file_path, result)

    return result_file_path

# Example of a function g (you can replace this with your own function)
def overlap_function(data1, data2):
    #print("shape of first snap = ")
    #print(data1.shape)
    #print("shape of second snap = ")
    #print(data2.shape)
    #print((data1+data2).shape)
    return data1*data2  # This is just an example, replace with your actual function

#Example usage of create_overlap
# directory_path = '/Users/gabinleroy/Desktop/spinglass_python/workspace_spinglass/test_data2'
# A_value1 = 1
# A_value2 = 2
# a = 0
# component1 = 'y'
# component2 = 'y'
# T = 2100
# alpha = 'x'
# alphai=150

# result_file_path = create_overlap(directory_path, T,alpha,alphai,a,component1,component2, A_value1, A_value2, overlap_function)
#print(f"Resulting array saved to: {result_file_path}")
