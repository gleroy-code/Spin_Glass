import matplotlib.pyplot as plt
import numpy as np
import math
import numpy.linalg
from itertools import product
import sys
from PIL import Image
from matplotlib.lines import Line2D
from general_functions import add_to_txt
#print(sys.executable)

def neumann(x):
   # Ensure x is positive for the logarith
    #print('neumann on' + str(x))
    if x<10e-15:
        return 0
    else:
        result = -x**2 * math.log2(x**2)
        #print(result)
        return result
    
def MPS(d,L,c,k,epsilon, additional_parameters={}, spectrum=False):
    '''
    input : - k=integer>2, number of singular values to keep at each SVD
            - number epsilon = other truncation parameter
            - integer L>2 = number of sites
            - integer d >1 = local dimension
            - array c = raw data vector
            - additional_parameter = dictionnary, for file_name of singular values .txt files
            - spectrum, if true, then distr of sing values ploted, at each site, all on same plot
    output :- Asigma = list of np.array, with the matrices of the MPS decomposition at each site
            - bond_d = np array, bond dimensions
    info : decomment savefig to save spectrum
    '''
#--------------------------------------------------    
    if spectrum == True:
        #Number of subplots per row
        subplots_per_row = 2
        number_rows = 4
        #Create a figure and subplots
        fig, axs = plt.subplots(number_rows, subplots_per_row, sharex='all', sharey='all')
        fig.supxlabel(f'$i$',fontsize='x-large')
        fig.supylabel(f'$s_i$',fontsize='x-large')
#--------------------------------------------------
    print("----> rank/epsilon of MPS form is = "+str(L)+','+str(epsilon))
    
    #in case of wrong inputs: warnings
    if L < 2:
        raise ValueError('The code does not work for L<2')
    
    if type(k)!= int or type(d)!=int:
        raise TypeError("Invalid type for k or d. Must be integer")

    if k < 2:
        raise ValueError("cannot truncate for k<2")

    #reshape it into a matrix of size d.d^{L-1}
    Psi = c.reshape(d,d**(L-1)) 

    #create the outputs 
    Asigma = [] #list that will contain at each site, the matrices of the MPS
    bond_d = np.array([],dtype = int) #bond_d[i]= bond dimension between i and i+1= number of columns of the matrix at site i
    EE = np.array([], dtype=float)
    
    #do the svd on reshaped data Psi
    U, S, Vh = np.linalg.svd(Psi, full_matrices=False)
    #print("S before trunc:")
    #print(S)

#--------------------------------------------------
    if spectrum == True:
        axs[0,0].plot(S, marker='o', color='blue', label='bond 1')
        #axs[0,0].set_ylabel('$s_i$', fontsize='x-large')
        #axs[0,0].set_xlabel('$i$', fontsize='x-large')
        axs[0,0].legend(fontsize='medium')
#--------------------------------------------------
    
    #compute the number of singular values > epsilon, which will be = rank+1
    indices = np.where(S > epsilon) #contains indices of element in S that are > epsilon
    # Check if any indices satisfy the condition
    if indices[0].size > 0:
        rank = np.max(indices)
        #print("Maximum index where L > eps", rank)
    else:
        #print("No elements in L are greater than eps.")
        #meaning that we truncate everything, so return each matrix =0
        bond_d = np.zeros(L-1)
        EE = np.zeros(L-1)
        return [0,bond_d,EE,0]

    #convert S to matrix
    S = np.diag(S)
    
    #if truncation bigger than rank, we don't troncate more than just rank truncation
    if k <= rank:
        rank = k-1
    #print("rank = "+str(rank))
    
    # Truncate the matrices  
    U = U[:, :rank+1]
    S = S[:rank+1,:rank+1]
    Vh = Vh[:rank+1, :]    
    
    #save diagonal element of S after truncation, in a file
    main_name = f'sing_values_locald{d}_rank{L}_eps{epsilon}'
    additional_parameters_name = '_'.join([f'{key}{value}' for key, value in additional_parameters.items()])
    full_file_name = f'{main_name}_{additional_parameters_name}.txt'
    directory = 'data_singular_values/'+full_file_name 
    add_to_txt(np.diagonal(S).tolist(), directory, True)

    #print("S after truncation :")
    #print(np.diagonal(S))
    #print("")
    #print("")
    
    #add EE of the bond to outputs
    EE = np.append(EE, sum(neumann(round(x,12)) for x in np.diag(S)) )
    #add matrices and bond dimension to outputs
    Asigma.append( np.array([U[i] for i in range(0,U.shape[0])]) ) #Asigma[l][i] = ith matrix of site l= ith row of U
    bond_d = np.append(bond_d,rank+1)
    
    #loop over the central tensors, L-2 tensors
    for i in range(1,L-1):
        
        #reshape previous S.Vh into a matrix of size (r_i.d, d^(L-1-i))
        Psi = np.dot(S, Vh)
        Psi = Psi.reshape(S.shape[0]*d, d**(L-1-i), order='F')

        #do SVD on this new Psi
        U, S, Vh = np.linalg.svd(Psi, full_matrices=False)
        #print("S before trunc:")
        #print(S)
    
#--------------------------------------------------        
        if spectrum == True and i<=7:
            I = i-4*((i)//4)
            J = (i)//4
            axs[I,J].plot(S, marker='o', color='blue', label=f'bond {i+1}')
            #axs[I,J].set_ylabel('$s_i$', fontsize='x-large')
            #axs[I,J].set_xlabel('$i$', fontsize='x-large')
            axs[I,J].legend(fontsize='medium')
#--------------------------------------------------
        
        #compute the number of singular values > epsilon, which will be = rank+1
        indices = np.where(S > epsilon)
        # Check if any indices satisfy the condition
        if indices[0].size > 0:
            rank = np.max(indices)
            #print("Maximum index where L > eps", rank)
        else:
            #print("No elements in L are greater than eps.")
            #meaning that we truncate everything, so return each matrix =0
            bond_d = np.zeros(L-1)
            EE = np.zeros(L-1)
            return [0,bond_d,EE,0]
        
        #convert S to matrix
        S = np.diag(S)
        
        #if truncation bigger than rank, we don't troncate more than just rank truncation
        if k <= rank:
            rank = k-1
        #print("rank = "+str(rank))
        
        # Truncate the matrices 
        U = U[:, :rank+1]
        S = S[:rank+1,:rank+1]
        Vh = Vh[:rank+1, :]
        #save diagonal element of S after truncation, in a file
        main_name = f'sing_values_locald{d}_rank{L}_eps{epsilon}'
        additional_parameters_name = '_'.join([f'{key}{value}' for key, value in additional_parameters.items()])
        full_file_name = f'{main_name}_{additional_parameters_name}.txt'
        directory = 'data_singular_values/'+full_file_name 
        add_to_txt(np.diagonal(S).tolist(), directory, True)

        #print("S after truncation :")
        #print(np.diagonal(S))
        #print("")
        #print("")

        #add EE of the bond to 
        EE = np.append(EE, sum(neumann(round(x,12)) for x in np.diag(S)))
        #add bond dimension of truncated matrices
        bond_d = np.append(bond_d, rank+1)

        #slice the matrix U into d matrices of size (ri,r_{i+1})
        U=np.split(U,d)

        #add the new set of matrices to Asigma
        Asigma.append(U)
        
    Psi = np.dot(S,Vh)
    #print(bond_d)
    Asigma.append( np.hsplit(np.dot(S,Vh),d) )
 
 #--------------------------------------------------
    if spectrum == True:
        #layout to prevent overlapping titles and labels
        #plt.tight_layout()
        #plt.grid("True")
        #plt.xlim(-0.05,2.05)
        #plt.ylim(-0.05, 1.05)
        #Show the plots
        plt.savefig(f'distr_singvalues_locald{d}_rank{L}_k{k}_eps{epsilon}.pdf',format='pdf')

        # Add red vertical dashed lines on each row at a given value (e.g., x=3)
        for ax in axs[:, 0]:
            ax.axhline(y=epsilon, color='red', linestyle='--')

        for ax in axs[:, 1]:
            ax.axhline(y=epsilon, color='red', linestyle='--')

        # Create a custom legend for the horizontal lines
        legend_line = Line2D([0], [0], color='red', linestyle='--', label=f'$\epsilon$')

        # Add the legend below the entire figure
        fig.legend(handles=[legend_line], loc='lower right',  ncol=1)

        # Adjust layout to prevent clipping of the legend
        plt.tight_layout(rect=(0, 0, 1, 0.99))
        
        #adjust font size of axis tick labels
        for ax in axs.flatten():
            ax.tick_params(axis='both', labelsize='x-large')

        #title of big figure
        fig.suptitle(f'Set of singular values at each bond, $L={L}$, $d={d}$, $\epsilon={epsilon}$, $k={k:.3e}$ ',fontsize='x-large')
        plt.show()
#--------------------------------------------------
    #print('EE is : ')
    #print(EE)

    return Asigma, bond_d, EE, np.sum(EE)

def f1(sigmas,L,d):
    ''' input : np array sigma_1,...,sigma_L, L = size of sigma
                d = leg dimension
        output : np array [i,j] with i row index and j column index
    '''
    
    matrix_indices = np.array([0,0])
    matrix_indices[0] = sigmas[0]
    
    for i in range(0,L-1):
        matrix_indices[1] += (d**i)*sigmas[i+1]
    
    return matrix_indices

def f2(matrix_indices, nb_columns, nb_rows):
    '''
    input: - matrix_indices = np.array, row/column indices in a matrix A 
           - nb_columns = int, number of columns of A
           - nb_rows = int,number of rows of A
    output: - the position I in the raw data vector obtained from row_ordering reshaping A
            - error if nb_columns and nb_rows are not compatible with matrix_indices
    '''
    
    if matrix_indices[0]<0 or matrix_indices[0]>nb_rows:
        raise IndexError('Matrix indices cannot be accessed')
    
    if matrix_indices[1]<0 or matrix_indices[1]>nb_columns:
        raise IndexError('Matrix indices cannot be accessed')
    
    return int(matrix_indices[0]*nb_columns+matrix_indices[1])

def bits_to_base10(bits):
    '''
    intput : - bits = list, of integers 0 or 1
    output : - x = corresponding number in base 10
    '''
    x = 0
    for j in range(0,len(bits)):
        x += bits[j]/(2**(j+1))
    return x  

def base3_to_base10(bits):
    '''
    intput : - bits = list, of integers 0 or 1
    output : - x = corresponding number in base 10
    '''
    x = 0
    for j in range(0,len(bits)):
        x += bits[j]*(3**(len(bits)-j-1))
    return x  

def image_to_vector_bits(image_name, R, already_array):
    '''
    input : - image_name = name of a square image
            - already_array = if true, then the image_name is a already an np.array of dim 2
            - R = int, where 2^R\times 2^R is the total number of pixel we put in the raw data vector
    output : - c = 1D np.array 
    '''
    
    #create output
    output = np.full(2**(2*R), 0.0)
    
    if already_array == False:
    
        #open image and convert to array
        IM = Image.open('images/'+image_name)

        #convert into an array
        IMarray = np.array(IM)
        
    elif already_array == True:
        
        IMarray = image_name

    #check d**L<number of pixels in image 
    if 2**(2*R) <= IMarray.shape[0]*IMarray.shape[1]:
       
        #generate a set containing all the posible values of physical indicies = {sigma_1,...,sigma_{2R}}
        i = [j for j in range(2)]
        cartesian_product = list(product(i, repeat=2*R))

        #go over all the physical indicies
        for i in cartesian_product:
            #print("")
            #print("")
            #print("i is :")
            #print(i)

            #extract the physical indicies s1=x1,s3=x2,...
            x = i[::2]
            #print("xs")
            #print(x)
            I = int( (2**R) * bits_to_base10(x) )

            #extract the physical indicies s2=y1,s4=y2,...
            y = i[1::2]
            #print('ys')
            #print(y)
            J = int ( (2**R) * bits_to_base10(y) )

            #print("position in image array is :")
            #print(I,J)
            #print("")

            #compute pixel value
            pixel = IMarray[I,J]
            #print('pixel value is '+str(pixel))
            #put the value in c
            pos = f2( f1(i,2*R,2), 2**(2*R-1), 2 )
            #print('we insert at pos'+str(pos))
            
            output[pos] = pixel
            #print('value after insertion '+str(output[pos]))
     
            
        return output
    
    else:
        print("Number of pixels > 2**(2*R)")
            
        return None,None

#example use of image_to_vector_bits
#M = np.array([[1,2,3,4],
#              [5,6,7,8],
#              [9,10,11,12],
#              [13,14,15,16]])
#image_to_vector_bits(M,2,True)

def find_R_from_snapshotsize(xsize,ysize,zsize=None):
    '''
    given a snapshot of size xsize*ysize, returns R such that (2**R)*(2**R) is the maximal size of snapshot we can 
    extract
    '''
    if zsize ==None:
        a = min(xsize,ysize)
        if a<=1:
            return ('cannot extract a snapshot a 2**2R from this size')
        else:
            R = 0 
            while 2**R<= a:

                R += 1
            return int(R-1)
    else:
        a = min(xsize,ysize,zsize)
        if a<=1:
            return ('cannot extract a snapshot a 2**2R from this size')
        else:
            R = 0 
            while 2**R<= a:

                R += 1
            return int(R-1)
    
def data3D_to_vector_bits(image_name, R, already_array):
    '''
    input : - image_name = name of a square image
            - already_array = if true, then the image_name is a already an np.array of dim 2
            - R = int, where 2^R\times 2^R is the total number of pixel we put in the raw data vector
    output : - c = 1D np.array 
    '''
    
    #dimentionality of space (equal 3 here, but for generalities)
    d_space = 3
    #used base (equal 2 here, but for generalities)
    q = 2
    #create output
    output = np.full(q**(d_space*R), 0.0)
    

    if already_array == False:
    
        #open image and convert to array
        IM = Image.open('images/'+image_name)

        #convert into an array
        IMarray = np.array(IM)
        
    elif already_array == True:
        
        IMarray = image_name
        

    #check R matches the image size
    if q**(d_space*R) <= IMarray.shape[0]*IMarray.shape[1]*IMarray.shape[2]:
       
        #generate a set containing all the posible values of physical indicies = {sigma_1,...,sigma_{2R}}
        i = [j for j in range(q)]
        cartesian_product = list(product(i, repeat=d_space*R))

        #go over all the physical indicies
        for i in cartesian_product:
            #print("")
            #print("")
            #print("i is :")
            #print(i)

            #extract the physical indicies x1,x2,...
            x = i[::d_space]
            #print("xs")
            #print(x)
            I = int( (q**R) * bits_to_base10(x) )

            #extract the physical indicies y1,y2,...
            y = i[1::d_space]
            #print('ys')
            #print(y)
            J = int ( (q**R) * bits_to_base10(y) )

            #extract the physical indicies z1,z2,...
            z = i[2::d_space]
            #print('zs')
            #print(z)
            K = int( (q**R) * bits_to_base10(z) )
            
            #print("position in image array is :")
            #print(I,J,K)
            #print("")

            #compute pixel value
            pixel = IMarray[I,J,K]
            #print('pixel value is '+str(pixel))
            #put the value in c
            pos = f2( f1(i,d_space*R,q), q**(d_space*R-1), q)
            #print('we insert at pos'+str(pos))
            
            output[pos] = pixel
            #print('value after insertion '+str(output[pos]))
     
            
        return output
    
    else:
        raise TypeError("Number of pixels > q**(d_spqce*R)")
            
        return None,None
    
def data1D_to_vector_base3(image_name, R, already_array):
    '''
    input : - image_name = name of a square image
            - already_array = if true, then the image_name is a already an np.array of dim 2
            - R = int, where 2^R\times 2^R is the total number of pixel we put in the raw data vector
    output : - c = 1D np.array 
    '''
    
    #dimentionality of space (equal 1 here, but for generalities)
    d_space = 1
    #used base (equal 3 here, but for generalities)
    q = 3
    #create output
    output = np.full(q**(d_space*R), 0.0)
    

    if already_array == False:
    
        #open image and convert to array
        IM = Image.open('images/'+image_name)

        #convert into an array
        IMarray = np.array(IM)
        
    elif already_array == True:
        
        IMarray = image_name
        

    #check R matches the image size
    if q**(d_space*R) <= IMarray.shape[0]:
       
        #generate a set containing all the posible values of physical indicies = {sigma_1,...,sigma_{2R}}
        i = [j for j in range(q)]
        cartesian_product = list(product(i, repeat=d_space*R))

        #go over all the physical indicies
        for i in cartesian_product:
            #print("")
            #print("")
            #print("i is :")
            #print(i)

            #extract the physical indicies x1,x2,...
            x = i
            #print("xs")
            #print(x)
            I = int(base3_to_base10(x))

            #print("position in image array is :")
            #print(I)
            #print("")

            #compute pixel value
            pixel = IMarray[I]
            #print('pixel value is '+str(pixel))
            #put the value in c
            pos = f2( f1(i,d_space*R,q), q**(d_space*R-1), q )
            #print('we insert at pos'+str(pos))
            
            output[pos] = pixel
            #print('value after insertion '+str(output[pos]))
     
            
        return output
    
    else:
        raise TypeError("Number of pixels > q**(d_spqce*R)")
            
        return None,None

#example use of data3D_to_vector_bits(image_name, R, already_array):
# M = np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],
#               [[17,18,19,20],[21,22,23,24],[25,26,27,28],[29,30,31,32]],
#               [[33,34,35,36],[37,38,39,40],[41,42,43,44],[45,46,47,48]],
#               [[49,50,51,52],[53,54,55,56],[57,58,59,60],[61,62,63,64]]
#                 ])
# data3D_to_vector_bits(M, 2, True)
    
                      
def approximate_data(d,L,mps,k,epsilon):

    Asigma = mps[0]

    if Asigma != 0:
        c_tilde = np.zeros(d**L)
        
        #print("number of sites L= "+str(L))
        #print("local dimension d= "+str(d))
        #print("raw data = "+str(c))
        #print("")
        #print("")

        #generate a set containing all the posible values of physical indicies = {sigma_1,...,sigma_L}
        i = [j for j in range(d)]
        cartesian_product = (list(product(i, repeat=L)))

        #go over all the physical indicies and compute the product of matrices, print the result
        for i in cartesian_product:

            #print("physical indices: "+str(i))

            #evaluate the product of matrices with these physical indicies

            prod = Asigma[0][i[0]]
            for j in range( 1,len(i) ):

                prod = np.dot( prod, Asigma[j][i[j]] )
                c_tilde[f2( f1(i,L,d), d**(L-1), d )] = prod[0]
                
            #print("product of matrices = "+str(prod))    
            #print( "position in the raw data vector = "+str( f2( f1(i,L), d**(L-1), d ) ) )    
            #print("")
        
        return c_tilde   
    
    else:
        return 0
    
def from_approximatedata_to_original_snapshot(approx_data,R):
    '''
    given a vector = approximate data obtained from applying MPS on image to vector bits
    returns the compressed original snapshot = np.array
    '''
    output = np.zeros((2**R,2**R))
    
    i = [j for j in range(2)]
    cartesian_product = list(product(i, repeat=2*R))

    #go over all the physical indicies
    for i in cartesian_product:
        #print("")
        #print("")
        #print("i is :")
        #print(i)

        #extract the physical indicies s1=x1,s3=x2,...
        x = i[::2]
        #print("xs")
        #print(x)
        I = int( (2**R) * bits_to_base10(x) )

        #extract the physical indicies s2=y1,s4=y2,...
        y = i[1::2]
        #print('ys')
        #print(y)
        J = int ( (2**R) * bits_to_base10(y) )

        #print("position in image array is :")
        #print(I,J)
        #print("")
        pos = f2( f1(i,2*R,2), 2**(2*R-1), 2 )
        data = approx_data[pos]
        output[I,J]=data


        
    return output