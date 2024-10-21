import matplotlib.pyplot as plt
import numpy as np
import math
import numpy.linalg
from itertools import product
import sys
from PIL import Image
from matplotlib.lines import Line2D
from general_functions import add_to_txt

def neumann(x):
   # Ensure x is positive for the logarith
    #print('neumann on' + str(x))
    if x<10e-15:
        return 0
    else:
        result = -x**2 * math.log2(x**2)
        #print(result)
        return result

def MPS(d,L,c,k,epsilon, start_trun, additional_parameters={}, spectrum=False):
    '''
    input : - k=integer>2, number of singular values to keep at each SVD
            - number epsilon = other truncation parameter
            - integer L>2 = number of sites
            - integer d >1 = local dimension
            - array c = raw data vector
            - start_trun = car at which we start to truncate
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
    if start_trun == 1:
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
    elif start_trun >1:
        S = np.diag(S)
        rank = S.shape[0] -1

    


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
        if i + 1 >= start_trun:
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
        elif i+1 < start_trun:
            S = np.diag(S)
            rank = S.shape[0]-1


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