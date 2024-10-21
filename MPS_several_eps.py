import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from general_functions import add_to_txt
from mps import neumann
from matplotlib.cm import get_cmap

def MPS_sev_eps(d,L,c,k,epsilons, additional_parameters={}, spectrum=False):
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
    markers = ['o','^']
    #--------- -----------------------------------------    
    if spectrum == True:
        #Number of subplots per row
        subplots_per_row = 2
        number_rows = 4
        #Create a figure and subplots
        fig, axs = plt.subplots(number_rows, subplots_per_row, sharex='all', sharey='all')
        fig.supxlabel(f'$i$',fontsize='xx-large')
        fig.supylabel(f'$s_i$',fontsize='xx-large')
        cmap = plt.cm.get_cmap('tab10')
    #--------------------------------------------------
    for (ep,epsilon) in enumerate(epsilons):

        print("----> rank of MPS form is = "+str(L))
        
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
            axs[0,0].plot(S, marker=markers[ep], color=cmap(ep), label='bond 1')
            #axs[0,0].set_ylabel('$s_i$', fontsize='x-large')
            #axs[0,0].set_xlabel('$i$', fontsize='x-large')
            #axs[0,0].legend(fontsize='medium')
            axs[0,0].set_title(f'Bond {1}',fontsize='xx-large')
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
            if spectrum == True:
                I = i-4*((i)//4)
                J = (i)//4
                axs[I,J].plot(S, marker=markers[ep], color=cmap(ep), label=f'bond {i+1}')
                #axs[I,J].set_ylabel('$s_i$', fontsize='x-large')
                #axs[I,J].set_xlabel('$i$', fontsize='x-large')
                #axs[I,J].legend(fontsize='medium')
                axs[I,J].set_title(f'Bond {i}',fontsize='xx-large')

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
            #plt.savefig(f'distr_singvalues_locald{d}_rank{L}_k{k}_eps{epsilon}.pdf',format='pdf')

            # Add red vertical dashed lines on each row at a given value (e.g., x=3)
            for ax in axs[:, 0]:
                ax.axhline(y=epsilon, color=cmap(ep), linestyle='--')

            for ax in axs[:, 1]:
                ax.axhline(y=epsilon, color=cmap(ep), linestyle='--')

        #Create a custom legend for the horizontal lines
        legend_line = Line2D([0], [0], color=cmap(0), linestyle='--', label=f'$\epsilon=0.02$')
        legend_line2 = Line2D([0], [0], color=cmap(1), linestyle='--', label=f'$\epsilon=0.1$')
        # Add the legend below the entire figure
        fig.legend(handles=[legend_line,legend_line2], loc='lower right',  ncol=2, fontsize='xx-large')
            
        #adjust font size of axis tick labels
        for ax in axs.flatten():
            ax.tick_params(axis='both', labelsize='xx-large')
    
        #title of big 
        fig.suptitle(f'$L={L}$, $d={d}$, $k={k:.3e}$ ',fontsize='xx-large')
            
    plt.show()
#--------------------------------------------------
    return Asigma, bond_d, EE, np.sum(EE)