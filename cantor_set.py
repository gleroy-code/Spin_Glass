import numpy as np
import matplotlib.pyplot as plt
def discrete_cantor_set(Niter,L):
    '''
    outputs a list of 1 and -1 = discrete cantor set at iteration Niter
    L must be = q**Niter
    '''

    #plt.figure()
    cset = [[0,L]]
    discrete_cset = np.full(3**Niter, -1)
    #print(discrete_cset)
    for i in range(0,Niter):
        nbseg = len(cset)
        #print('')
        #print('----------------------------------------------------------------------------')
        #print("at iteration i = "+str(i+1)+ ', the number of segments is = ' + str(nbseg))
        k = 0
        while k <= (nbseg - 1) * 2:
            

            currenseg = cset[k]
            init = cset[k][0]
            final = cset[k][1]
            length = final - init
            newsegmentss = [[init, init+length/3], 
                            [init+2*length/3, final]]
            #print('cut segment number = '+str(k)+' with coordinates'+str(cset[k]))
            cset[k] = newsegmentss[0]
            #print('forming the segments with follozing coordinates : '+str(newsegmentss[0])+str(newsegmentss[1]))
            cset.insert(k+1,newsegmentss[1])
            k += 2
            #print('')
            #print('cantor set coordinates are : '+str(cset))
    for seg in cset:
        discrete_cset[int(seg[0]):int(seg[1])]*=-1
    
    # Define a function that is 1 when the array element is 1 and 0 when the array element is -1
    binary_function = np.vectorize(lambda x: 1 if x == 1 else 0)
    print(discrete_cset)
    # Apply the function to the array
    result = binary_function(discrete_cset)

    # Plotting
    # plt.stem(result, linefmt='-', markerfmt='o', basefmt=' ')
    # plt.xlabel('Index')
    # plt.ylabel('Function Value')
    # plt.title('Function Plot')
    # plt.show()
    # print(result)
    return result
# Niter=3
# L = 3**Niter
# discrete_cantor_set(Niter,L)
