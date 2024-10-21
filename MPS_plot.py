import matplotlib.pyplot as plt
from general_functions import save_list_as_txt

def save_plot_MPS(y, directory, rank, local_dim, k, epsilon, typeplot, additional_parameters={},save_as_txt=False):
    '''plots y as a function of car, where y can be EE, bond dimension, totalEE, depending on typeplot value
       and save the figure in 'directory' 
       
       parameters : - 'directory' must be string
                    - 'rank','local_dim','k','epsilon','a','A','T' are numbers
                    - 'typeplot' must be integer in {1,2,3}
                    - 'additional_parameters' : dictionnary of parameters to include in title : {'parameter':value, ...}
  
    '''
    # Create a figure
    fig, ax = plt.subplots()

    x_values = [i for i in range(0, rank-1)]
    
    #checking type of plot variable
    if type(typeplot)!=int:
        raise TypeError("Invalid type for typeplot. Must be an integer.")

    if typeplot not in {1,2,3}:
        raise ValueError("Invalid value for typeplot. Must be in {1,2,3}")

    #determining the type of plot, based on typeplot value:
    if typeplot == 1:
        ylabel = "Bonddimension"
    if typeplot == 2:
        ylabel = "EE"
    if typeplot == 3:
        return None
    
    #plot the data y as a function of x_values
    plt.scatter(x_values, y, marker='.',color='blue')
    # Connect points with lines
    plt.plot(x_values, y, linestyle='-', color='blue')
    plt.xlabel('car')
    plt.ylabel(ylabel)

    #set title of figure and name of saved file
    main_title = f'MPS param. : local d={local_dim}, rank={rank}, $k$={k:.0e}, $\epsilon$={epsilon:.0e}'
    main_name = ylabel+'_plot_'+f'locald{local_dim}_rank{rank}_k{k}_eps{epsilon}'

    if additional_parameters != {}:
        additional_parameters_title = ', '.join([f'{key}={value}' for key, value in additional_parameters.items()])
        additional_parameters_name = '_'.join([f'{key}{value}' for key, value in additional_parameters.items()])
        full_title = f'{main_title}\nSnapshot param. : {additional_parameters_title}'
        full_name = f'{main_name}_{additional_parameters_name}.pdf'
    elif additional_parameters == {}:
        full_title = f'{main_title}'
        full_name =f'{main_name}.pdf'
    #add title
    plt.title(full_title)
    #save plot
    plt.savefig(directory+'/' + full_name, format='pdf', bbox_inches='tight')

    #if desired, save data as.txt 
    save_list_as_txt(y, directory, {'datafromplotof':ylabel,'locald':local_dim, 'rank':rank, 'k':k, 'eps':epsilon, **additional_parameters}, '\t')
    
        
    print('--------------------------------------------------------------------------')