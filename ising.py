from PIL import Image
import os
import numpy as np
from mps import MPS,image_to_vector_bits
import matplotlib.pyplot as plt
import statistics as stat

#list of temperatures for which we have snapshots of Ising 2d
# Ts = [1.9, 1.91, 1.92, 1.93, 1.94, 1.95, 1.96, 1.97, 1.98, 1.99, 
#  2.0, 2.01, 2.02, 2.03, 2.04, 2.05, 2.06, 2.07, 2.08, 2.09, 
#  2.1, 2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 2.18, 2.19, 
#  2.2, 2.205, 2.21, 2.22, 2.225, 2.23, 2.235, 2.24, 2.245, 
#  2.25, 2.255, 2.257, 2.257, 2.26, 2.265, 2.267, 2.27, 2.273, 
#  2.275, 2.276, 2.278, 2.28, 2.281, 2.282, 2.283, 2.285, 2.287, 
#  2.288, 2.29, 2.295, 2.297, 2.3, 2.305, 2.310 ,2.32, 2.33, 2.34, 
#  2.35, 2.36, 2.37, 2.38, 2.39, 2.4, 2.41, 2.42, 2.43, 2.44, 2.45, 
#  2.46, 2.47, 2.48, 2.49, 2.5, 2.51, 2.52, 2.53, 2.54, 2.55, 2.56, 
#  2.57, 2.58, 2.59, 2.6, 2.61, 2.62, 2.63, 2.64, 2.65, 2.66, 2.67, 2.68, 2.69, 2.7, 100, 1.000]
#Ts = [1.000]

def load_ising_data(Ts):
    # Initialize an empty list to store snapshots
    ising_arrays = []
    
    
    compteur = 0 
    
    for T in Ts:
        
        #read the 3 decimal digits of the names
        Tstring = "{:.3f}".format(T)
        
        # Set the path to the folder containing PNG images
        folder_path = "wolff/T="+Tstring
        ising_arrays.append([T])
        
        # Get a list of all PNG files in the folder
        png_files = [file for file in os.listdir(folder_path) if file.endswith(".png")]
        
        # Loop through each PNG file and convert to (array, file name) tuple
        for png_file in png_files:
            # Construct the full path to the image file
            image_path = os.path.join(folder_path, png_file)
            
            # Open the image using PIL
            img = Image.open(image_path).convert("L")
            #img.show()
            
            # Convert the image to a NumPy array
            img_array = np.array(img)
            #print(img_array.dtype)
            #print('before')
            #print(img_array)
            #since white is 0 and black 255
            img_array = img_array*(2/255)-1
            #print('after')
            #print(img_array)
            # Append the tuple (array, file name without extension) to the list
            ising_arrays[compteur].append(img_array)
    
        compteur += 1
    return ising_arrays

#ising_a = load_ising_data(Ts)
#print(len(ising_a[0]))
#for el in ising_a:
#    print(el)
    

#L = [0.3295767211725597, 0.5830572220280413, 0.784432279222904, 0.15397197590222064, 0.15875002414536524, 0.1611248782923689, 0.1623168773815102, 0.1629159581905779, 0.16321206217008954, 0.16335760039048478, 0.1634289914138167, 0.16346805132004416, 0.16348316928755394, 0.1634913947486808, 0.16349447048223742, 0.16349544901767246, 0.16349575217886872]
# Define a function to find the plateau value
def find_plateau_value(arr, L,threshold=1e-2):
    for i in range(len(arr) - 1):
        if np.abs(arr[i] - arr[i+1]) < threshold:
            return i
    return L-2

#print(find_plateau_value(L,2*7))  
def totalEE(R,k,epsilon, ising_arrays):

    #dimension and rank of the corresponding MPS
    L = 2 * R
    d = 2

    #xs contains the temperatures and ys the mean manue of total EE for each temperature
    temperatures = []
    totalEEs = []

    #for each temperature
    for temp in ising_arrays:
    
        #temp[0] is the temperature
        temperatures.append(temp[0])
    
        #ytempor contains the values of total EE for each temperature
        ytempor=[]
        #print('for T='+str(temp[0]))
    
        #go over all snapshots 
        for snap in temp[1::]:
    
            #encode snapshots to TTF using our method
            y = image_to_vector_bits(snap, R, True)
    
            #normalize and put in MPS to compute total EE
            y = y / np.linalg.norm(y)
            y = MPS(d, L, y, k, epsilon)[3]
    
            #add total EE to ytempor
            ytempor.append(y)  
            
            #p=Image.fromarray(snap)
            #p.show()
            #print('htot on snapshot = '+str(y))
    
        #add the mean over the samples to ys
        totalEEs.append(stat.mean(ytempor))
        #print('mean of htot = '+str(stat.mean(ytempor)))

    return totalEEs