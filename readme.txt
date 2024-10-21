From Spin glass Tota N. data to snapshots and MPS analysis

Summary : when executing the main.py, this code does the following : 
- goes over all file in the input main directory
- if the file is a snapshot with filename of the form snap.00a.00A.0000MCsteps, and is in a directory with name of the form nxxxtyyyy (see parameters for meaning of a,A,xxx,yyyy)  
    -> extract the snapshot with coordinates (alpha,alphai) (see pdf file for meaning of coordinate of a snapshot)
    -> save snapshot figure in the same directory as the snapshot file
    -> apply MPS algorithm on this snapshot and produces the plot of 
        - Entenglement entropy vs car ('typeplot' = 2) or
        - Bond dimension vs car ('typeplot' = 1)
        and save the figure in the same directory as snapshot file

This is the structure of the folder that contains the python function files, the main, and the data : 
.
└── spinglass_python/
    └── workspace_spinglass/
        ├── general_functions.py
        ├── main.py
        ├── MPS_plot.py
        ├── mps.py
        ├── reading_data.py
        ├── reading_tota_data.py
        ├── readme.txt
        └── tota_n_data/
            └── nxxxtyyyy/
                └── snap.00a.00A.0000MCsteps 

where xxx is the system size (cubic lattice of x,y in {0,...,L-1} and z in {0,...,L}) and yyyy is the temperature and inside nxxxtyyyy/ the snapshots file are present, 
where A = replica index, a = random configuration of bonds.

When executing the main, the user must enter the following parameters :
- alpha : string in {x,y,z}
- alphai : number that lies in the allowed range of alpha (for example if alpha = x), the alphai must be in {0,...,Lx}, must be int
- k,eps : k is maximum bond dimension, and eps is truncation parameter of singular values. k must be int and eps float
- typeplot : type of plot desired after MPS algorithm, must be int in {1,2} 
- component of spin : must be a string in {x,y,z} and decides which components of heisenberg spin for select on each site
- directory of tota N. data :  path_to_spinglassfolder in machine where you execute code + '/spinglass_python/tota_n_data'
