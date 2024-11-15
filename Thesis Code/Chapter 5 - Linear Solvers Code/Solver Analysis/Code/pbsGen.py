def master():
    # Regime 1 Files
    i = 0
    n = 5000
    p_seq = range(50, 5000, 50)
    for p in p_seq:
        NumPy_Solve_pbs(n,p,i)
        i += 1
        SciPy_Solve_pbs(n,p,i)
        i += 1
        SciPy_CG_High_pbs(n,p,i)
        i += 1
        SciPy_CG_Med_pbs(n,p,i)
        i += 1
        SciPy_CG_Low_pbs(n,p,i)
        i += 1
        rSolve_pbs(n,p,i)
        i += 1
        GLMNet_High_pbs(n,p,i)
        i += 1
        GLMNet_Med_pbs(n,p,i)
        i += 1
        GLMNet_Low_pbs(n,p,i)
        i += 1
        cPCG_CG_High_pbs(n,p,i)
        i += 1
        cPCG_CG_Med_pbs(n,p,i)
        i += 1
        cPCG_CG_Low_pbs(n,p,i)
        i += 1
    
    p = 5000
    n_seq = range(50, 5000, 50)
    for n in n_seq:
        NumPy_Solve_pbs(n,p,i)
        i += 1
        SciPy_Solve_pbs(n,p,i)
        i += 1
        SciPy_CG_High_pbs(n,p,i)
        i += 1
        SciPy_CG_Med_pbs(n,p,i)
        i += 1
        SciPy_CG_Low_pbs(n,p,i)
        i += 1
        rSolve_pbs(n,p,i)
        i += 1
        GLMNet_High_pbs(n,p,i)
        i += 1
        GLMNet_Med_pbs(n,p,i)
        i += 1
        GLMNet_Low_pbs(n,p,i)
        i += 1
        cPCG_CG_High_pbs(n,p,i)
        i += 1
        cPCG_CG_Med_pbs(n,p,i)
        i += 1
        cPCG_CG_Low_pbs(n,p,i)
        i += 1

    n_seq = range(50, 5001, 50)
    for n in n_seq:
        p = n
        NumPy_Solve_pbs(n,p,i)
        i += 1
        SciPy_Solve_pbs(n,p,i)
        i += 1
        SciPy_CG_High_pbs(n,p,i)
        i += 1
        SciPy_CG_Med_pbs(n,p,i)
        i += 1
        SciPy_CG_Low_pbs(n,p,i)
        i += 1
        rSolve_pbs(n,p,i)
        i += 1
        GLMNet_High_pbs(n,p,i)
        i += 1
        GLMNet_Med_pbs(n,p,i)
        i += 1
        GLMNet_Low_pbs(n,p,i)
        i += 1
        cPCG_CG_High_pbs(n,p,i)
        i += 1
        cPCG_CG_Med_pbs(n,p,i)
        i += 1
        cPCG_CG_Low_pbs(n,p,i)
        i += 1




def NumPy_Solve_pbs(n, p, i):
    # Define the content of the PBS file as a multi-line string
    file_content = f"""#!/bin/bash

    #PBS -M hua.hu@student.unsw.edu.au
    #PBS -m ae

    source /home/z5313381/environments/combss/bin/activate
    cd $PBS_O_WORKDIR

    python3 NumPy_Solve.py {n} {p}
    """

    # File path where the PBS file will be saved
    file_path = f'job{i}.pbs'
    
    # Writing the content to the file
    with open(file_path, 'w') as file:
        file.write(file_content)

def SciPy_Solve_pbs(n, p, i):
    # Define the content of the PBS file as a multi-line string
    file_content = f"""#!/bin/bash

    #PBS -M hua.hu@student.unsw.edu.au
    #PBS -m ae

    source /home/z5313381/environments/combss/bin/activate
    cd $PBS_O_WORKDIR

    python3 SciPy_Solve.py {n} {p}
    """

    # File path where the PBS file will be saved
    file_path = f'job{i}.pbs'
    
    # Writing the content to the file
    with open(file_path, 'w') as file:
        file.write(file_content)

def SciPy_CG_High_pbs(n, p, i):
    # Define the content of the PBS file as a multi-line string
    file_content = f"""#!/bin/bash

    #PBS -M hua.hu@student.unsw.edu.au
    #PBS -m ae

    source /home/z5313381/environments/combss/bin/activate
    cd $PBS_O_WORKDIR

    python3 SciPy_CG_High.py {n} {p}
    """

    # File path where the PBS file will be saved
    file_path = f'job{i}.pbs'
    
    # Writing the content to the file
    with open(file_path, 'w') as file:
        file.write(file_content)
    
def SciPy_CG_Med_pbs(n, p, i):
    # Define the content of the PBS file as a multi-line string
    file_content = f"""#!/bin/bash

    #PBS -M hua.hu@student.unsw.edu.au
    #PBS -m ae

    source /home/z5313381/environments/combss/bin/activate
    cd $PBS_O_WORKDIR

    python3 SciPy_CG_Med.py {n} {p}
    """

    # File path where the PBS file will be saved
    file_path = f'job{i}.pbs'
    
    # Writing the content to the file
    with open(file_path, 'w') as file:
        file.write(file_content)

def SciPy_CG_Low_pbs(n, p, i):
    # Define the content of the PBS file as a multi-line string
    file_content = f"""#!/bin/bash

    #PBS -M hua.hu@student.unsw.edu.au
    #PBS -m ae

    source /home/z5313381/environments/combss/bin/activate
    cd $PBS_O_WORKDIR

    python3 SciPy_CG_Low.py {n} {p}
    """

    # File path where the PBS file will be saved
    file_path = f'job{i}.pbs'
    
    # Writing the content to the file
    with open(file_path, 'w') as file:
        file.write(file_content)

def rSolve_pbs(n, p, i):
    # Define the content of the PBS file as a multi-line string
    file_content = f"""#!/bin/bash

    #PBS -M hua.hu@student.unsw.edu.au
    #PBS -m ae

    source /home/z5313381/environments/combss/bin/activate
    cd $PBS_O_WORKDIR

    Rscript rSolve.R {n} {p}
    """

    # File path where the PBS file will be saved
    file_path = f'job{i}.pbs'
    
    # Writing the content to the file
    with open(file_path, 'w') as file:
        file.write(file_content)

def GLMNet_High_pbs(n, p, i):
    # Define the content of the PBS file as a multi-line string
    file_content = f"""#!/bin/bash

    #PBS -M hua.hu@student.unsw.edu.au
    #PBS -m ae

    source /home/z5313381/environments/combss/bin/activate
    cd $PBS_O_WORKDIR

    Rscript GLMNet-High.R {n} {p}
    """

    # File path where the PBS file will be saved
    file_path = f'job{i}.pbs'
    
    # Writing the content to the file
    with open(file_path, 'w') as file:
        file.write(file_content)

def GLMNet_Med_pbs(n, p, i):
    # Define the content of the PBS file as a multi-line string
    file_content = f"""#!/bin/bash

    #PBS -M hua.hu@student.unsw.edu.au
    #PBS -m ae

    source /home/z5313381/environments/combss/bin/activate
    cd $PBS_O_WORKDIR

    Rscript GLMNet-Med.R {n} {p}
    """

    # File path where the PBS file will be saved
    file_path = f'job{i}.pbs'
    
    # Writing the content to the file
    with open(file_path, 'w') as file:
        file.write(file_content)

def GLMNet_Low_pbs(n, p, i):
    # Define the content of the PBS file as a multi-line string
    file_content = f"""#!/bin/bash

    #PBS -M hua.hu@student.unsw.edu.au
    #PBS -m ae

    source /home/z5313381/environments/combss/bin/activate
    cd $PBS_O_WORKDIR

    Rscript GLMNet-Low.R {n} {p}
    """

    # File path where the PBS file will be saved
    file_path = f'job{i}.pbs'
    
    # Writing the content to the file
    with open(file_path, 'w') as file:
        file.write(file_content)

def cPCG_CG_High_pbs(n, p, i):
    # Define the content of the PBS file as a multi-line string
    file_content = f"""#!/bin/bash

    #PBS -M hua.hu@student.unsw.edu.au
    #PBS -m ae

    source /home/z5313381/environments/combss/bin/activate
    cd $PBS_O_WORKDIR

    Rscript cPCG_CG-High.R {n} {p}
    """

    # File path where the PBS file will be saved
    file_path = f'job{i}.pbs'
    
    # Writing the content to the file
    with open(file_path, 'w') as file:
        file.write(file_content)

def cPCG_CG_Med_pbs(n, p, i):
    # Define the content of the PBS file as a multi-line string
    file_content = f"""#!/bin/bash

    #PBS -M hua.hu@student.unsw.edu.au
    #PBS -m ae

    source /home/z5313381/environments/combss/bin/activate
    cd $PBS_O_WORKDIR

    Rscript cPCG_CG-Med.R {n} {p}
    """

    # File path where the PBS file will be saved
    file_path = f'job{i}.pbs'
    
    # Writing the content to the file
    with open(file_path, 'w') as file:
        file.write(file_content)

def cPCG_CG_Low_pbs(n, p, i):
    # Define the content of the PBS file as a multi-line string
    file_content = f"""#!/bin/bash

    #PBS -M hua.hu@student.unsw.edu.au
    #PBS -m ae

    source /home/z5313381/environments/combss/bin/activate
    cd $PBS_O_WORKDIR

    Rscript cPCG_CG-Low.R {n} {p}
    """

    # File path where the PBS file will be saved
    file_path = f'job{i}.pbs'
    
    # Writing the content to the file
    with open(file_path, 'w') as file:
        file.write(file_content)

master()