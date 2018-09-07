# TEST SGE SCRIPT
#$ -l tmem=8.0G 
#$ -l h_vmem=8.0G
#$ -l h_rt=20:00:00
#$ -j y
#$ -S /bin/bash
#$ -cwd
#$ -wd /cluster/project1/FFPredRNN/Log_files/num_go_terms_log_file/branch_number/

/home/vpapaste/anaconda3/bin/python3 /cluster/project1/FFPredRNN/MLDNN/MLDNN_CPU/num_go_terms/branch_number/1.py
