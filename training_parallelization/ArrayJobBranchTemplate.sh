# TEST SGE SCRIPT
#$ -l tmem=7.0G 
#$ -l h_vmem=7.0G
#$ -l h_rt=48:00:00
#$ -j y
#$ -l hostname=!chong*
#$ -S /bin/bash
#$ -t 1-100
#$ -N array
#$ -cwd
#$ -wd /cluster/project1/FFPredRNN/Log_files/num_go_terms_log_file/branch_number/

/home/vpapaste/anaconda3/bin/python3 /cluster/project1/FFPredRNN/MLDNN/MLDNN_CPU/num_go_terms/branch_number/$SGE_TASK_ID.py
