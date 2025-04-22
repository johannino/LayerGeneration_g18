#!/bin/bash
#BSUB -J nextlayer
#BSUB -q hpc
#BSUB -W 30
#BSUB -R "rusage[mem=4096MB]"
#BSUB -n 1         
#BSUB -R "span[hosts=1]"   
#BSUB -o batch_out/sleeper_%J.out
#BSUB -e batch_out/sleeper_%J.err
#BSUB -B   
#BSUB -N  

python NextLayerPred.py
