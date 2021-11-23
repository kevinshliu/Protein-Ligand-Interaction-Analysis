#!/bin/bash

module load python

mkdir contact-chainA-pockets

nohup python PL-Docking.py /nfs/data/bsd/sl9/uthsc-project/fgf-23/5w21/screening/chainA/run-Liu21/A433/top20-combine-pdb/out-pdb chainA.pdb chainA_11 chainA-pockets.txt /nfs/data/bsd/sl9/uthsc-project/fgf-23/5w21/screening/chainA/run-Liu21/A433/contact-chainA-pockets > contact-chainA-pockets.out &

wait

cp contact-chainA-pockets/all_ligands_res_level_interactions.csv contact-chainA-pockets.csv

python contact-chainA-pockets.py
