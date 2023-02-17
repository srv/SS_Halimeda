import os 
import argparse
import shutil
import glob

"""
Training-inference-eval pipeline for multiple trainings

Param selection commands generation

"""


entrenos=["a","b","c","d","e"]


learning_rates=["0.003","0.009","0.00033","0.00011"]
learning_rates_str=["0003","0009","000033","000011"]

# path=../runs/3/da_lr/1024_3_{lr}_{entreno}_da


training_instruction= "python3 train.py --run_path ../runs/3/da_lr/1024_3_{}_{}_da/ --data_path /data/splits/cross2/a_da/ --shape 1024 --batch 3 --learning {}"

inference_instruction="python3 inference.py --run_path ../runs/3/da_lr/1024_3_{}_{}_da/ --data_path ../data/splits/cross2/a_da/test/img --shape 1024 "

evaluation_instruction = "python3 evaluation.py --run_path ../runs/3/da_lr/1024_3_{}_{}_da/ --mask_path ../data/splits/cross2/a_da/test/mask --shape 1024 "


for lr,lr_str in zip(learning_rates,learning_rates_str):
    for entreno in entrenos:
        T=training_instruction.format(str(lr_str),str(entreno),str(lr)) 
        os.system(T)
        I=inference_instruction.format(str(lr_str),str(entreno)) 
        os.system(I)
        E=evaluation_instruction.format(str(lr_str),str(entreno)) 
        os.system(E)

