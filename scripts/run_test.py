import os

command = "python3 train.py --run_path ../runs/1/000033/ --data_path ../data/1/ --shape 1024 --batch 3 --learning 0.00033"
os.system(command)
command = "python3 inference.py --run_path ../runs/1/000033/ --data_path ../data/test/img --shape 1024"
os.system(command)
command = "python3 evaluation.py --run_name 000033 --pred_path ../runs/1/000033/inference --out_path ../runs/1/000033/ --gt_path ../data/test/mask --shape 1024"
os.system(command)
