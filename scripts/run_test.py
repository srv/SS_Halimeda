import os

command = "python3 train.py --run_path ../runs/1/000033/ --data_path ../data/1/ --shape 1024 --batch 3 --learning 0.00033"
os.system(command)
command = "python3 inference.py --run_path ../runs/1/000033/ --data_path ../data/test/img --shape 1024"
os.system(command)
command = "python3 matrix.py --run_name 000033 --path_pred ../runs/1/000033/inference --path_out ../runs/1/000033/ --path_gt ../data/test/mask"
os.system(command)
