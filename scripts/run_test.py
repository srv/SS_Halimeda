import os

command = "python3 train.py --run_path ../runs/2/000033_da/ --data_path ../data/2_da/ --shape 1024 --batch 3 --learning 0.00033"
os.system(command)
command = "python3 inference.py --run_path ../runs/2/000033_da/ --data_path ../data/test/img --shape 1024"
os.system(command)
command = "python3 eval.py --run_name 000033_da --path_pred ../runs/2/000033_da/inference --path_out ../runs/1/000033_da/ --path_gt ../data/test/mask"
os.system(command)
