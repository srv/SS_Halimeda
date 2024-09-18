import os

## 1
# train
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.03/1/ --data_path ../Asparagopsis/5_fold/cross/1/ --shape 1024 --batch 16 --learning 0.03"
os.system(command)
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.03/2/ --data_path ../Asparagopsis/5_fold/cross/2/ --shape 1024 --batch 16 --learning 0.03"
os.system(command)
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.03/3/ --data_path ../Asparagopsis/5_fold/cross/3/ --shape 1024 --batch 16 --learning 0.03"
os.system(command)
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.03/4/ --data_path ../Asparagopsis/5_fold/cross/4/ --shape 1024 --batch 16 --learning 0.03"
os.system(command)
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.03/5/ --data_path ../Asparagopsis/5_fold/cross/5/ --shape 1024 --batch 16 --learning 0.03"
os.system(command)
# inference
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.03/1/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.03/2/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.03/3/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.03/4/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.03/5/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
# evaluation
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.03/1/inference/ --out_path 5_fold/l_r_0.03/1/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.03/2/inference/ --out_path 5_fold/l_r_0.03/2/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.03/3/inference/ --out_path 5_fold/l_r_0.03/3/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.03/4/inference/ --out_path 5_fold/l_r_0.03/4/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.03/5/inference/ --out_path 5_fold/l_r_0.03/5/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)


## 2
# train
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.01/1/ --data_path ../Asparagopsis/5_fold/cross/1/ --shape 1024 --batch 16 --learning 0.01"
os.system(command)
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.01/2/ --data_path ../Asparagopsis/5_fold/cross/2/ --shape 1024 --batch 16 --learning 0.01"
os.system(command)
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.01/3/ --data_path ../Asparagopsis/5_fold/cross/3/ --shape 1024 --batch 16 --learning 0.01"
os.system(command)
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.01/4/ --data_path ../Asparagopsis/5_fold/cross/4/ --shape 1024 --batch 16 --learning 0.01"
os.system(command)
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.01/5/ --data_path ../Asparagopsis/5_fold/cross/5/ --shape 1024 --batch 16 --learning 0.01"
os.system(command)
# inference
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.01/1/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.01/2/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.01/3/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.01/4/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.01/5/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
# evaluation
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.01/1/inference/ --out_path 5_fold/l_r_0.01/1/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.01/2/inference/ --out_path 5_fold/l_r_0.01/2/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.01/3/inference/ --out_path 5_fold/l_r_0.01/3/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.01/4/inference/ --out_path 5_fold/l_r_0.01/4/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.01/5/inference/ --out_path 5_fold/l_r_0.01/5/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)


## 3
# train
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.0033/1/ --data_path ../Asparagopsis/5_fold/cross/1/ --shape 1024 --batch 16 --learning 0.0033"
os.system(command)
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.0033/2/ --data_path ../Asparagopsis/5_fold/cross/2/ --shape 1024 --batch 16 --learning 0.0033"
os.system(command)
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.0033/3/ --data_path ../Asparagopsis/5_fold/cross/3/ --shape 1024 --batch 16 --learning 0.0033"
os.system(command)
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.0033/4/ --data_path ../Asparagopsis/5_fold/cross/4/ --shape 1024 --batch 16 --learning 0.0033"
os.system(command)
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.0033/5/ --data_path ../Asparagopsis/5_fold/cross/5/ --shape 1024 --batch 16 --learning 0.0033"
os.system(command)
# inference
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.0033/1/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.0033/2/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.0033/3/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.0033/4/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.0033/5/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
# evaluation
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.0033/1/inference/ --out_path 5_fold/l_r_0.0033/1/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.0033/2/inference/ --out_path 5_fold/l_r_0.0033/2/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.0033/3/inference/ --out_path 5_fold/l_r_0.0033/3/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.0033/4/inference/ --out_path 5_fold/l_r_0.0033/4/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.0033/5/inference/ --out_path 5_fold/l_r_0.0033/5/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)

## 4
# train
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.0011/1/ --data_path ../Asparagopsis/5_fold/cross/1/ --shape 1024 --batch 16 --learning 0.0011"
os.system(command)
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.0011/2/ --data_path ../Asparagopsis/5_fold/cross/2/ --shape 1024 --batch 16 --learning 0.0011"
os.system(command)
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.0011/3/ --data_path ../Asparagopsis/5_fold/cross/3/ --shape 1024 --batch 16 --learning 0.0011"
os.system(command)
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.0011/4/ --data_path ../Asparagopsis/5_fold/cross/4/ --shape 1024 --batch 16 --learning 0.0011"
os.system(command)
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.0011/5/ --data_path ../Asparagopsis/5_fold/cross/5/ --shape 1024 --batch 16 --learning 0.0011"
os.system(command)
# inference
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.0011/1/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.0011/2/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.0011/3/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.0011/4/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.0011/5/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
# evaluation
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.0011/1/inference/ --out_path 5_fold/l_r_0.0011/1/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.0011/2/inference/ --out_path 5_fold/l_r_0.0011/2/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.0011/3/inference/ --out_path 5_fold/l_r_0.0011/3/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.0011/4/inference/ --out_path 5_fold/l_r_0.0011/4/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.0011/5/inference/ --out_path 5_fold/l_r_0.0011/5/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)

## 5
# train
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.00037/1/ --data_path ../Asparagopsis/5_fold/cross/1/ --shape 1024 --batch 16 --learning 0.00037"
os.system(command)
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.00037/2/ --data_path ../Asparagopsis/5_fold/cross/2/ --shape 1024 --batch 16 --learning 0.00037"
os.system(command)
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.00037/3/ --data_path ../Asparagopsis/5_fold/cross/3/ --shape 1024 --batch 16 --learning 0.00037"
os.system(command)
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.00037/4/ --data_path ../Asparagopsis/5_fold/cross/4/ --shape 1024 --batch 16 --learning 0.00037"
os.system(command)
command = "python3 scripts/train.py --run_path 5_fold/l_r_0.00037/5/ --data_path ../Asparagopsis/5_fold/cross/5/ --shape 1024 --batch 16 --learning 0.00037"
os.system(command)
# inference
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.00037/1/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.00037/2/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.00037/3/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.00037/4/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
command = "python3 scripts/inference.py --run_path 5_fold/l_r_0.00037/5/ --data_path ../Asparagopsis/5_fold/test/images/ --shape 1024 --shape_out 1024"
os.system(command)
# evaluation
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.00037/1/inference/ --out_path 5_fold/l_r_0.00037/1/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.00037/2/inference/ --out_path 5_fold/l_r_0.00037/2/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.00037/3/inference/ --out_path 5_fold/l_r_0.00037/3/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.00037/4/inference/ --out_path 5_fold/l_r_0.00037/4/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
command = "python3 scripts/evaluation.py --run_name evaluation --pred_path 5_fold/l_r_0.00037/5/inference/ --out_path 5_fold/l_r_0.00037/5/ --gt_path ../Asparagopsis/5_fold/test/gt/ --shape 1024"
os.system(command)
