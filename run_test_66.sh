echo 'Testing...'
cd /home/wangjianbing/RLLL/
python test.py --batch_size 64 --model_name lwf --num_workers 0 --dels logs_ckpts_outputs --log_level info --init_method tcp://127.0.0.1:46622