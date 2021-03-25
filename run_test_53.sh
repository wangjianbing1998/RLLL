echo 'Testing...'
cd /home/wangjianbing/RLLL/

GPU_IDS=0
INIT_METHOD=tcp://127.0.0.1:45322
BATCH_SIZE=64
LOG_LEVEL=debug
NUM_WORKERS=0
USE_DISTRIBUTED=none



MODEL_NAME=finetune
SUFFIX=" --model_name $MODEL_NAME --use_distributed $USE_DISTRIBUTED --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS  --log_level $LOG_LEVEL --init_method $INIT_METHOD --gpu_ids $GPU_IDS "

DOC1=$MODEL_NAME"_test_mnist-cifar10-cifar100.txt"
rm $DOC1
nohup python test.py --dataset_list mnist cifar10 cifar100 $SUFFIX > $DOC1 2>&1 &

DOC2=$MODEL_NAME"_test_cifar100-cifar10-mnist.txt"
rm $DOC2
# nohup python test.py --dataset_list cifar100 cifar10 mnists $SUFFIX > $DOC2 2>&1 &

tail -f $DOC1
