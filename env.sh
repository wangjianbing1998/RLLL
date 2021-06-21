# Prepare environments

conda activate
pip uninstall apex
git clone https://www.github.com/nvidia/apex
cd apex
python3 setup.py install --user
# cd ..
# rm -rf apex
#
#
# run_train()
# {
#     model=$1;
#     dataset_list=$2
#     echo $1,$2
# }
