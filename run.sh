# Prepare environments

conda activate
pip uninstall apex
git clone https://www.github.com/nvidia/apex
cd apex
python setup.py install --user
cd ..
rm -rf apex

