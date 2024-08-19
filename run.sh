# pip install tensorboard tensorboardX
# python main.py --ncpu 50 --device 0,1,2,3,4,5 --nconf 80 --lr 0.00001 --data_path train --epochs 600
# tensorboard --logdir=logs/train
# firefox http://localhost:6006
pip install -r requirements.txt
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
conda install openbabel -c conda-forge