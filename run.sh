pip uninstall torch torchvision torchaudio
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
python main.py --ncpu 50 --device 0,1,2,3,4,5 --nconf 80 --lr 0.00001 --data_path train --epochs 600
#tensorboard --logdir=logs/train
