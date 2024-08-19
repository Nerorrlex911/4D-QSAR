pip install tensorboard tensorboardX
python main.py --ncpu 50 --device 0,1,2,3,4,5 --nconf 80 --lr 0.00001 --data_path train --epochs 600
tensorboard --logdir=logs/train
firefox http://localhost:6006