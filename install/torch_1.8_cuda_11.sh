conda create -n tubelet_torch180_cuda11 python=3.7
conda activate tubelet_torch180_cuda11
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python==4.6.0.66
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

pip install imutils
pip install scipy
pip install einops
pip install tensorboard

