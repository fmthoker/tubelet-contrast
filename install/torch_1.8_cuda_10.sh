
# conda env for  pytorch 1.8.0 and cuda10
conda create -n tubelet_torch180_cuda10 python=3.6
conda activate tubelet_torch180_cuda10
#
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
#
pip install opencv-python==4.6.0.66
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html
python -c "import mmcv; print('MMCV version: ', mmcv.__version__)"
pip install imutils
pip install scipy
#
pip install tensorboard

