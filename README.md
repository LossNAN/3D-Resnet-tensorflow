# Inflate_ResNet2D_3D
Inflate 2dresnet to 3dresnet and use imagenet2d pretrain for train kinetics by tensorflow  
<img align=center width="600" height="500" src="https://github.com/LossNAN/3D-Resnet-tensorflow/blob/master/others/Inflated-resnet.png" alt="3dresnet-model"/>
### This code also for training your own dataset
### Setup
First follow the instructions for [install I3D-Tensorflow](https://github.com/LossNAN/I3D-Tensorflow)<br>
Then, clone this repository using<br>
```linux
$git clone https://github.com/LossNAN/Inflate_ResNet2D_3D.git
```
## How to use our code?
### 1.Data_process
>>1>download Kinetics dataset by yourself, [dataset](https://www.dropbox.com/s/wcs01mlqdgtq4gn/compress.tar.gz?dl=0)<br>
>>2>extract RGB frames by your self(25fps or 30fps), such as:<br>
* ~PATH/Kinetics/train_256/abseiling/-3B32lodo2M_000059_000069 for rgb frames<br>
>>3>convert images to list for train and test<br>
```linux
cd ./experiments/kinetics-400/data_list/
python gen_train_list.py
python gen_test_list.py
```
* you will get npy_files for your own dataset<br>
* such as: train_data_list.npy<br>
### 2.Train your own dataset(Kinetics as example)
>>1>if you get path errors, please modify by yourself
```linux
cd ./experiments/kinetics-400
python multi_gpu_train.py
```
>>2>argues
* learning_rate: Initial learning rate
* max_steps: Number of steps to run trainer
* batch_size: Batch size
* num_frame_per_clib: Nummber of frames per clib
* crop_size: Crop_size
* classics: The num of class
>>3>models will be stored at ./models, and tensorboard logs will be stored at ./visul_logs
```linux
tensorboard --logdir=~path/experiments/Kinetics-400/visual_logs/
```
### 3.Train-error-curve

#### 1.Paper-curve
<img align=center width="350" height="200" src="https://github.com/LossNAN/3D-Resnet-tensorflow/blob/master/others/paper-curve.png" alt="paper-curve"/>

#### 2.Our-curve
<img align=center width="800" height="300" src="https://github.com/LossNAN/3D-Resnet-tensorflow/blob/master/others/curve2.png" alt="error-curve"/>

### 4.Test your own models
>>1>if you get path errors, please modify by yourself
```linux
cd ./experiments/kinetics-400
python multi_gpu_test.py
```
### 5.Result on my linux
  Architecture | Iters | Pre_train | ACC/top1/top5
  ------------- | -------------  | ------------- | -------------
 I3D_baseline  |  60k  | IMAGENET  |66.3/86.7
