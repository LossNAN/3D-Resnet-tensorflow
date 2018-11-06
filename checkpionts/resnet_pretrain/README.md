# Inflate_ResNet2D_3D
Inflate tensorflow 2dresnet imagenet-pretrain model to 3dresnet imagenet-pretrain 
### Get 2d resnet imagenet-pretrain tensorflow models
### Follow
First download imagenet-pre models from [pretrain-model](https://raw.githubusercontent.com/ry/tensorflow-resnet/master/data/tensorflow-resnet-pretrained-20160509.tar.gz.torrent)<br>
Then, unzip models in /resnet_pretrain<br>
Back to root:<br>
```linux
python inflate_variables.py
```
Then, you will get inflated 3x1x1 pre-model refer to [nonlocal](https://github.com/facebookresearch/video-nonlocal-net)<br>
