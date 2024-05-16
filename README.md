# Z2P: Instant Visualization of Point Clouds - A follow-up research

<img src='![gap](https://github.com/Nyctolex/z2pContinue/assets/65441185/c772924e-b8a9-4994-9946-892d6c588898)' align="right" width=325>

Original paper can be found at [[Paper]](https://arxiv.org/abs/2105.14548) [[Demo]](https://huggingface.co/spaces/galmetzer/z2p)<br>
# Getting Started


#### Setup Environment 
- Create a virtual env using conda/python.
- Run `pip install -r requiement.txt` to install all dependencies.
- Notice that the version of Pytorch requires a Unix distribution.
  
# Running 

## Data
The datasets used for the paper can be downloaded from Google Drive. For the follow-up research we used the 'Simple dataset' used in the original paper. <br>
[Train](https://drive.google.com/file/d/1-cUSVSVOX2pwVoCn1qekYjHnYrZVBeDs/view?usp=sharing), 
[Test](https://drive.google.com/file/d/1YvsHuaGV_2KsgkinJtbER0zojhkcuZpK/view?usp=sharing)

If you wish to use the ***cache option*** at train time please space for around 350GB of disk space. <br>
This will save the 2D point cloud z-buffers to disk and allow for faster training. 


##TODO Training
First make sure the datasets are downloaded and extracted to disk.

There are two training scripts ``train_regular.sh`` and ``train_metal_roughness.sh``, 
corresponding to the dataset that should be used for training.<br>
Both scripts can be found in the ``scripts`` folder and require three inputs: trainset dir, testset dir, export dir.

For example:  
```
train_regular.sh /home/gal/datasets/renders_shade_abs /home/gal/datasets/renders_shade_abs_test /home/gal/exports/train_regular
```

```
train_metal_roughness.sh /home/gal/datasets/renders_mr /home/gal/datasets/renders_mr_test /home/gal/exports/train_mr
```

## Inference TODO
Inference with the pre-trained demos is available in an interactive [demo app](https://huggingface.co/spaces/galmetzer/z2p), 
as well as with demo scripts in this repo. <br>
The ``scripts`` folder containes two inference scripts: 
* ``inference_goat.sh``
* ``inference_chair.sh``

The scripts require an ``export dir`` output, for example:  

```
scripts/inference_goat.sh /home/gal/exports/goat_results
```

The scripts use ``inference_pc.py`` which allows for more inference options like:
* ``--show_results`` enables showing the results with matplotlib instead of exporting them
* ``--checkpoint`` enables loading a trained checkpoint instead of the pretrained models pulled from drive
* ``--model_type`` toggle between ``regular`` and ``metal_roughness`` models 
* ``--rx``, ``--ry``, ``--rz`` rotate the pc before projecting
* ``--rgb`` control over the color parameter
* ``--light`` control over the lights parameters
* ``--metal``, ``--roughness`` control over the metal and roughness parameters

and more options accessible through ``python inference_pc.py --help``
 

# Questions / Issues
If you have questions or issues running this code, please open an issue.
