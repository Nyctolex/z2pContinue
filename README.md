# Z2P: Instant Visualization of Point Clouds - A follow-up research
<img src="https://github.com/Nyctolex/z2pContinue/assets/65441185/972204ff-0f92-4eee-b295-6b1bdf4ce91a" alt="drawing" width="1000rem"/>


Original paper can be found at [[Paper]](https://arxiv.org/abs/2105.14548)<br>
# Getting Started


#### Setup Environment 
- Create a virtual env using conda/python.
- Run `pip install -r requiement.txt` to install all dependencies.
- Notice that the version of Pytorch requires a Unix distribution.
  
# Running 

## Data
The datasets used for the paper can be downloaded from Google Drive. For the follow-up research, we used the 'Simple dataset' used in the original paper. <br>
[Train](https://drive.google.com/file/d/1-cUSVSVOX2pwVoCn1qekYjHnYrZVBeDs/view?usp=sharing), 
[Test](https://drive.google.com/file/d/1YvsHuaGV_2KsgkinJtbER0zojhkcuZpK/view?usp=sharing)

If you wish to use the ***cache option*** at train time please space for around 350GB of disk space. <br>
This will save the 2D point cloud z-buffers to disk and allow for faster training. 

## Pre-Trained models and Results Preview
You can download the weights of our final model and sample outputs by clicking  <a href="https://drive.google.com/file/d/1p3WJp6dH4q1O_loa-VQmoEfLNJPWkSCe/view"> here. </a>. <br>


### Exploring Training Variations:

To gain a deeper understanding of how the model's behavior is affected by different loss functions and training strategies, we invite you to download the <a href="https://drive.google.com/file/d/1DejloVqvhuunAaPSufPQXbi9N6VXZdSO/view?usp=sharing"> following folder. </a>. This folder contains various models trained under different settings, along with their corresponding results. <br>

## Evaluation
To generate images from point clouds using our model please see the 'demo.ipynb' file. There you can download our model's weights as well as a sample from the dataset and see the model in work. For additional examples please download the test or train datasets and update the 'data_path' variable in the notebook to the correct path of your dataset.

## Training
The structure of the model providing the final RGBA images is built on three sub-modules: the Gray-scale module, the Outlining module, and the Coloring module. Each of these modules can be trained independently.<be>
In cases where only the gray-scale result is required, you could go to the 'Training the Gray-scale module' section and ignore the rest. <br>
Before proceeding to any of the following steps, please make sure the datasets are downloaded and extracted to disk.

### Training the Gray-scale module

The script for training the gray-scale module with the default parameters from the follow-up paper can be found in the ``scripts`` folder under the name ``scripts/train_grayscale.sh``. 
Please make sure to provide the script with the path to the dataset and the path for exporting the trained model. For Example:
```
scripts/train_grayscale.sh.sh /path/to/train/dataset/dir /path/to/export/folder /path/to/test/dataset/dir /path/to/checkpoint/folder
```

### Training the outline module
Similarly to the gray-scale module, you should run the script ``scripts/train_outline.sh`` with the same order of parameters as specified in the previous section.

### Training the coloring module
Make sure that previous to this step you had trained both the outlining and gray-scale modules. You should be able to find .pt files in the ``export directory`` and .pkl files in the ``checkpoint directory`` of the previous modules. Alternatively, you can download the pre-trained modules in the following <a href="https://drive.google.com/file/d/1p3WJp6dH4q1O_loa-VQmoEfLNJPWkSCe/view"> link</a>. <br>
Before training the Coloring Module, you should run the ``create_color_ds.sh`` script found in the 'scripts' folder. This would create an additional dataset constructed from the predictions of the Gray-scale and Outlining modules. <br> 
```
create_coloring_ds.sh /path/to/dataset/renders_shade_abs /path/to/export/dir path/to/grayscale.pkl path/to/outlining.pkl
```
After the dataset for the coloring module is created you should run the corresponding training script from the ``scripts`` folder. For example:
```
scripts/train_color.sh /path/to/train/dataset/dir /path/to/export/folder /path/to/test/dataset/dir /path/to/checkpoint/folder
```

## Inference
The notebook ``demo.ipynb`` provides an example of full inference. You can use the pre-trained weights and the sample point found <a href="https://drive.google.com/file/d/1p3WJp6dH4q1O_loa-VQmoEfLNJPWkSCe/view"> here </a>.

<img src="https://github.com/Nyctolex/z2pContinue/assets/65441185/626348fe-c0a5-4d2d-a1da-e470744acada" alt="drawing" width="1000rem"/>
<img src="https://github.com/Nyctolex/z2pContinue/assets/65441185/608d66b4-90c3-44bc-98d6-5334148d09a4" alt="drawing" width="1000rem"/>



# Questions / Issues
If you have questions or issues running this code, please open an issue.
