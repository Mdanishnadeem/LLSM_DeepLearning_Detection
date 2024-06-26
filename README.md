# pyLattice_deepLearning

Patch-Trained 3D U-Nets for Binary Segmentation. 3 Clear Jupyter Notebooks.

![3D U-Net Architecture](images/u-net_architecture.PNG)

pyLattice_deepLearning was created to segment puncta in 3D microscopy data. However, our Jupyter Notebooks will walk you through training a 3D U-Net on data of your choice. Since our microscopy data was highly skewed, we developed code to split images into cube patches (x,y,z), discard patches with low or no signal, and train on the remaining patches. For predictions, our code once again splits the image into cube patches, performs segmentation within each patch, and stitches the patches together for the final output. Therefore, the trained U-Nets can segment images with large arbitrary dimensions (x,y,z), given that these images have the same resolution as the training data.  

## Getting Started

### Prerequisites
* Anaconda (python 3.*)

### Installing on Macs with M chips (for GPU Optimization)

1. Clone this repository
```
$ git clone git@github.com:Mdanishnadeem/LLSM_DeepLearning_Detection.git
```
2. Install [Homebrew](https://brew.sh/)

3. Install [miniforge](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh)
```
$ chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
$ sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
$ source ~/miniforge3/bin/activate
```

4. Setup a new Conda environment 

```
$ cd LLSM_DeepLearning_Detection
$ conda create --prefix ./env python=3.8
$ conda activate ./env
$ conda install -c apple tensorflow-deps
$ python -m pip install tensorflow-macos
$ python -m pip install tensorflow-metal
$ pip install -r requirements.txt 
```

### General Installation

```
$ git clone git@github.com:Mdanishnadeem/LLSM_DeepLearning_Detection.git
$ cd LLSM_DeepLearning_Detection
$ conda create --name LLSM_DL python==3.8 
$ conda activate LLSM_DL
$ pip install -r requirements_gen.txt

```

## Usage

Run our quickstart notebooks! Look for comments in the notebook to guide you as you train the model.

![Raw-Mask-Prediction](images/raw_mask_prediction.PNG)

A: Raw Data B: Ground Truth C: Prediction

### Preprocessing ([quickstart-1GenData.ipynb](https://github.com/Mdanishnadeem/LLSM_DeepLearning_Detection/blob/main/src/quickstart-1GenData.ipynb))

This notebook generates and saves cube patches from the training data you provide.

Prior to running this notebook, create 2 folders under ```pyLattice_deepLearning/src/```
1. ```pyLattice_deepLearning/src/quickstart-data/```
2. ```pyLattice_deepLearning/src/quickstart-gendata/```

Currently our code supports grayscale images. If you're looking to use RGB images, you'll need to edit how the numpy arrays are handled in [generator.py](https://github.com/Mdanishnadeem/LLSM_DeepLearning_Detection/blob/main/src/generator.py), [predict.py](https://github.com/Mdanishnadeem/LLSM_DeepLearning_Detection/blob/main/src/predict.py), and [visualize.py](https://github.com/Mdanishnadeem/LLSM_DeepLearning_Detection/blob/main/src/visualize.py), in addition to the 3 Jupyter Notebooks.

### Training ([quickstart-2Unet3D.ipynb](https://github.com/Mdanishnadeem/LLSM_DeepLearning_Detection/blob/main/src/quickstart-2Unet3D.ipynb))

This notebook trains a 3D U-Net.

### Predicting ([quickstart-3Load_Model.ipynb](https://github.com/Mdanishnadeem/LLSM_DeepLearning_Detection/blob/main/src/quickstart-3Load_Model.ipynb))

This notebook loads a 3D U-Net from the weights and exports the prediction.

## Authors

* **Joh Schoeneberg** - *Post Doc* - [Website](https://www.schoeneberglab.org)
* **Gautham Raghupathi** - *High School Intern* - [LinkedIn](https://www.linkedin.com/in/gurugautham/)
  
## References
If you use our code, please consider citing: 
```
@inproceedings{schöneberg_raghupathi,
  author={Schöneberg, Johannes and Raghupathi, Gautham and Betzig, Eric and Drubin, David},
  title={3D Deep Convolutional Neural Networks in Lattice Light-Sheet Data Puncta Segmentation},
  booktitle={2019 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2019}
  pages={2369–2372}
}
```
[https://doi.org/10.1109/BIBM47256.2019.8983012](https://doi.org/10.1109/BIBM47256.2019.8983012)

## License
[BSD-3-Clause License](https://github.com/JohSchoeneberg/pyLattice_deepLearning/blob/master/LICENSE)
