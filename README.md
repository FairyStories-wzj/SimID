# SimID: WiFi-based Few-Shot Cross-Domain User Recognition with Identity Similarity Learning

This is the code to [paper link], including different few-shot learning networks, different feature encoders, and different similarity computation methods.

---

# Usage

Follow our guidance to run SimID.


## 1 Installation

### 1.1 model

Clone or download our codes, and then install the required packages:

~~~
torch==2.6.0+cu126
numpy==2.1.2
matplotlib==3.9.2
pandas==2.0.3
~~~

The operations used in this project are all basic in nature. Therefore, although the dependencies listed above were used during testing, it should also be possible to use other versions of the corresponding dependencies.

If you encounter issues running the code, it's recommended to create a new environment specifically for SimID instead of sharing it with other projects. 

### 1.2 XRF55 Dataset

Our training and evaluation is based on the XRF55 dataset: https://aiotgroup.github.io/XRF55/

You should download the "**XRF55 dataset part1**" as well as "**XRF55 dataset part2**" from it, and if your operation is correct, you should get a file architecture like this:

~~~
XRF55_1/
├──Scene1/
|   └──Scene1/
|       ├──RFID/
|       ├──WiFi/
|       |   ├── 01_01_01.npy
|       |   ├── 01_01_02.npy
|       |   └── ...
|       └──mmWave/
├──Scene2/
|   └── ...
├──Scene3/
|   └── ...
└──Scene4
    └── ...

XRF55_2/
└──Scene1_part2/
    ├──RFID/
    ├──WiFi/
    |   ├── 12_01_01.npy
    |   ├── 12_01_02.npy
    |   └── ...
    └──mmWave/
~~~

## 2 Preprocessing

Before training, you should first extract Wi-Fi data from the dataset, filter the raw data, and split it into our data splitting settings (CPCS, CA, CP, CACP). We have prepared codes for you to do this.

### 2.1 data moving

First, we have to extract Wi-Fi data from the XRF55 dataset, and merge the two parts of Scene 1.

Open the file `simid-dataset/move_data.py`, and fill in the input addresses and output addresses. 


~~~python
# the path to the input dataset
# part1 scene1
PART1_SCENE1_WIFI = "your-path-to-XRF55_1\\Scene1\\Scene1\\WiFi"
# part2 scene1
PART2_SCENE1_WIFI = "your-path-to-XRF55_2\\Scene1_part2\\WiFi"
# part1 scene2
PART1_SCENE2_WIFI = "your-path-to-XRF55_1\\Scene2\\Scene2\\WiFi"
# part1 scene3
PART1_SCENE3_WIFI = "your-path-to-XRF55_1\\Scene3\\Scene3\\WiFi"
# part1 scene4
PART1_SCENE4_WIFI = "your-path-to-XRF55_1\\Scene4\\Scene4\\WiFi"

# the path to the output of the dataset
SCENE1 = "your-output-path\\xrf55_for_simid\\scene1\\"
SCENE2 = "your-output-path\\xrf55_for_simid\\scene2\\"
SCENE3 = "your-output-path\\xrf55_for_simid\\scene3\\"
SCENE4 = "your-output-path\\xrf55_for_simid\\scene4\\"
~~~

Then, run file `simid-dataset/move_data.py`, and you will get a file architecture like this in your output path:

~~~
xrf55_for_simid/
├──scene1/
|   ├── 01_01_01.npy
|   ├── 01_01_02.npy
|   └── ...
├──scene2/
|   └── ...
├──scene3/
|   └── ...
└──scene4/
    └── ...
~~~

### 2.2 data filtering

After moving the data, we need to apply a Butterworth low-pass filter to remove noise from the raw CSI data. We have provided a script `simid-dataset/filter_data.py` for this purpose.

Open the file `simid-dataset/filter_data.py`, and modify the following parameters at the bottom of the file:

```python
# Please set your own data root path here. Example:
# input_folder = r"your-output-path\\xrf55_for_simid"
input_folder = r"<your_data_root_path>"  # <-- Change this to your own data path.
sub_folders = ["scene1", "scene2", "scene3", "scene4"]  # <-- Change or extend as needed
```

The script will:
1. Apply a Butterworth low-pass filter to each .npy file
2. Process all files in parallel using multi-threading for acceleration
3. Save the filtered results to new folders with the suffix '_filtered'

Run the script:
```bash
python simid-dataset/filter_data.py
```

You will get a file architecture like this:

~~~
xrf55_for_simid/
├──scene1_filtered/
|   ├── 01_01_01.npy
|   ├── 01_01_02.npy
|   └── ...
├──scene2_filtered/
|   └── ...
├──scene3_filtered/
|   └── ...
└──scene4_filtered/
    └── ...
~~~

The script will output processing progress and a summary of successful/failed files at the end.

### 2.3 data seperation

Finally, we can seperate the data into four data splitting settings: CPCS, CA, CP, and CACP.

Now make sure you have the file architecture like this:

~~~
xrf55_for_simid/
├──scene1_filtered/
|   ├── 01_01_01.npy
|   ├── 01_01_02.npy
|   └── ...
├──scene2_filtered/
|   └── ...
├──scene3_filtered/
|   └── ...
└──scene4_filtered/
    └── ...
~~~

Open the file `simid-dataset/separate_data.py`, and fill in the address:

~~~python
# the path to the data
DATA_ROOT = "you-path-to-simid\\xrf55_for_simid"
scene1_filtered_path = os.path.join(DATA_ROOT, "scene1_filtered")
scene2_filtered_path = os.path.join(DATA_ROOT, "scene2_filtered")
scene3_filtered_path = os.path.join(DATA_ROOT, "scene3_filtered")
scene4_filtered_path = os.path.join(DATA_ROOT, "scene4_filtered")
~~~

Run the file `simid-dataset/separate_data.py`, you will get the file architecture like this:

~~~
xrf55_for_simid/
├── CA/
|   ├── test/
|   |   ├── 01_12_01.npy
|   |   ├── 01_12_02.npy
|   |   └── ...
|   └── train/
|       ├── 01_01_01.npy
|       ├── 01_01_02.npy
|       └── ...  
├──CACP/
|   ├── test/
|   |   └── ...  
|   └── train/
|       └── ...  
├──CP/
|   └── ...
└──CPCS/
    └── ...
~~~

Now you can use your data to train or test our models.

## 3 Training

### 3.1 Siamese Networks & Prototypical Network

Open file `Siamese Networks/train.py` or `Prototypical Network/train.py`, and fill in parameters in "Addresses" "Network architecture" "Hyperparameters" "Configs", and "GPU", like what is already in it.

Run the file, and you will get:

1. A training curve under `FIGURE_PATH`
2. Several checkpoints under `CHECKPOINT_PATH`
3. Information about each iteration output in console

### 3.2 Relation Network

Please follow the guide in `Relation Network/README.md`

### 3.3 Pretrained Checkpoints

You can also use our pretrained checkpoints provided in https://drive.google.com/drive/folders/10zbW54rddU2gWswb3kkEfWYFfGhEJbpF?usp=sharing instead of training by yourself.

## 4 Test

Please note: Since the testing is random, you may not be able to reproduce results that are exactly the same as those in the paper,
but you should be able to achieve results that are fairly close.

### 4.1 Siamese Networks & Prototypical Network

Open file `Siamese Networks/test.py` or `Prototypical Network/test.py`, and fill in parameters like what is already in it.

Run the file, and you will get:

1. The accuracy and evaluation time of each checkpoint under `CHECKPOINT_PATH`
2. The best and average accuracy over all the checkpoints
3. The average evaluation time of all the evaluation rounds and its standard deviation

### 4.2 Relation Network

Please follow the guide in `Relation Network/README.md`

---

# Citation

If you find our work helpful for your research, please consider citation:

[cite information]