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

Lei Ouyang will write this later.

### 2.3 data seperation

Finally, we can 


## 3 Training

## 4 Test

---

# Citation

The Seperation of Data

After filtering the dataset, at the path to the filtered dataset, run

~~~bash
python move_data.py
~~~

This would move and reorganize the dataset.

Then at the path to the moved dataset, run

~~~bash
python seperate_data.py
~~~

to seperate the dataset.