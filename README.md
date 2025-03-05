# Inconsistent Multivariate Time Series Forecasting
![Python 3.11](https://img.shields.io/badge/python-3.11-green.svg?style=plastic)
![PyTorch 2.1.0](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![CUDA 11.8](https://img.shields.io/badge/cuda-11.8-green.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

This is the origin Pytorch implementation of FPPformer-MD in the following paper: 
[Inconsistent Multivariate Time Series Forecasting] (Manuscript submitted to IEEE Transactions on Knowledge and Data Engineering). The appendix of this paper can be found at `./appendix/FPPformer_MD_Appendix.pdf`. 

## Model Architecture
Traditional statistical time series forecasting models rely on model identification methods to identify the worthiest model variants to investigate; therefore, the model parameters change with the statistical features of rolling windows to reach optimality. Currently, although deep-learning-based methods achieve promising multivariate forecasting performance, their representations of variable correlations are consistent regardless of the observed local time series properties and dynamic cross-variable relations, rendering them prone to overfitting. To bridge this gap, we propose FPPformer-MD, a novel inconsistent time series forecasting transformer. FPPformer-MD leverages multiresolution analysis to transform each univariate series into multiple frequency scales and evaluate the local variable correlations via their variances. Thus, FPPformer-MD receives richer input features, and its inner inconsistent cross-variable attention mechanism enables the adaptive extraction of cross-variable features. To further alleviate the overfitting problem, we apply dynamic mode decomposition to perform cross-variable data augmentation, which reconstructs the sequence outliers with other correlated sequences during the model training process. Extensive experiments conducted on thirteen real-world benchmarks demonstrate the state-of-the-art performance of FPPformer-MD.
<p align="center">
<img src="./img/FPPformer-MD.jpg" height = "300" alt="" align=center />
<br><br>
<b>Figure 1.</b> The architecture of FPPformer-MD with a $N$-stage encoder and a $M$-stage decoder  ($N=M=6$ in experiment). The colored modules are the novel methods proposed in this work.
</p>


## Requirements
- python == 3.11.4
- numpy == 1.24.3
- pandas == 1.5.3
- scipy == 1.11.3
- torch == 2.1.0+cu118
- scikit-learn == 1.3.0
- PyWavelets == 1.4.1
- astropy == 6.1
- h5py == 3.7.0
- geomstat == 2.5.0

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data

ETT, ECL, Traffic and Weather dataset were acquired at: [here](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing). Solar dataset was acquired at: [Solar](https://drive.google.com/drive/folders/1Gv1MXjLo5bLGep4bsqDyaNMI2oQC9GH2?usp=sharing). PeMSD3, PeMSD4, PeMSD7 and PeMSD8 were acquired at: [PeMS](https://github.com/guoshnBJTU/ASTGNN/tree/main/data). PeMS-Bay was acquired at: [PeMS-Bay](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX).

### Data Preparation
After you acquire raw data of all datasets, please separately place them in corresponding folders at `./data`. 

We place ETT in the folder `./ETT-data`, ECL in the folder `./electricity`  and weather in the folder `./weather` of [here](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing) (the folder tree in the link is shown as below) into folder `./data` and rename them from `./ETT-data`,`./electricity`, `./traffic` and `./weather` to `./ETT`, `./ECL`, `./Traffic` and`./weather` respectively. We rename the file of ECL/Traffic from `electricity.csv`/`traffic.csv` to `ECL.csv`/`Traffic.csv` and rename its last variable from `OT`/`OT` to original `MT_321`/`Sensor_861` separately.
```
The folder tree in https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing:
|-autoformer
| |-ETT-data
| | |-ETTh1.csv
| | |-ETTh2.csv
| | |-ETTm1.csv
| | |-ETTm2.csv
| |
| |-electricity
| | |-electricity.csv
| |
| |-traffic
| | |-traffic.csv
| |
| |-weather
| | |-weather.csv
```

To standardize the data format, we convert the data file of [Solar](https://drive.google.com/drive/folders/1Gv1MXjLo5bLGep4bsqDyaNMI2oQC9GH2?usp=sharing) from 'solar_AL.txt' to 'solar_AL.csv'. Then we compress this file and upload it to the folder `./data/Solar`, where you can get the data file by simply unzipping the 'solar_AL.zip' file.

We place the NPZ files ('PEMS03.npz', 'PEMS04.npz', 'PEMS07.npz', 'PEMS08.npz') of PeMSD3, PeMSD4, PeMSD7 and PeMSD8 in the folder `./PEMS03`, `./PEMS03`, `./PEMS03` and `./PEMS03` of [PeMS](https://github.com/guoshnBJTU/ASTGNN/tree/main/data) 
 into the folder `./data/PEMS`. We place the H5 file ('pems-bay.h5') of PeMS-Bay in the [PeMS-Bay](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) into the folder `./data/PEMS`. The folder trees in the mentioned two links are shown as below:

```
The folder tree in https://github.com/guoshnBJTU/ASTGNN/tree/main/data:
|-PEMS03
| |-PEMS03.csv
| |-PEMS03.npz
| |-PEMS03.txt
|-PEMS04
| |-PEMS04.csv
| |-PEMS04.npz
|-PEMS07
| |-PEMS07.csv
| |-PEMS07.npz
|-PEMS08
| |-PEMS08.csv
| |-PEMS08.npz

The folder tree in https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX:
|-metr-la.h5
|-pems-bay.h5
```

After you process all the datasets, you will obtain folder tree:
```
|-data
| |-ECL
| | |-ECL.csv
| |
| |-ETT
| | |-ETTh1.csv
| | |-ETTh2.csv
| | |-ETTm1.csv
| | |-ETTm2.csv
| |
| |-PEMS
| | |-PEMS03.npz
| | |-PEMS04.npz
| | |-PEMS07.npz
| | |-PEMS08.npz
| | |-pems-bay.h5
| |
| |-Solar
| | |-solar_AL.csv
| |
| |-Traffic
| | |-Traffic.csv
| |
| |-weather
| | |-weather.csv

```

## Usage
Commands for training and testing FPPformer-MD of all datasets are in `./scripts/Main.sh`. 

More parameter information please refer to `main.py`.

We provide a complete command for training and testing FPPformer-MD:

```
python -u main.py --data <data> --input_len <input_len> --pred_len <pred_len> --encoder_layer <encoder_layer> --layer_stack <layer_stack> --MODWT_level<MODWT_level> --patch_size<patch_size> --d_model <d_model> --augmentation_len<augmentation_len> --augmentation_ratio<augmentation_ratio>  --learning_rate <learning_rate> --dropout <dropout> --batch_size <batch_size> --train_epochs <train_epochs> --itr <itr> --train --decoder_IN --patience <patience> --decay<decay>
```

Here we provide a more detailed and complete command description for training and testing the model:

| Parameter name |                                          Description of parameter                                          |
|:--------------:|:----------------------------------------------------------------------------------------------------------:|
|      data      |                                              The dataset name                                              |
|   root_path    |                                       The root path of the data file                                       |
|   data_path    |                                             The data file name                                             |
|  checkpoints   |                                       Location of model checkpoints                                        |
|   input_len    |                                           Input sequence length                                            |
|    pred_len    |                                         Prediction sequence length                                         |
|     enc_in     |                                                 Input size                                                 |
|    dec_out     |                                                Output size                                                 |
|    d_model     |                                             Dimension of model                                             |
|  encoder_layer |                                            The number of stages                                            |
|   layer_stack  |                                       The number of layers per stage                                       |
|   patch_size   |                                The initial patch size in patch-wise attention                              |
|  MODWT_level   |                                           The level of MODWT/MRA                                           |
|augmentation_method   |                                           Augmentation method                                           |
|  augmentation_ratio   |                                           Augmentation ratio                                           |
|  augmentation_len   |                                           Augmentation length                                           |
|  decoder_IN   |                                          Whether to perform IN for decoder inputh                                           |
|    dropout     |                                                  Dropout                                                   |
|    num_workers     |                                                  Data loader num workers                                                   |
|      itr       |                                             Experiments times                                              |
|  train_epochs  |                                      Train epochs of the second stage                                      |
|   batch_size   |                         The batch size of training input data                          |
|   decay   |                         Decay rate of learning rate per epoch                         |
|    patience    |                                          Early stopping patience                                           |
| learning_rate  |                                          Optimizer learning rate                                           |


## Results
The experiment parameters of each data set are formated in the `Main.sh` files in the directory `./scripts/`. You can refer to these parameters for experiments, and you can also adjust the parameters to obtain better mse and mae results or draw better prediction figures. We also provide the commands for obtain the results of FPPformer-MD with longer input sequence length (336) in the file `./scripts/Main.sh`. We present the full results of multivariate forecasting results in Figure 2 and Figure 3. Moreover, we compare FPPformer-MD with other outperforming baselines equipped with individual settings in Figure 4.

<p align="center">
<img src="./img/result1.jpg" height = "500" alt="" align=center />
<br><br>
<b>Figure 2.</b> Multivariate forecasting results (Input length = 96). ETTh denotes the average of ETTh1 and ETTh2. ETTm denotes the average of ETTm1 and ETTm2. PeMS denotes the average of {PeMSD3, PeMSD4, PeMSD7, PeMSD8}.
</p>

<p align="center">
<img src="./img/result2.jpg" height = "500" alt="" align=center />
<br><br>
<b>Figure 3.</b> Full multivariate forecasting results (Input length = 96)
</p>

<p align="center">
<img src="./img/result3.jpg" height = "150" alt="" align=center />
<br><br>
<b>Figure 4.</b> Multivariate forecasting results (Individual settings)
</p>


## Contact
If you have any questions, feel free to contact Li Shen through Email (shenli@buaa.edu.cn) or Github issues. Pull requests are highly welcomed!
