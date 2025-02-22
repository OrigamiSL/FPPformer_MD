# Inconsistent Multivariate Time Series Forecasting
![Python 3.11](https://img.shields.io/badge/python-3.11-green.svg?style=plastic)
![PyTorch 2.1.0](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![CUDA 11.8](https://img.shields.io/badge/cuda-11.8-green.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

This is the origin Pytorch implementation of FPPformer-MD in the following paper: 
[Inconsistent Multivariate Time Series Forecasting] (Manuscript submitted to IEEE Transactions on Knowledge and Data Engineering).

## Model Architecture
This work intends to solve two general problems involved in deep MTSF. (1) The existing approaches for addressing variable correlations, including CD and CI approaches, consistently ignore or extract all possible variable correlations, making them either insufficient or inappropriate. (2) The existing data augmentation methods hardly exploit variable correlations to generate more training instances. Moreover, since we mainly combine the proposed inconsistent MTSF approach with the FPPformer, this work also attempts to compensate for the inadequacy of the input features in the embedding layer of the FPPformer. To address these problems, we propose an MVCI method that dynamically identifies local variable correlations and an ICVA module that adaptively extracts the cross-variable features of the correlated variables, thus solving general problem (1). Moreover, the MODWT smooths produced by MVCI can additionally provide interpretable input features to enrich the input of the FPPformer. Then, the unique problem of the FPPformer is solved. We also propose a CVDA method to generate more training instances by conducting DMD on the multivariate input sequences, hence addressing general problem (2). As shown in Fig. 1, five major steps are required to upgrade the FPPformer to our proposed FPPformer-MD model:
<p align="center">
<img src="./img/FPPformer_MD_structure.pdf" height = "700" alt="" align=center />
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

ETT, ECL, Traffic and Weather dataset were acquired at: [here](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing). Solar dataset was acquired at: [Solar](https://drive.google.com/drive/folders/1Gv1MXjLo5bLGep4bsqDyaNMI2oQC9GH2?usp=sharing). The raw data of Air dataset was acquired at: [Air](https://archive.ics.uci.edu/dataset/360/air+quality). The raw data of River dataset was acquired at: [River](https://www.kaggle.com/datasets/samanemami/river-flowrf2). The raw data of BTC dataset was acquired at: [BTC](https://www.kaggle.com/datasets/prasoonkottarathil/btcinusd). The raw data of ETH dataset was acquired at: [ETH](https://www.kaggle.com/datasets/franoisgeorgesjulien/crypto).

### Data Preparation
After you acquire raw data of all datasets, please separately place them in corresponding folders at `./FPPformerV2/data`. 

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

We place Solar in the folder `./financial` of [Solar](https://drive.google.com/drive/folders/1Gv1MXjLo5bLGep4bsqDyaNMI2oQC9GH2?usp=sharing) (the folder tree in the link is shown as below) into the folder `./data` and rename them as `./Solar` respectively. 

```
The folder tree in https://drive.google.com/drive/folders/1Gv1MXjLo5bLGep4bsqDyaNMI2oQC9GH2?usp=sharing:
|-dataset
| |-financial
| | |-solar_AL.txt
```

We place Air/River/BTC/ETH in [Air](https://archive.ics.uci.edu/dataset/360/air+quality) /[River](https://www.kaggle.com/datasets/samanemami/river-flowrf2) /[BTC](https://www.kaggle.com/datasets/prasoonkottarathil/btcinusd) /[ETH](https://www.kaggle.com/datasets/franoisgeorgesjulien/crypto) (the folder tree in the link is shown as below) into the folder `./Air`/`./River`/`./BTC`/`./ETH` respectively. 

```
The folder tree in https://archive.ics.uci.edu/dataset/360/air+quality:
|-air+quality
| |-AirQualityUCI.csv
| |-AirQualityUCI.xlsx

The folder tree in https://www.kaggle.com/datasets/samanemami/river-flowrf2:
|-river-flowrf2
| |-RF2.csv

The folder tree in https://www.kaggle.com/datasets/prasoonkottarathil/btcinusd:
|-btcinusd
| |-BTC-Hourly.csv

The folder tree in https://www.kaggle.com/datasets/franoisgeorgesjulien/crypto:
|-crypto
| |-Binance_ETHUSDT_1h (1).csv
```

Then you can run `./data/preprocess.py` to preprocess the raw data of Air, River, BTC and ETH datasets. We replace the missing values, which are tagged with -200 value, by the average values of normal ones. We remove the variable `NMHC(GT)` in Air dataset in that all data of thisvariable in test subset is missing. We rename the file of BTC/ETH from `RF2.csv`/`BTC-Hourly.csv`/`Binance_ETHUSDT_1h (1).csv` to `River`/`BTC.csv`/`ETH.csv` After you successfully run `./data/preprocess.py`, you will obtain folder tree:
```
|-data
| |-Air
| | |-Air.csv
| |
| |-BTC
| | |-BTC.csv
| |
| |-ECL
| | |-ECL.csv
| |
| |-ETH
| | |-ETH.csv
| |
| |-ETT
| | |-ETTh1.csv
| | |-ETTh2.csv
| | |-ETTm1.csv
| | |-ETTm2.csv
| |
| |-River
| | |-River.csv
| |
| |-Solar
| | |-solar_AL.txt
| |
| |-Traffic
| | |-Traffic.csv
| |
| |-weather
| | |-weather.csv

```

## Baseline
We select eight up-to-date baselines, including three TSFT (ARM, iTransformer, Basisformer), two TSFM(TSMixer, FreTS), one TCN (ModernTCN), one RNN-based forecasting method (WITRAN) and one cutting-edge statistics-based forecasting method (OneShotSTL).  Most of these baselines are relative latecomers to FPPformer and their state-of-the-art performances are competent in challenging or even surpassing it. Their source codes origins are given below:


| Baseline | Source Code |
|:---:|:---:|
| ARM | [https://openreview.net/forum?id=JWpwDdVbaM](https://openreview.net/forum?id=JWpwDdVbaM) |
| iTransformer | [https://github.com/thuml/iTransformer](https://github.com/thuml/iTransformer) |
| Basisformer | [https://github.com/nzl5116190/Basisformer](https://github.com/nzl5116190/Basisformer) |
| TSMixer | [https://github.com/google-research/google-research/tree/master/tsmixer](https://github.com/google-research/google-research/tree/master/tsmixer) |
| FreTS | [https://github.com/aikunyi/frets](https://github.com/aikunyi/frets) |
| ModernTCN |  |
| WITRAN | [https://github.com/Water2sea/WITRAN](https://github.com/Water2sea/WITRAN) |
| OneShotSTL | [https://github.com/xiao-he/oneshotstl](https://github.com/xiao-he/oneshotstl) |



Moreover, the default experiment settings/parameters of aforementioned seven baselines are given below respectively:

<table>
<tr>
<th>Baselines</th>
<th>Settings/Parameters name</th>
<th>Descriptions</th>
<th>Default mechanisms/values</th>
</tr>
<tr>
<th rowspan=8>ARM</th>
<th>d_model</th>
<th>The number of hidden dimensions</th>
<th>64</th>
</tr>
<tr>
<th>n_heads</th>
<th>The number of heads in multi-head attention mechanism</th>
<th>8</th>
</tr>
<tr>
<th>e_layers</th>
<th>The number of encoder layers</th>
<th>2</th>
</tr>
<tr>
<th>d_layers</th>
<th>The number of decoder layers</th>
<th>1</th>
</tr>
<tr>
<th>preprocessing_method</th>
<th>The preprocessing method</th>
<th>AUEL</th>
</tr>
<tr>
<th>conv_size</th>
<th>The kernel size of conv layer</th>
<th>[49, 145, 385]</th>
</tr>
<tr>
<th>conv_padding</th>
<th>Padding</th>
<th>[24, 72, 192]</th>
</tr>
<tr>
<th>ema_alpha</th>
<th>The trainable EMA parameter</th>
<th>0.9</th>
</tr>
<tr>
<th rowspan=4>iTransformer</th>
<th>d_model</th>
<th>The number of hidden dimensions</th>
<th>512</th>
</tr>
<tr>
<th>d_ff</th>
<th>Dimension of fcn</th>
<th>512</th>
</tr>
<tr>
<th>n_heads</th>
<th>The number of heads in multi-head attention mechanism</th>
<th>8</th>
</tr>
<tr>
<th>e_layers</th>
<th>The number of encoder layers</th>
<th>3</th>
</tr>
<tr>
<th rowspan=6>Basisformer</th>
<th>N</th>
<th>The number of learnable basis</th>
<th>10</th>
</tr>
<tr>
<th>block_nums</th>
<th>The number of blocks</th>
<th>2</th>
</tr>
<tr>
<th>bottleneck</th>
<th>reduction of bottleneck</th>
<th>2</th>
</tr>
<tr>
<th>map_bottleneck</th>
<th>reduction of mapping bottleneck</th>
<th>2</th>
</tr>
<tr>
<th>n_heads</th>
<th>The number of heads in multi-head attention mechanism</th>
<th>16</th>
</tr>
<tr>
<th>d_model</th>
<th>The number of hidden dimensions</th>
<th>100</th>
</tr>
<tr>
<th rowspan=2>TSMixer</th>
<th>n_block</th>
<th>The number of block for deep architecture</th>
<th>2</th>
</tr>
<tr>
<th>d_model</th>
<th>The hidden feature dimension</th>
<th>64</th>
</tr>
<tr>
<th rowspan=3>FreTS</th>
<th>embed_size</th>
<th>The number of embedding dimensions</th>
<th>128</th>
</tr>
<tr>
<th>hidden_size</th>
<th>The number of hidden dimensions</th>
<th>256</th>
</tr>
<tr>
<th>channel_independence</th>
<th>Whether channels are dependent</th>
<th>1</th>
</tr>
<tr>
<th rowspan=6>ModernTCN</th>
<th>kernel_size</th>
<th>The number of hidden dimensions</th>
<th>512</th>
</tr>
<tr>
<th>d_ff</th>
<th>Dimension of fcn</th>
<th>2048</th>
</tr>
<tr>
<th>n_heads</th>
<th>The number of heads in multi-head attention mechanism</th>
<th>8</th>
</tr>
<tr>
<th>e_layers</th>
<th>The number of encoder layers</th>
<th>2</th>
</tr>
<tr>
<th>d_layers</th>
<th>The number of decoder layers</th>
<th>1</th>
</tr>
<tr>
<th>modes1</th>
<th>The number of Fourier modes to multiply</th>
<th>32</th>
</tr>
</table>


## Usage
Commands for training and testing FPPformer of all datasets are in `./scripts/Main.sh`.

More parameter information please refer to `main.py`.

We provide a complete command for training and testing FPPformer:

For multivariate forecasting:
```
python -u main.py --data <data> --features <features> --input_len <input_len> --pred_len <pred_len> --encoder_layer <encoder_layer> --patch_size <patch_size> --d_model <d_model> --learning_rate <learning_rate> --dropout <dropout> --batch_size <batch_size> --train_epochs <train_epochs> --patience <patience> --itr <itr> --train
```
For univariate forecasting:
```
python -u main_M4.py --data <data> --freq <freq> --input_len <input_len> --pred_len <pred_len> --encoder_layer <encoder_layer> --patch_size <patch_size> --d_model <d_model> --learning_rate <learning_rate> --dropout <dropout> --batch_size <batch_size> --train_epochs <train_epochs> --patience <patience> --itr <itr> --train
```

Here we provide a more detailed and complete command description for training and testing the model:

| Parameter name |                                          Description of parameter                                          |
|:--------------:|:----------------------------------------------------------------------------------------------------------:|
|      data      |                                              The dataset name                                              |
|   root_path    |                                       The root path of the data file                                       |
|   data_path    |                                             The data file name                                             |
|    features    | The forecasting task. This can be set to `M`,`S` (M : multivariate forecasting, S : univariate forecasting |
|     target     |                                         Target feature in `S` task                                         |
|      freq      |                                   Sampling frequency for M4 sub-datasets                                   |
|  checkpoints   |                                       Location of model checkpoints                                        |
|   input_len    |                                           Input sequence length                                            |
|    pred_len    |                                         Prediction sequence length                                         |
|     enc_in     |                                                 Input size                                                 |
|    dec_out     |                                                Output size                                                 |
|    d_model     |                                             Dimension of model                                             |
| representation |                      Representation dims in the end of the intra-reconstruction phase                      |
|    dropout     |                                                  Dropout                                                   |
| encoder_layer  |                                        The number of encoder layers                                        |
|   patch_size   |                                           The size of each patch                                           |
|      itr       |                                             Experiments times                                              |
|  train_epochs  |                                      Train epochs of the second stage                                      |
|   batch_size   |                         The batch size of training input data in the second stage                          |
|    patience    |                                          Early stopping patience                                           |
| learning_rate  |                                          Optimizer learning rate                                           |


## Results
The experiment parameters of each data set are formated in the `Main.sh` files in the directory `./scripts/`. You can refer to these parameters for experiments, and you can also adjust the parameters to obtain better mse and mae results or draw better prediction figures. We provide the commands for obtain the results of FPPformer with longer input sequence lengths in the file `./scripts/LongInput.sh` and those of FPPformer with different encoder layers  in the file `./scripts/ParaSen.sh`. 

<p align="center">
<img src="./img/Multivariate.png" height = "500" alt="" align=center />
<br><br>
<b>Figure 2.</b> Multivariate forecasting results
</p>

<p align="center">
<img src="./img/Univariate.png" height = "400" alt="" align=center />
<br><br>
<b>Figure 3.</b> Univariate forecasting results
</p>

### Full results
Moreover, we present the full results of multivariate forecasting results with long input sequence lengths in Figure 4, that of ablation study in Figure 5 and that of parameter sensitivity in Figure 6.
<p align="center">
<img src="./img/Long.png" height = "500" alt="" align=center />
<br><br>
<b>Figure 4.</b> Multivariate forecasting results with long input lengths
</p>
<p align="center">
<img src="./img/Ablation.png" height = "400" alt="" align=center />
<br><br>
<b>Figure 5.</b> Ablation results with the prediction length of 720
</p>
<p align="center">
<img src="./img/Parameter.png" height = "500" alt="" align=center />
<br><br>
<b>Figure 6.</b> Results of parameter sensitivity on stage numbers
</p>

## Contact
If you have any questions, feel free to contact Li Shen through Email (shenli@buaa.edu.cn) or Github issues. Pull requests are highly welcomed!
