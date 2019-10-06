# Battery Remaining Time Prediction Model for EVs

* Creadted by Sanghyeon Lee, Taeo Kim, Duckki Lee, and Sung-Wook Park at [Gangneung-Wonju National University].
* This project is collaborated with [SJ tech](http://sjseal.com/) and [Korea Automotive Technology Institute (KATECH)](http://www.katech.re.kr/english/index.asp)

# Introduction
The model generates a warning signal 3 minutes or 30 minutes before the battery runs out.\
Training datasets for 3 minutes prediction was cited from NASA Prognostic Center, Randomized Battery Usage Data Set.\
Also, the training datasets for 30 minutes prediction was generated from KATECH.

## Evaluation Setup
For training and performance evaluation of the warning model, Python 3.6.5 and Tensorflow-gpu 1.7.1 on GeForce GTX 1080ti (GPU) were used.

## Usage
Input dataset should be formed time series dataset which have [Voltage, Current, Temperature] variables.\
The program will modify its variables and return warning signal in every [seqence length] sec.

training.py is the newly simplified version.

## Citation

New version is submitted on "IEIE Transactions on Smart Processing & Computing (IEIE SPC)"

```
@inproceedings{lee2019bms,
  title={Deep Learning based Battery Remaining Time Warning Algorithm},
  author={Sanghyeon Lee and Taeo Kim and Duckki Lee, Sung-Wook Park},
  booktitle={Summer Annual Conference of IEIE, 2019},
  pages={1,016 - 1,017},
  year={2019},
  organization={IEIE}
}
```
