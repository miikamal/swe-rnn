# swe-rnn

This GitHub repository contains the source code necessary to reproduce the snow water equivalent forecasting step of the models proposed in paper **Snow Water Equivalent Forecasting in Sub-Arctic and Arctic Regions: Efficient Recurrent Neural Networks Approach** with Python 3.8.10.

## Cloning the repository

The time2vector layer has been given as git submodule, to clone the repo together with t2v submodule clone the project with the `--recursive` option:
```
git clone --recursive https://github.com/miikamal/swe-forecast.git
```

## Usage
NumPy, TensorFlow and Matplotlib is needed for this demo. To install these you can use:
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

To run the forecasting step, execute the Python script `forecast.py`.

If you find this code useful in your research, please cite the article:

**TBA**