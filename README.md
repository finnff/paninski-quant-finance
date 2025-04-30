# Paninski method to price options using entropy derived volatility in the Black-Scholes model


## Pre-requisites

### To download historical spy options data (`./spy_2020_2022.csv`) please run 

```bash
#!/bin/bash
curl -L -o ./spy-daily-eod-options-quotes-2020-2022.zip\
  https://www.kaggle.com/api/v1/datasets/download/kylegraupe/spy-daily-eod-options-quotes-2020-2022
unzip spy-daily-eod-options-quotes-2020-2022.zip
```

### Install packages with anaconda or venv: 

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
