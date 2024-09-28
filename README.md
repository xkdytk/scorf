# scorf

TETrack: Target-aware Token Emphasis for Visual Object Tracking

<!--
![TETrack_Framework](tracking/TETrack_network.png)
-->

## Installation

```
git clone https://github.com/yenchenlin/nerf-pytorch.git
cd nerf-pytorch
pip install -r requirements.txt
```

## Data Preparation
You can download the data to use scenes presented in the paper.

[Nerf Synthetic Dataset](http://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset/)

[Nerf Dataset](http://www.kaggle.com/datasets/sauravmaheshkar/nerf-dataset)

Put the example datasets in ./data. It should look like:
```
├── configs                                                                                                       
│   ├── ...                                                                                     
│                                                                                               
├── data                                                                                                                                                                                                       
│   ├── nerf_llff_data                                                                                                  
│   │   └── fern                                                                                                                             
│   │   └── flower  # downloaded llff dataset                                                                                  
│   │   └── horns   # downloaded llff dataset
|   |   └── ...
|   ├── nerf_synthetic
|   |   └── lego
|   |   └── ship    # downloaded synthetic dataset
|   |   └── ...
```

## Running code

---

To train NeRF on different datasets: 

```
python run_nerf.py --config configs/{DATASET}.txt
```

replace `{DATASET}` with `trex` | `horns` | `flower` | `fortress` | `lego` | etc.

---

