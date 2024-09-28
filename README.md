# SCoRF

TETrack: Target-aware Token Emphasis for Visual Object Tracking

<!--
![TETrack_Framework](tracking/TETrack_network.png)
-->

## Installation

```
git clone https://github.com/xkdytk/scorf.git
cd scorf
pip install -r requirements.txt
```

## Data Preparation
You can download the data to use scenes presented in the paper.

---

[Nerf Synthetic Dataset](http://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset/)

[Nerf Dataset](http://www.kaggle.com/datasets/sauravmaheshkar/nerf-dataset)

---

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

To train SCoRF on datasets: 

```
python train.py --config configs/{DATASET}.txt
```

To test SCoRF on datasets: 

```
python eval.py --config configs/{DATASET}.txt --ckpt result/{CHECKPOINT} --i_video
```

replace `{DATASET}` with `fern` | `lego` | etc.

replace `{CHECKPOINT}` with file path to use.


