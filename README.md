# scorf

## Installation

```
git clone https://github.com/yenchenlin/nerf-pytorch.git
cd nerf-pytorch
pip install -r requirements.txt
```

## Running code

Download data for two example datasets: `lego` and `fern`
```
bash download_example_data.sh
```

To train a low-res `lego` NeRF:
```
python run_nerf.py --config configs/lego.txt
```

---

To train a low-res `fern` NeRF:
```
python run_nerf.py --config configs/fern.txt
```
