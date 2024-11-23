## The official implementation of VIVID

![Python 3.9.16](https://img.shields.io/badge/python-3.9.16-green.svg?style=plastic)

## Requirements

- python == 3.9.16
- numpy == 1.26.4
- faiss == 1.7.4
- geopy == 2.4.1
- networkx == 3.2.1
- osmnx == 1.9.3
- scipy == 1.13.0
- fastkde == 2.1.3
- pandas == 2.2.2
- PyYAML == 6.0.1

The environment is set up with CUDA 11.3. These dependencies can be installed using the following commands:

```bash
conda env create -f environment.yaml
```
or
```bash
conda create --name VIVID python==3.9.16
conda activate VIVID
pip install -r requirements.txt
```

## Configuration
There are the configuration files in "./config" folder, where one can edit and set test options.


## File Description

### code
- common folder
  - some common utils.
- config folder
  - the configuration file are saved here.
- datasets folder
  - the files for cache generation across datasets.
- topk folder
  - some classes for top-k retrieval of visually similar snapshots.
- main.py
  - the main function file.
- traj_rec_solver.py
  - path inference query algorithm.

## Cache Data Preparation
Before testing, run the .py file in the "./cache_data" to generation cache data. 

## Run the Code
After setting the configuration, to start VIVID, simply run:

```
python main.py \
--cfg ./config/cityflow.yaml \
topk_filtering_coef 4. \
vel_range_coeff 5. \
vel_std_thres 200. \
merge_time_gap 400. \
u_turn_penalty_coeff -0.5 \
edge_weight_thres 0.
```

Note that all parameters can be reconfigured. There are some arguments to be assigned commonly:
Argument	              Description
topk_filtering_coef	    Coefficient for filtering the top-k candidates
vel_range_coeff	        Coefficient defining the acceptable range of velocities
vel_std_thres	          Threshold for the velocity standard deviation
merge_time_gap	        Maximum allowable time gap for merging nodes
u_turn_penalty_coeff	  Coefficient for penalizing large-angle turns
edge_weight_thres	      Threshold for edge weights in the graph

## Dataset
Please download from our [repository](https://terabox.com/s/10qwTOV4juZZ7xB3VeUMF1A). 
