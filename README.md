# FL edge case

## Overview

- `metric.py` contains the methods for calculating **MC-Time, SC-Ratio (device heterogeneity)**  and **ST-Client-Ratio, ST-Ratio (state heterogeneity)** 

- `sampling.py` contains **DPGMM** and **DPCSM**.

- `data/` folder contains all the databases used and the sampled heterogeneous datasets

## Usage

### 1. Prepare Dataset

We use different datasets for device heterogeneity and state heterogeneity.
- dataset for device heterogeneity
  - `data/device/mobiperf_tcp_down_2018.json`
  - `data/device/mobiperf_tcp_down_2019.json`
- dataset for state heterogeneity
  - `data/state/cached_timers.json`

### 2. Sampling

#### DPGMM (Device heterogeneity)

```python
# network speed infomations refer to 'data/device/mobiperf_tcp_down_2018.json'
speed_info = [
    {
        "tcp_speed_results": [4121, 4753.5, ...], # network speed list from Mobiperf
        ...
    },
    ...
]
n = 2466 # number of clients
mu = 6000 # expected average speeed
K = 50 # number of clusters
simga = 0. # control of divergence
random_seed = 42

_, sampled_speed_mean, sampled_speed_std, samples = DPGMM_sampling(speed_info, mu0=mu, K=k, sigma=sigma, n=2466, seed=random_seed)
```

#### DPCSM (State heterogeneity)

```python
# state score dict used for sampling by DPCSM
score_dict = {
    '681': 0.1,
    '573': 0.2,
    ...
}
n = 2466 # number of clients
alpha = 100 # control of divergence
start_rank = 0 # control of start rank, the same as $StartRank$ in the paper

# return a list of length n=2466 with elements that are keys in score_dict
samples = DPCSM_sampling(score_dict, n=2466, alpha=alpha, start_rank=start_rank)
```

### 3. Metric

We use SC-Ratio and MC-Time for assessing device heterogeneity. We use ST-Client-Ratio and ST-Client-Ratio to assess state heterogeneity of each client and a given sampled state database, respectively.
- SC-Ratio for assessing device heterogeneity
  - ```sc_ratio``` in ```metric.py```
- MC-Time for assessing device heterogeneity
  - ```mc_time``` in ```metric.py```
- ST-Client-Ratio for assessing state heterogeneity of each client
  - ```st_client_ratio``` in ```metric.py```
- ST-Ratio for assessing state heterogeneity of a given sampled database
  - ```st_ratio``` in ```metric.py```

Please refer to [metric_example.ipynb](metric_example.ipynb) for snippets.
