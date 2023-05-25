import json
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def get_speed(device: dict) -> Dict[str, List[float]]:
    return {key: device[key]['tcp_speed_results'] for key in device}


def get_device_mean_and_std(speed: Dict[str, List[float]]) -> Tuple[float, float]:
    speed_mus = [np.mean(v) for v in speed.values()]
    return np.mean(speed_mus), np.std(speed_mus)


def find_closest_device(new_mu, speed: Dict[str, List[float]]) -> int:
    speed_mus = {key: np.mean(value) for key, value in speed.items()}
    closest_device = min(speed_mus.items(), key=lambda x: abs(x[1] - new_mu))[0]
    return closest_device


def DPGMM_sampling(device, mu0, K, sigma, n,
                   alpha=1000, seed=42, is_plot=False) -> Tuple[int, float, float, List[dict]]:
    """
    Args:
        device: dict, device information includes 'tcp_speed_results'
        mu0: float, mean of the prior
        alpha: float, concentration parameter
        K: int, number of sampled distinct devices
        sigma: float, divergence of the sampled distinct devices
        n: int, number of clients
        seed: int, random seed
        is_plot: bool, whether to plot the sampled distribution
    Returns:
    """
    generator = np.random.default_rng(seed)
    speed = get_speed(device)

    alpha = alpha * np.ones(K)
    rhos = generator.dirichlet(alpha, size=1).tolist()[0]
    assignment = generator.choice(K, size=n, p=rhos)

    # initial prior
    simga0 = get_device_mean_and_std(speed)[1]

    # assign a device for each client according to the assignment
    groups = {}  # to store the distribution for each group
    for i in set(assignment):
        new_mu = generator.normal(mu0, sigma * simga0)
        new_device = find_closest_device(new_mu, speed)
        new_data = speed[new_device]
        groups[i] = (new_device, new_data)

        speed.pop(new_device)  # do not choose same device for different clusters

    client_data = {}
    client_device = {}
    for i in range(n):
        client_assignment = assignment[i]
        client_device[i] = groups[client_assignment][0]
        client_data[i] = groups[client_assignment][1]

    sampled_data = [device[idx] for idx in client_device.values()]
    sampled_speed_mean, sampled_speed_std = get_device_mean_and_std(client_data)

    if is_plot:
        device_client = defaultdict(list)
        for key, value in client_device.items():
            device_client[value].append(key)

        plt.rcParams['font.size'] = 18
        plt.rcParams['font.weight'] = 600
        plt.rcParams["axes.labelweight"] = 600
        fig = plt.figure(figsize=(8, 4), dpi=600)
        ax = fig.add_subplot(111)
        ax.grid(True, linestyle='--', linewidth=0.2, color='gray')
        ### kdeplot
        for key, value in device_client.items():
            device = key
            client_list = value
            data = client_data[client_list[0]]
            sns.kdeplot(data, linewidth=1.8, shade=True,
                        label='Client ' + (' '.join([str(item + 1) + ',' for i, item in enumerate(client_list)])).strip(
                            ','))
        plt.ylabel('')
        plt.xlabel('Speed (kb/s)')
        plt.ylabel('Density')
        plt.xlim([0, 30000])
        plt.title(f'$K_n$ = {K}, $\sigma$ = {sigma}')
        plt.legend()
        plt.show()
    return len(groups), sampled_speed_mean, sampled_speed_std, sampled_data


def DPCSM_sampling(score_dict, n, alpha, start_rank):
    """

    :param score_dict: states' score used for sampling
    :param n: sample size
    :param alpha: concentration parameter
    :param start_rank: Start rank of the states
    :return:
    """
    generator = np.random.default_rng(2022)
    sorted_score_dict = dict(sorted(score_dict.items(), key=lambda x: x[1], reverse=True))
    sorted_state_keys = list(sorted_score_dict.keys())

    rhos = [0]
    # Instantiate the first topic
    assignment = [0]  # first point must be assigned to first cluster
    rho_1 = generator.beta(1, alpha)
    remainder = 1 - rho_1
    rhos = [remainder, rho_1]
    new_or_existk = [-1, 0]
    ntopics = 1
    for i in range(1, n):
        k = generator.choice(new_or_existk, p=rhos)
        if k == -1:
            # generate a new topic
            new_rho = generator.beta(1, alpha) * remainder
            remainder -= new_rho
            rhos[0] = remainder
            rhos.append(new_rho)

            ntopics += 1
            assignment.append(ntopics - 1)
            new_or_existk.append(ntopics - 1)
        else:
            assignment.append(k)
    sampled_state_keys = {}
    for i in range(n):
        sampled_rank = assignment[i]
        if sampled_rank >= 0 and sampled_rank + start_rank < len(sorted_state_keys):
            sampled_rank += start_rank
        elif sampled_rank >= 0:  # last one
            sampled_rank = len(sorted_state_keys) - 1
        else:
            raise ValueError("sampled_rank < 0")
        sampled_state_keys[i] = sorted_state_keys[sampled_rank]
    return list(sampled_state_keys.values())
