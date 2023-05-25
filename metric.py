import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple


def mc_time(network_speed_list: List[List[float]],
            quantile: float,
            iterations: int,
            total_client_num: int,
            rounds: int,
            selected_num_per_round: int = 16,
            model_size: float = 100) -> np.ndarray:
    """
    Calculate the time cost with Monte Carlo method
    :param network_speed_list: list, each element is a list of speed tested with MobiPerf
    :param quantile: float, the quantile of the speed
    :param iterations: int, number of Monte Carlo iterations
    :param total_client_num: int, number of simulated clients
    :param rounds: int, number of communication rounds
    :param selected_num_per_round: int, number of selected clients per round
    :param model_size: float, model size
    :return:
    """
    generator = np.random.default_rng(42)
    sim_time_list = []
    for _ in tqdm(range(iterations)):

        round_time = []
        # communication rounds
        for r in range(rounds):
            selected_client_idx = generator.choice(total_client_num,
                                                   selected_num_per_round,
                                                   replace=False)
            client_data = [generator.choice(network_speed_list[idx], size=1)[0] for idx in selected_client_idx]

            # wait for client to finish
            client_data_quantile = np.percentile(client_data, quantile)
            round_time.append(model_size / client_data_quantile)
        sim_time_list.append(np.mean(round_time))
    return np.mean(sim_time_list)


def sc_ratio(network_speed_list: List[List[float]],
             ddl: float,
             model_size: float = 85800194 * 4) -> np.ndarray:
    """
    Calculate the SC-Ratio
    :param network_speed_list: list, each element is a list of speed
    :param ddl: float, specified deadline
    :param model_size: float, nerual network size
    :return: 
    """
    sc_ratio_seq = []
    for trace in network_speed_list:
        trace_vector = np.array(trace)
        down, up = np.meshgrid(trace_vector, trace_vector)
        down_t, up_t = model_size / down / 1000, model_size / up / 1000
        total_cost = down_t + up_t
        ratio = np.sum(total_cost < ddl) / total_cost.size
        sc_ratio_seq.append(ratio)
    return np.mean(sc_ratio_seq)


def success_ratio(trace, guid, t_cost=30, ddl=30, is_bit=True):
    """
    Calculate the ST-Client-Ratio
    :param trace: raw data from cached_timers.json includes 'trace_start', 'trace_end' and 'ready_time'
    :param guid: the guid of the trace to be calculated
    :param t_cost: simulation time cost
    :param ddl: specified deadline
    :param is_bit: whether the model size is in bit or not
    :return:
    """
    start = int(trace[guid]["trace_start"])
    end = int(trace[guid]["trace_end"])
    ready_time = trace[guid]["ready_time"]
    if is_bit:
        t_cost = int(t_cost / 4) + 1
        ddl = ddl // 4
    time_interval = np.zeros(end - start)
    exp_time = []
    suc_times = 0
    for s, e in ready_time:
        time_interval[int(s) - start:int(e) - start] += 1
    for idx, (s, e) in enumerate(ready_time):
        s, e = int(s), int(e)
        if e - s > t_cost:
            index_list = list(range(int(e - ddl + 1), e, 1))
            for i in index_list:
                now = i
                act_time = 0
                cost_time = 0
                for _s, _e in (ready_time[idx:] + ready_time):
                    if now > _e:
                        cost_time += end - now
                        now = start
                    if now <= _s:
                        cost_time += _s - now
                        now = _s
                    if _e - now + act_time < t_cost:
                        cost_time += _e - now
                        act_time += _e - now
                        now = _e
                    else:
                        cost_time += t_cost - act_time
                        act_time = t_cost
                        break
                exp_time.append((i, cost_time))
        else:
            for i in range(s, e + 1):
                act_time = 0
                cost_time = 0
                now = i
                for _s, _e in (ready_time[idx:] + ready_time):
                    if now > _e:
                        cost_time += end - now
                        now = start
                    if now <= _s:
                        cost_time += _s - now
                        now = _s
                    if _e - now + act_time < t_cost:
                        cost_time += _e - now
                        act_time += _e - now
                        now = _e
                    else:
                        cost_time += t_cost - act_time
                        act_time = t_cost
                        break
                exp_time.append((i, cost_time))
    for exp_t in exp_time:
        if exp_t[1] < ddl:
            suc_times += 1

    return suc_times / (end - start)


def st_client_ratio(trace, guid, ddl):
    """
    Calculate the ST-Client-Ratio
    :param trace: raw data from cached_timers.json includes 'trace_start', 'trace_end' and 'ready_time'
    :param guid: the guid of the trace to be calculated
    :param ddl: specified deadline
    :return:
    """

    success_ratio_list = []
    for t_cost in range(1, 1 + ddl, ddl // 10):
        success_ratio_list.append(success_ratio(trace, guid, t_cost, ddl))

    return np.mean(success_ratio_list)


def st_ratio(trace, guids, ddl):
    """
    Calculate the ST-Ratio
    :param trace: raw data from cached_timers.json includes 'trace_start', 'trace_end' and 'ready_time'
    :param ddl: specified deadline
    :return:
    """
    client_ratio_list = []
    for guid in tqdm(guids):
        client_ratio_list.append(st_client_ratio(trace, guid, ddl))
    return np.mean(client_ratio_list)
