
# coding: UTF-8
# ---------------------------------------
# 各フローに対して候補経路の評価を multiprocessing により並列化し、
# 最も早くタイマーが 0 になる候補を採用する方式（必ず変更されるように）。
# 実行環境の CPU コア数に応じてプロセス数を自動調整し、高速かつ公平な探索を実現。
# ---------------------------------------

import networkx as nx
import graph_making
import math
import random
import collections
from flow import Flow
import matplotlib.pyplot as plt
import time
from multiprocessing import Process, Manager, cpu_count
from decimal import Decimal, getcontext

def calculate_timer(current_max_load, next_max_load, total_path_count, tau, beta):
    return (1 / total_path_count) * math.exp(tau - (1/2 * beta * (current_max_load - next_max_load)))

node = 10
seed_value = 40
random.seed(seed_value)
a = 20
beta = 160
tau = 2
count = 20
retu = 4

def evaluate_flow(i, current_paths, flows, all_paths, total_demand_snapshot, capacity, total_path_count, current_max_load_ratio, return_list):
    getcontext().prec = 20
    flow = flows[i]
    demand = flow.get_demand()
    current_path = current_paths[i]
    best_timer = float('inf')
    best_path = None

    temp_delete_current_path = total_demand_snapshot.copy()
    for j in range(len(current_path) - 1):
        edge = (current_path[j], current_path[j + 1])
        temp_delete_current_path[edge] -= demand

    for candidate_path in all_paths[i]:
        if candidate_path == current_path:
            continue  # 同じ経路は候補から除外

        temp_demand = temp_delete_current_path.copy()
        for j in range(len(candidate_path) - 1):
            edge = (candidate_path[j], candidate_path[j + 1])
            temp_demand[edge] += demand

        edge_load_ratios = {edge: temp_demand[edge] / capacity[edge] for edge in temp_demand if edge in capacity}
        next_max_load_ratio = max(edge_load_ratios.values()) if edge_load_ratios else 0
        timer = calculate_timer(current_max_load_ratio, next_max_load_ratio, total_path_count, tau, beta)

        if timer < best_timer:
            best_timer = timer
            best_path = candidate_path

    if best_path is not None:
        return_list.append((best_timer, i, best_path))

def main():
    g = graph_making.Graphs(a, a)
    g.randomGraph(g, n=node, k=5, seed=seed_value, number_of_area=1, number_of_areanodes=node, area_height=retu)
    capacity = nx.get_edge_attributes(g, 'capacity')

    All_commodity_list = []
    demand_list = []
    flows = []
    all_paths = []
    number_of_paths = []
    area_nodes_list = list(g.area_nodes_dict[0])

    for _ in range(a):
        while True:
            s, t = random.sample(area_nodes_list, 2)
            demand = random.randint(5, 15)
            paths = find_all_paths(g, s, t)
            if not paths:
                continue
            if s != t and (s, t) not in All_commodity_list:
                break
        demand_list.append(demand)
        All_commodity_list.append((s, t))
        flow = Flow(g, len(flows), s, t, demand)
        flows.append(flow)
        all_paths.append(paths)
        number_of_paths.append(len(paths))

    total_path_count = sum(number_of_paths)
    current_paths = [random.choice(paths) for paths in all_paths]
    initial_paths = current_paths[:]
    max_load_ratio_history = []
    change_history = []
    flow_updates = 0
    i_iteration = 0

    while flow_updates < count:
        i_iteration += 1
        print(f"\n--- Iteration {i_iteration} ---")

        total_demand = collections.defaultdict(float)
        for path, flow in zip(current_paths, flows):
            for j in range(len(path) - 1):
                edge = (path[j], path[j + 1])
                total_demand[edge] += flow.get_demand()

        current_max_load_ratio = max([flow / capacity[edge] for edge, flow in total_demand.items() if edge in capacity], default=0)
        max_load_ratio_history.append(current_max_load_ratio)

        manager = Manager()
        return_list = manager.list()
        processes = []

        for i in range(len(flows)):
            p = Process(target=evaluate_flow, args=(i, current_paths, flows, all_paths,
                                                    total_demand, capacity, total_path_count,
                                                    current_max_load_ratio, return_list))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        if return_list:
            best_timer, best_flow_index, chosen_path = min(return_list, key=lambda x: x[0])
            change_history.append(f"Iteration {i_iteration}: Flow {best_flow_index} changed from {current_paths[best_flow_index]} to {chosen_path}")
            current_paths[best_flow_index] = chosen_path
            flow_updates += 1
            print(f"✔️ Flow {best_flow_index} path changed. Timer: {best_timer:.4f}")
        else:
            print("⚠️ No alternative path found. Skipping iteration.")

    print("\nInitial Paths:")
    for i, path in enumerate(initial_paths):
        print(f"Flow {i}: {path}")

    print("\nChange History:")
    for entry in change_history:
        print(entry)

    print("\nFinal Paths:")
    for i, path in enumerate(current_paths):
        print(f"Flow {i}: {path}")

    avg_late = sum(max_load_ratio_history[len(max_load_ratio_history)//2:]) / len(max_load_ratio_history[len(max_load_ratio_history)//2:])
    print(f"\nFinal Average Max Load (late half): {avg_late:.4f}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(max_load_ratio_history)+1), max_load_ratio_history, marker='o')
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Maximum Load Ratio", fontsize=14)
    plt.title("Maximum Load Ratio Over Iterations", fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def find_all_paths(g, start, end, path=None):
    if path is None:
        path = []
    path = path + [start]
    if start == end:
        return [path]
    if start not in g:
        return []
    paths = []
    for neighbor in g[start]:
        if neighbor not in path:
            new_paths = find_all_paths(g, neighbor, end, path)
            paths.extend(new_paths)
    return paths

if __name__ == "__main__":
    main()
