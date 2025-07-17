# coding: UTF-8

from mip import *
import networkx as nx
import graph_making
import math
import random
import collections
from flow import Flow
import matplotlib.pyplot as plt
import time
import heapq
import itertools

# K-shortest paths with cost threshold epsilon

def find_k_shortest_paths(G, source, target, k, epsilon, over_factor):
    # 経路コスト計算
    def path_cost(path):
        return sum(G[u][v].get("length", 1) for u, v in zip(path[:-1], path[1:]))

    # ホップ数優先で単純経路を生成
    try:
        gen = nx.shortest_simple_paths(G, source, target, weight=None)
    except nx.NetworkXNoPath:
        return []

    # 候補を k*over_factor 本取得
    candidate_paths = list(itertools.islice(gen, k * over_factor))

    paths = []
    best_cost = float("inf")
    for path in candidate_paths:
        if len(paths) >= k:
            break
        c = path_cost(path)
        if best_cost == float("inf"):
            best_cost = c
        if c <= best_cost * (1 + epsilon):
            paths.append(path)

    return paths

# タイマーの計算関数（新しいT_kの式）
def calculate_timer(current_max_load, next_max_load, total_path_count, tau, beta):
    return (1 / total_path_count) * math.exp(tau - (1/2 * beta * (current_max_load - next_max_load)))

# --- 追加: demand を変更するタイミングと値を指定 ---
demand_change_schedule = {
    # 1: [(0, 2), (4, 2)],
    # 50: [(0, 13), (4,10)],
    # 100: [(1, 20), (8,25)],
}

# グラフ生成パラメータ
node = 14
seed_value = 42
random.seed(seed_value)
a = 15
b = a
beta = 100
tau = 2
count = 300
retu = 4
graph_model = "random"
k_paths = 20 # 1フローあたりの最大経路数
epsilon = 0.1 # 最短経路と比較して許容するコスト差 0.1=10% 

# メイン処理
for _ in range(1):
    g = graph_making.Graphs(a, b)
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
            demand = random.randint(10, 20)
            paths = find_k_shortest_paths(g, s, t, k=k_paths, epsilon=epsilon, over_factor=5)
            if not paths:
                continue
            if(s != t and ((s,t) not in All_commodity_list)):
                break
        demand_list.append(demand)
        All_commodity_list.append((s, t))
        flow = Flow(g, len(flows), s, t, demand)
        flows.append(flow)
        all_paths.append(paths)
        number_of_paths.append(len(paths))

    min_global_max_load_ratio = float('inf')
    max_load_ratio_history = []

    flow_updates = 0
    selected_paths = [None] * len(flows)
    current_paths = [paths[0] if paths else None for paths in all_paths]
    initial_paths = current_paths[:]

    total_path_count = sum(number_of_paths)
    start_time = time.time()
    end_simulation = False

    iterations = []
    max_load_ratios = []
    flow_paths_history = []
    change_history = []

    i_iteration = 0
    loop_start_time = time.time()
    while not end_simulation:
        # カウントを更新
        i_iteration += 1

        # --- 指定されたタイミングで demand を変更 ---
        if i_iteration in demand_change_schedule:
            for flow_idx, new_dem in demand_change_schedule[i_iteration]:
                flows[flow_idx].set_demand(new_dem)
                demand_list[flow_idx] = new_dem
                print(f"Iteration {i_iteration}: Demand for Flow {flow_idx} changed to {new_dem}")

        # 総需要量の計算
        total_demand = collections.defaultdict(float)
        for path, flow in zip(current_paths, flows):
            for j in range(len(path) - 1):
                edge = (path[j], path[j + 1])
                total_demand[edge] += flow.get_demand()

        current_max_load_ratio = max([flow_val / capacity[edge] for edge, flow_val in total_demand.items() if edge in capacity], default=0)
        max_load_ratio_history.append(current_max_load_ratio)
        min_global_max_load_ratio = min(min_global_max_load_ratio, current_max_load_ratio)

        best_timer = float('inf')
        best_flows = []

        for i, (flow, paths, current_path) in enumerate(zip(flows, all_paths, current_paths)):
            demand_val = flow.get_demand()
            temp_delete_current_path = total_demand.copy()
            for j in range(len(current_path) - 1):
                edge = (current_path[j], current_path[j + 1])
                temp_delete_current_path[edge] -= demand_val
            for candidate_path in paths:
                temp_demand = temp_delete_current_path.copy()
                for j in range(len(candidate_path) - 1):
                    edge = (candidate_path[j], candidate_path[j + 1])
                    temp_demand[edge] += demand_val
                edge_load_ratios = {edge: val / capacity[edge] for edge, val in temp_demand.items() if edge in capacity}
                next_max_load_ratio = max(edge_load_ratios.values()) if edge_load_ratios else 0
                timer = calculate_timer(current_max_load_ratio, next_max_load_ratio, total_path_count, tau, beta)
                print(f"Iteration {i_iteration}, Flow {i}: Path {candidate_path}, Timer: {timer:.4f}, Max Load Ratio: {next_max_load_ratio:.4f}")
                if timer < best_timer:
                    best_timer = timer
                    best_flows = [(i, candidate_path)]
                elif timer == best_timer:
                    best_flows.append((i, candidate_path))
        if best_flows:
            best_flow_index, chosen_path = random.choice(best_flows)
            change_history.append(f"Iteration {i_iteration}: Flow {best_flow_index} changed from {current_paths[best_flow_index]} to {chosen_path}")
            current_paths[best_flow_index] = chosen_path
            flow_updates += 1

        iterations.append(i_iteration)
        max_load_ratios.append(current_max_load_ratio)
        flow_paths_history.append(current_paths[:])

        if flow_updates >= count:
            end_simulation = True
    
    loop_end_time = time.time()
    # 結果表示
    print("\nInitial Flow Paths:")
    for i, path in enumerate(initial_paths):
        print(f"Flow {i}: {path}")
    print("\nFlow Change History:")
    for change in change_history:
        print(change)
    print("\nFinal Flow Paths:")
    for i, path in enumerate(current_paths):
        print(f"Flow {i}: {path}")
    print("\n需要量：", demand_list)
    print(All_commodity_list)

    half_index = len(max_load_ratio_history) // 2
    avg_max_load_ratio_late = sum(max_load_ratio_history[half_index:]) / len(max_load_ratio_history[half_index:])
    print(f"\nObserved Minimum Maximum Load Ratio: {min_global_max_load_ratio:.4f}")
    print(f"Average Maximum Load Ratio in the Latter Half: {avg_max_load_ratio_late:.4f}")
    print(f"last Load Ratio: {current_max_load_ratio}")
    print(f"Exploration loop time: {loop_end_time - loop_start_time:.4f} seconds")
    print(capacity)
    print("フローの経路数合計")
    print(total_path_count)

plt.figure(figsize=(10, 6))
plt.plot(iterations, max_load_ratios, marker='o', label='Max Load Ratio')
plt.xlabel("Iteration", fontsize=18)
plt.ylabel("Maximum Load Ratio", fontsize=18)
plt.title("Maximum Load Ratio Over Iterations", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
