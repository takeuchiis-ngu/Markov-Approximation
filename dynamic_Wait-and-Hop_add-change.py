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

# すべてのパスの組み合わせの中からフローが取り得る経路を求める関数
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

# タイマーの計算関数（新しいT_kの式）
def calculate_timer(current_max_load, next_max_load, total_path_count, tau, beta):
    return (1 / total_path_count) * math.exp(tau - (1/2 * beta * (current_max_load - next_max_load)))

# --- 追加: demand を変更するタイミングと値を指定 ---
# {iteration_number: [(flow_index, new_demand), ...], ...}
demand_change_schedule = {
    1: [(0, 5), (4, 5)],
    # 例: 50回目のループでフロー0の需要を25に変更
    30: [(0, 9), (4,10)],
    # 100回目でフロー1を30, フロー2を35に変更
    # 100: [(1, 30), (2, 35)],
}

# グラフ生成パラメータ
node = 14  # NSFNET使用中は14で固定
seed_value = 40  # 任意のシード値を指定
random.seed(seed_value)
a = 15  # 品種数
b = a
beta = 100
tau = 2
count = 300  # 経路変更の最大回数
retu = 4
graph_model = "random"

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
            paths = find_all_paths(g, s, t)
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
    current_paths = [random.choice(paths) for paths in all_paths]
    initial_paths = current_paths[:]

    total_path_count = sum(number_of_paths)
    start_time = time.time()
    end_simulation = False

    iterations = []
    max_load_ratios = []
    flow_paths_history = []
    change_history = []

    i_iteration = 0
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

    half_index = len(max_load_ratio_history) // 2
    avg_max_load_ratio_late = sum(max_load_ratio_history[half_index:]) / len(max_load_ratio_history[half_index:])
    print(f"\nObserved Minimum Maximum Load Ratio: {min_global_max_load_ratio:.4f}")
    print(f"Average Maximum Load Ratio in the Latter Half: {avg_max_load_ratio_late:.4f}")
    print(f"last Load Ratio: {current_max_load_ratio}")
    print(capacity)
    print("フローの経路数合計")
    print(total_path_count)
    print(All_commodity_list)

plt.figure(figsize=(10, 6))
plt.plot(iterations, max_load_ratios, marker='o', label='Max Load Ratio')
plt.xlabel("Iteration", fontsize=18)
plt.ylabel("Maximum Load Ratio", fontsize=18)
plt.title("Maximum Load Ratio Over Iterations", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
