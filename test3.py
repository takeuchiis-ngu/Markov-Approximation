# coding: UTF-8

#　使ってない

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
    path = path + [start]  # 現在の頂点を経路に追加
    
    if start == end:  # 終点に到達した場合、経路を返す
        return [path]
    
    if start not in g:  # 次に進む辺がない場合、空リストを返す
        return []
    
    paths = []  # すべての経路を格納するリスト
    for neighbor in g[start]:  # 隣接する頂点を探索
        if neighbor not in path:  # 無限ループを防ぐため訪問済みでない頂点のみ進む
            new_paths = find_all_paths(g, neighbor, end, path)
            paths.extend(new_paths)  # 新しい経路を追加
    
    return paths

# タイマーの計算関数（新しいT_kの式）
def calculate_timer(current_max_load, next_max_load, total_path_count, tau=1, beta=30):
    return (1 / total_path_count) * math.exp(tau - beta * (current_max_load - next_max_load)) + 0.2

# シード値の設定
seed_value = 42  # 任意のシード値を指定
random.seed(seed_value)

# グラフ生成パラメータ
node = 14
retu = 9
graph_model = "random"
beta = 20
zeta = 1
count = 20  # 経路変更の最大回数

a = 10  # フロー数はノード数を超えても良い
b = a

# グラフの生成
g = graph_making.Graphs(a, b)
g.randomGraph(g, n=node, k=5, seed=seed_value, number_of_area=1, number_of_areanodes=node, area_height=retu)

# 容量の取得
capacity = nx.get_edge_attributes(g, 'capacity')

# フローの初期化
flows = []
all_paths = []
number_of_paths = []
area_nodes_list = list(g.area_nodes_dict[0])

for _ in range(a):  # フローの設定
    while True:
        s, t = random.sample(area_nodes_list, 2)  # s と t を異なるノードとして選択
        demand = random.randint(5, 15)
        
        paths = find_all_paths(g, s, t)
        if not paths:
            continue  # 経路がない場合は再選択
        
        if not any(flow.get_s() == s and flow.get_t() == t for flow in flows):
            break  # 重複しない場合のみループを抜ける
    
    flow = Flow(g, len(flows), s, t, demand)
    flows.append(flow)
    all_paths.append(paths)
    number_of_paths.append(len(paths))

min_global_max_load_ratio = float('inf')
max_load_ratio_history = []

# フローごとの状態管理
flow_timers = [0] * len(flows)
flow_updates = [0] * len(flows)
selected_paths = [None] * len(flows)

total_path_count = sum(number_of_paths)
start_time = time.time()
end_simulation = False

# 可視化の準備
plt.figure(figsize=(10, 6))
plt.xlabel("Iteration")
plt.ylabel("Maximum Load Ratio")
plt.title("Maximum Load Ratio Over Iterations")
plt.grid(True)

iterations = []
max_load_ratios = []
flow_paths_history = []

i_iteration = 0
while not end_simulation:
    i_iteration += 1
    current_time = time.time() - start_time
    total_demand = collections.defaultdict(float)
    
    for flow in flows:
        path = random.choice(all_paths[flow.get_id()])
        for j in range(len(path) - 1):
            edge = (path[j], path[j + 1])
            total_demand[edge] += flow.get_demand()
    
    current_max_load_ratio = max([flow / capacity[edge] for edge, flow in total_demand.items() if edge in capacity], default=0)
    max_load_ratio_history.append(current_max_load_ratio)
    
    base_start_time = time.time()
    best_timer = float('inf')
    best_flow_index = None
    best_next_path = None
    
    for i, (flow, paths) in enumerate(zip(flows, all_paths)):
        demand = flow.get_demand()
        
        for candidate_path in paths:
            temp_demand = total_demand.copy()
            for j in range(len(candidate_path) - 1):
                edge = (candidate_path[j], candidate_path[j + 1])
                temp_demand[edge] += demand
            
            edge_load_ratios = {edge: flow / capacity[edge] for edge, flow in temp_demand.items() if edge in capacity}
            next_max_load_ratio = max(edge_load_ratios.values()) if edge_load_ratios else 0
            
            timer = calculate_timer(current_max_load_ratio, next_max_load_ratio, total_path_count)
            
            if timer < best_timer:
                best_timer = timer
                best_flow_index = i
                best_next_path = candidate_path
    
    if best_flow_index is not None:
        selected_paths[best_flow_index] = best_next_path
        flow_timers[best_flow_index] = base_start_time + best_timer
        flow_updates[best_flow_index] += 1
    
    time.sleep(max(0, best_timer - (time.time() - base_start_time)))
    
    iterations.append(i_iteration)
    max_load_ratios.append(current_max_load_ratio)
    flow_paths_history.append(selected_paths[:])
    
    if i_iteration % 5 == 0:
        plt.plot(iterations, max_load_ratios, marker='o', label='Max Load Ratio')
        plt.legend()
        plt.pause(0.5)
    
    if all(update >= count for update in flow_updates):
        end_simulation = True

plt.legend()
plt.show()
