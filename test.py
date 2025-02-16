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

# タイマーの計算関数
def calculate_timer(current_max_load, next_max_load, total_path_count, tau=2, beta=20):
    return (1 / total_path_count) * math.exp(tau - beta * (current_max_load - next_max_load))

# シード値の設定
seed_value = 42
random.seed(seed_value)

# グラフ生成パラメータ
node = 14
retu = 9
graph_model = "random"
beta = 20
zeta = 1
count = 20

a = 15
b = a

# ユーザーが入力可能なパラメータ
# flow_add_timing = 10,20
# add_flow = (1,5),(2,7)
flow_add_timing = 10
add_flow = (1,5)
# flow_add_timing = None
# add_flow = None
new_flow_iterations = [flow_add_timing]  # 追加するタイミング
new_flow_pairs = [add_flow]  # 追加するフローの始点と終点

# edge_failed_timing = 15,20
# failed_edge = (2,5),(6,7)
edge_failed_timing = 15
failed_edge = (2,5)
edge_failed_timing = None
failed_edge = None
failed_edge_iterations = [edge_failed_timing]  # エッジ障害発生のタイミング
failed_edges_list = [failed_edge]  # 消失するエッジ

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
current_paths = []

def add_new_flow(s, t):
    if s is None or t is None:
        return
    demand = random.randint(5, 15)
    paths = find_all_paths(g, s, t)
    if paths:
        flow = Flow(g, len(flows), s, t, demand)
        flows.append(flow)
        all_paths.append(paths)
        number_of_paths.append(len(paths))
        current_paths.append(random.choice(paths))
        
def handle_edge_failure(edge):
    if edge is None:
        return    
    print(f"Edge {edge} failed. Re-routing affected flows.")
    failed_edges.add(edge)
    if edge in capacity:
        del capacity[edge]
    
    affected_flows = []
    for i, path in enumerate(current_paths):
        if any((path[j], path[j+1]) == edge for j in range(len(path) - 1)):
            affected_flows.append(i)
    
    for i in affected_flows:
        new_paths = [p for p in all_paths[i] if not any((p[j], p[j+1]) in failed_edges for j in range(len(p) - 1))]
        if new_paths:
            new_path = random.choice(new_paths)
            print(f"Flow {i} re-routed from {current_paths[i]} to {new_path}")
            current_paths[i] = new_path

def get_safe_demand(flow):
    if flow is None:
        return 0
    try:
        return flow.get_demand()
    except AttributeError:
        return 0
        
for _ in range(a):
    while True:
        s, t = random.sample(area_nodes_list, 2)
        demand = random.randint(5, 15)
        
        paths = find_all_paths(g, s, t)
        if not paths:
            continue
        
        if not any(flow.get_s() == s and flow.get_t() == t for flow in flows):
            break
    
    flow = Flow(g, len(flows), s, t, demand)
    flows.append(flow)
    all_paths.append(paths)
    number_of_paths.append(len(paths))
    current_paths.append(random.choice(paths))

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

failed_edges = set()

while not end_simulation:
    i_iteration += 1
    total_demand = collections.defaultdict(float)
    
    for path, flow in zip(current_paths, flows):
        for j in range(len(path) - 1):
            edge = (path[j], path[j + 1])
            if edge not in failed_edges and flow is not None:
                total_demand[edge] += get_safe_demand(flow)
    if failed_edge_iterations and i_iteration in failed_edge_iterations:
        index = failed_edge_iterations.index(i_iteration)
        handle_edge_failure(failed_edges_list[index])    
    if new_flow_iterations and i_iteration in new_flow_iterations:
        index = new_flow_iterations.index(i_iteration)
        add_new_flow(*new_flow_pairs[index])
    
    iterations.append(i_iteration)
    max_load_ratios.append(min_global_max_load_ratio)    
    flow_paths_history.append(current_paths[:])
    
    if flow_updates >= count:
        end_simulation = True

half_index = len(max_load_ratio_history) // 2
avg_max_load_ratio_late = sum(max_load_ratio_history[half_index:]) / len(max_load_ratio_history[half_index:])

print(f"\nObserved Minimum Maximum Load Ratio: {min_global_max_load_ratio:.4f}")
print(f"Average Maximum Load Ratio in the Latter Half: {avg_max_load_ratio_late:.4f}")

print("\nInitial Flow Paths:")
for i, path in enumerate(initial_paths):
    print(f"Flow {i}: {path}")

print("\nFlow Change History:")
for change in change_history:
    print(change)

print("\nFinal Flow Paths:")
for i, path in enumerate(current_paths):
    print(f"Flow {i}: {path}")

plt.figure(figsize=(10, 6))
plt.plot(iterations, max_load_ratios, marker='o', label='Max Load Ratio')
plt.xlabel("Iteration")
plt.ylabel("Maximum Load Ratio")
plt.title("Maximum Load Ratio Over Iterations")
plt.legend()
plt.grid(True)
plt.show()
