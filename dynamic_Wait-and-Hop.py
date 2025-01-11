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

# すべてのパスの組み合わせの中から負荷率が最小になる組み合わせを見つける関数
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

# 滞在時間の計算関数
def calculate_stay_time(flow_path_count,total_path_count, zeta, beta, min_global_max_load_ratio, flow_min_max_load_ratio):
    stay_time = (flow_path_count * zeta * math.exp(beta * min_global_max_load_ratio)) / \
                (flow_path_count * math.exp(beta * flow_min_max_load_ratio))
    return stay_time


a = 20
b = a #int (input('area品種数>>'))
# グラフ生成パラメータ
node = 10
retu = 9
graph_model = "random"
beta = 20
zeta = 1

count = 20
# シード値の設定
seed_value = 42  # 任意のシード値を指定
random.seed(seed_value)



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
        s = random.choice(area_nodes_list)
        t = random.choice(area_nodes_list)
        while s == t:
            t = random.choice(area_nodes_list)

        demand = random.randint(5, 15)  # 需要量をランダムに設定
        # demand = 10
        if not any(flow.get_s() == s and flow.get_t() == t and flow.get_demand() == demand for flow in flows):
            break  # 重複しない場合のみループを抜ける
        
        
    flow = Flow(g, len(flows), s, t, demand)
    flows.append(flow)  #各品種の識別番号、始点、終点、需要量
    
    paths = find_all_paths(g, s, t)
    all_paths.append(paths) # フローの総経路情報
    number_of_paths.append(len(paths))  #各フローの総経路数

min_global_max_load_ratio = float('inf')
max_load_ratio_history = []  # 最大負荷率の履歴を記録

# フローごとの状態管理
flow_timers = [0] * len(flows)
flow_updates = [0] * len(flows)

start_time = time.time()
end_simulation = False



while not end_simulation:
    # 現在の時刻を取得
    current_time = time.time() - start_time

    # 辺ごとの需要量を累積
    total_demand = collections.defaultdict(float)
    selected_paths = []

    for i, (flow, paths) in enumerate(zip(flows, all_paths)):
        if flow_timers[i] <= current_time:  # フローの経路変更タイミング
            demand = flow.get_demand()
            selected_path = random.choice(paths)  # ランダムに経路を選択
            selected_paths.append(selected_path)
            
            print(f"Flow {i + 1} selects path: {selected_path} with demand: {demand}")

            # 負荷を更新
            for j in range(len(selected_path) - 1):
                edge = (selected_path[j], selected_path[j + 1])
                total_demand[edge] += demand

            # 負荷率を計算
            edge_load_ratios = {}
            for edge, total_flow in total_demand.items():
                edge_load_ratios[edge] = total_flow / capacity[edge]

            # 最大負荷率を計算
            current_max_load_ratio = max(edge_load_ratios.values())
            max_load_ratio_history.append(current_max_load_ratio)
            min_global_max_load_ratio = min(min_global_max_load_ratio, current_max_load_ratio)

            # 各フローの最大負荷率を計算
            max_ratio = 0
            for j in range(len(selected_path) - 1):
                edge = (selected_path[j], selected_path[j + 1])
                max_ratio = max(max_ratio, edge_load_ratios[edge])
            
            print(f"Flow {i + 1} max ratio: {max_ratio}")

            # 滞在時間を計算
            flow_path_count = len(selected_path)
            total_path_count = sum(number_of_paths)
            stay_time = calculate_stay_time(flow_path_count, total_path_count, zeta, beta, min_global_max_load_ratio, max_ratio)
            
            print(f"Flow {i + 1} stay time: {stay_time:.4f}")


            # 次の経路変更タイミングを設定
            flow_timers[i] = current_time + stay_time
            flow_updates[i] += 1

    # 全フローの経路変更が規定回数に達したか確認
    if all(update >= count for update in flow_updates):
        end_simulation = True

# 時間平均の最大負荷率を計算
if len(max_load_ratio_history)>100:
    time_average_max_load_ratio = sum(max_load_ratio_history[1000:]) / len(max_load_ratio_history[1000:])
    print(f"\nTime-Averaged Maximum Load Ratio: {time_average_max_load_ratio:.4f}")
else:
    time_average_max_load_ratio = sum(max_load_ratio_history) / len(max_load_ratio_history)
    print(f"\nTime-Averaged Maximum Load Ratio: {time_average_max_load_ratio:.4f}")

# 最後に観測した最大負荷率の中で最小の値を表示
min_final_max_load_ratio = min(max_load_ratio_history)
print(f"Minimum of Final Observed Maximum Load Ratios: {min_final_max_load_ratio:.4f}")

# 全フローの始点と終点を表示
print("\nAll Flows' Start and End Points:")
for i, flow in enumerate(flows):
    print(f"Flow {i + 1}: Start = {flow.get_s()}, End = {flow.get_t()}")

# 全フローの需要量を表示
print("\nAll Flows' Demands:")
for i, flow in enumerate(flows):
    print(f"Flow {i + 1}: Demand = {flow.get_demand()}")

# 可視化
plt.figure(figsize=(10, 6))
plt.plot(range(len(max_load_ratio_history)), max_load_ratio_history, marker='o', label='Max Load Ratio')
plt.axhline(y=min_final_max_load_ratio, color='r', linestyle='--', label='Min of Final Max Load Ratios')
plt.title("Maximum Load Ratio Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Maximum Load Ratio")
plt.legend()
plt.grid(True)
plt.show()
