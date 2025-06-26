import time
import random
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
import graph_making
import threading

# ==========================
# ユーザー設定パラメータ
# ==========================
flow_count = 10         # フロー数（aに渡す）
area_k = 1              # エリア数（bに渡す）
node = 20               # ノード数
seed_value = 42         # シード値
transition_limit = 10   # 状態遷移の回数
retu = 3                # グリッド縦高さ（エリア描画用）
# ==========================

random.seed(seed_value)

# ==========================
# グラフの生成
# ==========================
g = graph_making.Graphs(flow_count, area_k)
G = nx.DiGraph()
g.randomGraph(g, n=node, k=5, seed=seed_value,
              number_of_area=1, number_of_areanodes=node, area_height=retu)

# ==========================
# 経路探索
# ==========================
def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for node in graph.neighbors(start):
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

# ==========================
# フロー生成
# ==========================
flows = []
for _ in range(flow_count):
    while True:
        s, d = random.sample(list(G.nodes), 2)
        paths = find_all_paths(G, s, d)
        if paths:
            flows.append({"source": s, "dest": d, "paths": paths})
            break

# 初期経路の割り当て
current_paths = [flow["paths"][0] for flow in flows]

# 単位需要量（全フロー共通）
def get_demand():
    return 1
# ==========================
# 負荷計算
# ==========================
def compute_total_demand(paths):
    demand = {}
    for path in paths:
        d = get_demand()
        for u, v in zip(path[:-1], path[1:]):
            demand[(u, v)] = demand.get((u, v), 0) + d
    return demand

# 各エッジの容量を取得
capacity_map = nx.get_edge_attributes(G, "capacity")

def compute_max_load(demand):
    max_ratio = 0
    for (u, v), d in demand.items():
        c = capacity_map.get((u, v), 1)
        ratio = d / c
        if ratio > max_ratio:
            max_ratio = ratio
    return max_ratio

# ==========================
# タイマーによる経路評価
# ==========================
history = []
current_demand = compute_total_demand(current_paths)
current_max_load = compute_max_load(current_demand)
history.append(current_max_load)
print(f"Initial max load: {current_max_load:.4f}")

lock = threading.Lock()
stop_flag = threading.Event()
best_result = {"index": None, "path": None, "timer": float("inf")}

def timer_task(flow_idx, start_time):
    global best_result
    current_path = current_paths[flow_idx]
    for candidate in flows[flow_idx]["paths"]:
        if candidate == current_path:
            continue  # 経路変更は必須

        temp_paths = current_paths[:]
        temp_paths[flow_idx] = candidate
        temp_demand = compute_total_demand(temp_paths)
        temp_max_load = compute_max_load(temp_demand)
        delta = current_max_load - temp_max_load

        if delta <= 0:
            continue

        timer_val = abs(delta)
        now = time.time()
        wait_time = max(0.0, start_time + timer_val - now)
        time.sleep(wait_time)

        if not stop_flag.is_set():
            with lock:
                if timer_val < best_result["timer"]:
                    best_result = {"index": flow_idx, "path": candidate, "timer": timer_val}
                    stop_flag.set()
        break

# ==========================
# 状態遷移ループ
# ==========================
for t in range(transition_limit):
    print(f"\n--- Transition {t + 1} ---")
    stop_flag.clear()
    best_result = {"index": None, "path": None, "timer": float("inf")}
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=min(flow_count, 16)) as executor:
        for i in range(flow_count):
            executor.submit(timer_task, i, start_time)

    if stop_flag.is_set() and best_result["index"] is not None:
        idx = best_result["index"]
        current_paths[idx] = best_result["path"]
        current_demand = compute_total_demand(current_paths)
        current_max_load = compute_max_load(current_demand)
        history.append(current_max_load)
        print(f"Changed path of flow {idx}, new max load: {current_max_load:.4f}")
    else:
        print("No better path found.")
        break

# ==========================
# 可視化
# ==========================
plt.figure(figsize=(8, 4))
plt.plot(history, marker='o')
plt.title("Maximum Link Load Ratio Over Transitions")
plt.xlabel("Transition Step")
plt.ylabel("Max Load Ratio")
plt.grid(True)
plt.tight_layout()
plt.show()
