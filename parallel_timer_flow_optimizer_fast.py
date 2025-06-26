from graph_making_compatible import generate_graph, find_all_paths, capacity
from flow import Flow
import networkx as nx
import collections
import concurrent.futures
import threading
import time
from decimal import Decimal
import matplotlib.pyplot as plt
import random
import os

def main():
    a = 4  # フロー数
    num_nodes = 20  # ノード数
    seed = 42  # 乱数シード
    count = 50  # 状態遷移の回数

    random.seed(seed)

    G = generate_graph(num_nodes)
    flows = [Flow(G) for _ in range(a)]

    all_paths = []
    for flow in flows:
        paths = find_all_paths(G, flow.source, flow.destination)
        if not paths:
            raise ValueError(f"No paths found from {flow.source} to {flow.destination}")
        all_paths.append(paths)

    current_paths = [paths[0] for paths in all_paths]
    total_demand = collections.defaultdict(int)

    for i, path in enumerate(current_paths):
        demand = flows[i].get_demand()
        for j in range(len(path) - 1):
            edge = (path[j], path[j + 1])
            total_demand[edge] += demand

    max_load_ratio_history = []
    capacity_map = capacity(G)

    def compute_max_load(demand_map):
        return max(
            [Decimal(flow) / Decimal(capacity_map[edge]) for edge, flow in demand_map.items() if edge in capacity_map],
            default=Decimal(0)
        )

    def update_total_demand(paths):
        temp_demand = collections.defaultdict(int)
        for i, path in enumerate(paths):
            demand = flows[i].get_demand()
            for j in range(len(path) - 1):
                edge = (path[j], path[j + 1])
                temp_demand[edge] += demand
        return temp_demand

    current_max_load_ratio = compute_max_load(total_demand)
    max_load_ratio_history.append(current_max_load_ratio)

    print(f"Initial max load ratio: {current_max_load_ratio:.4f}")

    cpu_count = os.cpu_count()
    lock = threading.Lock()
    stop_flag = threading.Event()
    best_result = {"index": None, "path": None, "timer": float("inf")}

    def timer_task(flow_index):
        nonlocal best_result
        local_best = {"path": None, "timer": float("inf")}

        for candidate_path in all_paths[flow_index]:
            if candidate_path == current_paths[flow_index]:
                continue

            temp_paths = current_paths[:]
            temp_paths[flow_index] = candidate_path
            temp_demand = update_total_demand(temp_paths)
            temp_max = compute_max_load(temp_demand)

            timer_val = float((current_max_load_ratio - temp_max).copy_abs())
            if timer_val <= 0:
                continue

            start_time = time.time()
            time.sleep(timer_val)  # 実際のカウントダウン

            if not stop_flag.is_set():
                with lock:
                    if timer_val < best_result["timer"]:
                        best_result.update({"index": flow_index, "path": candidate_path, "timer": timer_val})
                        stop_flag.set()
            break
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as executor:
            executor.map(timer_task, range(a))

        if stop_flag.is_set() and best_result["index"] is not None:
            current_paths[best_result["index"]] = best_result["path"]
            total_demand = update_total_demand(current_paths)
            current_max_load_ratio = compute_max_load(total_demand)
            print(f"Selected flow {best_result['index']} with timer {best_result['timer']:.6f}")
            print(f"Updated max load ratio: {current_max_load_ratio:.4f}")
        else:
            print("No better path found.")
            break

        max_load_ratio_history.append(current_max_load_ratio)

    # 可視化
    plt.plot([float(val) for val in max_load_ratio_history], marker='o')
    plt.title("Max Load Ratio Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Max Load Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
