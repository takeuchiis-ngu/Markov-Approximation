# coding: UTF-8
import networkx as nx
import graph_making
import math
import random
import collections
from flow import Flow
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
import time
import concurrent.futures
import multiprocessing

getcontext().prec = 30  # 高精度計算用

def calculate_timer(current_max_load, next_max_load, total_path_count, tau, beta):
    diff = Decimal(current_max_load) - Decimal(next_max_load)
    exponent = Decimal(tau) - Decimal("0.5") * Decimal(beta) * diff
    return Decimal(1) / Decimal(total_path_count) * Decimal(math.exp(float(exponent))) + Decimal(0.1)

def evaluate_candidate(i, current_path, candidate_path, flows, total_demand, capacity, total_path_count, current_max_load_ratio):
    flow = flows[i]
    demand = flow.get_demand()

    temp_demand = total_demand.copy()
    for j in range(len(current_path) - 1):
        edge = (current_path[j], current_path[j + 1])
        temp_demand[edge] -= demand

    for j in range(len(candidate_path) - 1):
        edge = (candidate_path[j], candidate_path[j + 1])
        temp_demand[edge] += demand

    edge_load_ratios = {
        edge: Decimal(temp_demand[edge]) / Decimal(capacity[edge])
        for edge in temp_demand if edge in capacity
    }

    next_max_load_ratio = max(edge_load_ratios.values()) if edge_load_ratios else Decimal(0)
    timer = calculate_timer(current_max_load_ratio, next_max_load_ratio, total_path_count, tau=2, beta=160)

    return {
        'flow_index': i,
        'candidate_path': candidate_path,
        'timer': timer,
        'next_max_load': next_max_load_ratio
    }

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

def main():
    node = 10
    seed_value = 42
    a = 15
    count = 30
    retu = 4

    random.seed(seed_value)
    g = graph_making.Graphs(a, a)
    g.randomGraph(
        g,
        n=node,
        k=5,
        seed=seed_value,
        number_of_area=1,
        number_of_areanodes=node,
        area_height=retu
    )

    capacity = nx.get_edge_attributes(g, 'capacity')
    area_nodes_list = list(g.area_nodes_dict[0])
    All_commodity_list = []
    demand_list = []
    flows = []
    all_paths = []
    number_of_paths = []

    for _ in range(a):
        while True:
            s, t = random.sample(area_nodes_list, 2)
            demand = random.randint(5, 15)
            paths = find_all_paths(g, s, t)
            paths.sort(key=lambda p: len(p))  # ホップ数の少ない順に並べる
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

        current_max_load_ratio = max(
            [Decimal(flow) / Decimal(capacity[edge]) for edge, flow in total_demand.items() if edge in capacity],
            default=Decimal(0)
        )
        max_load_ratio_history.append(current_max_load_ratio)

        early_accept_threshold = Decimal("1e-8")
        found = False
        best_result = None
        with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = []
            for i, (current_path, paths) in enumerate(zip(current_paths, all_paths)):
                for candidate_path in paths:
                    if candidate_path == current_path:
                        continue
                    futures.append(executor.submit(
                        evaluate_candidate,
                        i,
                        current_path,
                        candidate_path,
                        flows,
                        total_demand,
                        capacity,
                        total_path_count,
                        current_max_load_ratio
                    ))

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                timer = result['timer']

                # 高精度で表示
                # print(f"[Timer Check] Flow {result['flow_index']} → {result['candidate_path']} | timer = {timer:.10E}")

                if timer <= early_accept_threshold:
                    print("🟢 Early accept: timer below threshold")
                    best_result = result
                    found = True
                    break

            if not found:
                # しきい値以下が無かった場合、最小の timer 値のものを選ぶ
                sorted_results = sorted(
                    [f.result() for f in futures],
                    key=lambda x: x['timer']
                )
                if sorted_results:
                    best_result = sorted_results[0]
                    # print("🟡 Accepted best (non-threshold) timer")

        if best_result:
            i = best_result['flow_index']
            new_path = best_result['candidate_path']
            change_history.append(
                f"Iteration {i_iteration}: Flow {i} changed from {current_paths[i]} to {new_path} "
                f"(timer = {best_result['timer']:.10E})"
            )
            current_paths[i] = new_path
            flow_updates += 1
        else:
            print("⚠️ No valid path change found.")
    print("\nInitial Paths:")
    for i, path in enumerate(initial_paths):
        print(f"Flow {i}: {path}")

    print("\nChange History:")
    for entry in change_history:
        print(entry)

    print("\nFinal Paths:")
    for i, path in enumerate(current_paths):
        print(f"Flow {i}: {path}")

    if len(max_load_ratio_history) > 0:
        half = len(max_load_ratio_history) // 2
        avg_late = sum(max_load_ratio_history[half:]) / Decimal(len(max_load_ratio_history[half:]))
        print(f"\nFinal Average Max Load (late half): {avg_late:.10f}")
    else:
        print("\nNo max load history recorded.")
    
    print(demand_list)

    # 可視化
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(max_load_ratio_history) + 1),
             [float(val) for val in max_load_ratio_history],
             marker='o')
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Maximum Load Ratio", fontsize=14)
    plt.title("Maximum Load Ratio Over Iterations", fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
