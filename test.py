# coding: UTF-8

# --- 省略部分 ---
from mip import *
import networkx as nx
import graph_making
from flow import Flow
import matplotlib.pyplot as plt
import random
import time
import csv

kurikaesi = 1
a = 15
b = a
c = 1
retu = 4
d = retu*retu
node = 10
graph_model = "random"
seed_value = 42
random.seed(seed_value)

while(kurikaesi > 0):

    if(graph_model == 'random'):
        g = graph_making.Graphs(a,b)
        g.randomGraph(g, n=node, k=5, seed=seed_value, number_of_area=1, number_of_areanodes=node, area_height=retu)

    r_kakai = list(enumerate(g.edges()))
    edge_dic_kakai = {(i,j):m for m,(i,j) in r_kakai}
    num_dic_kakai = {m:(i,j) for m,(i,j) in r_kakai}

    k = g.k
    area_k = g.area_k
    ELB_paths = dict()
    All_commodity_list = []
    demand_list = []
    max_L_optimize = 0
    commodity_count = 0

    for area in range(g.number_of_area):
        area_nodes_list = list(g.area_nodes_dict[area])
        tuples = []
        flows_list = []
        area_commodity_count = 0

        for _ in range(a):
            while True:
                s, t = random.sample(area_nodes_list, 2)
                demand = random.randrange(30, 40)
                if(s != t and ((s,t) not in All_commodity_list)):
                    break

            tuples.append((s,t))
            All_commodity_list.append((s,t))
            demand_list.append(demand)

            f = Flow(g,commodity_count,s,t,demand)
            f.set_area_flow_id(area_commodity_count)
            flows_list.append(f)
            g.all_flows.append(f)
            commodity_count += 1
            area_commodity_count += 1

        g.all_flow_dict[area] = flows_list

    UELB_kakai = Model("UELB_kakai")
    L_kakai = UELB_kakai.add_var("L_kakai", lb=0)

    flow_var_kakai = []
    for l in range(len(g.all_flows)):
        x_kakai = [UELB_kakai.add_var(f"x{l}_{m}", var_type=BINARY) for m, _ in r_kakai]
        flow_var_kakai.append(x_kakai)

    capacity = nx.get_edge_attributes(g, 'capacity')

    for e in range(len(g.edges())):
        edge = r_kakai[e][1]
        cap = capacity.get(edge)
        if cap is None:
            rev_edge = (edge[1], edge[0])
            cap = capacity.get(rev_edge)
        if cap is None or cap == 0:
            continue
        UELB_kakai += ((xsum([flow_var_kakai[l.get_id()][e] * l.get_demand() for l in g.all_flows])) / cap) <= L_kakai

    for l in g.all_flows:
        UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*l.get_demand() for e in range(len(g.edges())) if r_kakai[e][1][0] == l.get_update_s()]) == l.get_demand()
        UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*l.get_demand() for e in range(len(g.edges())) if r_kakai[e][1][1] == l.get_update_s()]) == 0
        UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*l.get_demand() for e in range(len(g.edges())) if r_kakai[e][1][0] == l.get_update_t()]) == 0
        UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*l.get_demand() for e in range(len(g.edges())) if r_kakai[e][1][1] == l.get_update_t()]) == l.get_demand()

        for v in g.nodes():
            if v != l.get_update_s() and v != l.get_update_t():
                UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*l.get_demand() for e in range(len(g.edges())) if r_kakai[e][1][0] == v]) \
                              == xsum([flow_var_kakai[l.get_id()][e]*l.get_demand() for e in range(len(g.edges())) if r_kakai[e][1][1] == v])

    UELB_kakai.objective = minimize(L_kakai)

    start_kakai = time.time()
    UELB_kakai.optimize()
    elapsed_time_kakai = time.time() - start_kakai

    if UELB_kakai.num_solutions > 0:
        print("Objective :", UELB_kakai.objective_value)
        print(All_commodity_list)
        print("需要量：", demand_list)
        print(capacity)

        result_flow_var_kakai = []
        for l in range(len(g.all_flows)):
            result_x = []
            for m in range(len(r_kakai)):
                result_x.append(flow_var_kakai[l][m].x)
            result_flow_var_kakai.append(result_x)

        all_path = []
        for l in range(len(g.all_flows)):
            commodity_path = []
            for m,(i,j) in r_kakai:
                if result_flow_var_kakai[l][m] == 1:
                    commodity_path.append((i,j))
            all_path.append(commodity_path)

        print(result_flow_var_kakai)
        print(all_path)
        print("time :", elapsed_time_kakai)
        print("--------------------------------------------")

        with open('mip.csv', 'a', newline='') as f:
            out = csv.writer(f)
            if a == 1:
                out.writerow(['Objective','time','graph'])
            out.writerow([UELB_kakai.objective_value, elapsed_time_kakai, graph_model])

        nx.draw(g, with_labels=True)
        plt.show()

    else:
        print("No solution found. Status:", UELB_kakai.status)

    kurikaesi -= 1
