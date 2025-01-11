# 厳密解の限界品種数を求めるプログラムファイル
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
import copy
import time
import csv
import numpy as np
import pandas as pd
import random
from Exact_Solution import Solve_exact_solution

from flow import Flow
import exact_limit_graph_making

import os


# 引数
solver_type = "pulp"
graph_model = "nsfnet"
commodity = 10 # nsfnet14node 14*13
capa_l = 100000 # capacityの範囲 500*182 = 91000
capa_h = 100000 # capacityの範囲
demand_l = 10 # 需要量の範囲
demand_h = 500 # 需要量の範囲
degree = 3
node = 1 # ランダムグラフのノード数
determin_st = []
commodity_list = []
limittime = 0.01

for c in range(1, commodity + 1):
    if(graph_model == 'random'):
        G = exact_limit_graph_making.Graphs(c) #品種数,areaの品種数
        G.randomGraph(G, degree , node, capa_l, capa_h) # 5 is each node is joined with its k nearest neighbors in a ring topology. 5はノード数に関係しそう　次数だから
    if(graph_model == 'nsfnet'):
        G = exact_limit_graph_making.Graphs(c) #品種数,areaの品種数　この時点では何もグラフが作られていない　インスタンスの作成？
        G.nsfnet(G, capa_l, capa_h)

    capacity_list = nx.get_edge_attributes(G,'capacity') # 全辺のcapacityの値を辞書で取得
    edge_list = list(enumerate(G.edges()))
    edges_notindex = []
    for z in range(len(edge_list)):
        edges_notindex.append(edge_list[z][1]) 
    # nx.write_gml(G, "./value/graph.gml") # グラフの保存?
    # 保存先ディレクトリを確認し、存在しなければ作成
    save_dir = "./value/"
    os.makedirs(save_dir, exist_ok=True)

    # グラフを保存
    nx.write_gml(G, os.path.join(save_dir, "graph.gml"))

     # 品種の作成
    s , t = tuple(random.sample(list(G.nodes), 2))# source，sink定義
    demand = random.randint(demand_l, demand_h) # demand設定
    tentative_st = [s,t]
    if c != 1:
        while True:
            if tentative_st in determin_st:
                s , t = tuple(random.sample(G.nodes, 2)) # source，sink再定義
                tentative_st = [s,t]
            else:
                break
    determin_st.append(tentative_st) # commodity決定
    commodity_list.append([s,t,demand])
    # commodity_list.sort(key=lambda x: -x[2]) # demand大きいものから降順


    with open('./value/commodity_data.csv','w') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerows(commodity_list) # 品種の保存

    exact_file_name = f'./value/exactsolution_limit.csv' 
    E = Solve_exact_solution(c,solver_type,exact_file_name,limittime) # Exact_Solution.pyの厳密解クラスの呼び出し
    objective_value,objective_time = E.solve_exact_solution_to_env() # 厳密解を計算

    node = node + 1 # ランダムグラフの場合の繰り上げ