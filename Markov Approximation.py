#MIP(UELB problem)
# coding: UTF-8

# import pulp
# from pulp import *
# from mip import Model, maximize, minimize, xsum
from mip import *
import networkx as nx
import graph_making
import matplotlib.pyplot as plt
import math
import random
import re
import collections
import time
import csv
import numpy as np
import copy
import sys
#import areaGraph
from flow import Flow
from operator import itemgetter, attrgetter
from collections import defaultdict

# すべてのパスの組合せの中から，負荷率が最小になる組合せを見つける関数
# O(|Pl|^k)になるので，指数時間になる

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



kurikaesi = 0
a = 3
b = a #int (input('area品種数>>'))
#d = 16 #int (input('エリアのノード数>>'))
retu = 9
d = retu*retu
node = 10
graph_model = "random"
# random.seed(2) #ランダムの固定化
elapsed_time_kakai = 0
beta = 20   #
zeta = 0.01 #遷移率に関わる

# logの削除
# if isinstance(pulp.LpSolverDefault, pulp.PULP_CBC_CMD):
#     pulp.LpSolverDefault.msg = False



while(kurikaesi < 1):

    if(graph_model == 'grid'):
        g = graph_making.Graphs(a,b) #品種数,areaの品種数
        g.gridMaker(g,1,d,retu,retu,0.1) #G,エリア数,エリアのノード数,列数,行数,ε浮動小数点
        # nx.draw(g, with_labels=True)
        # plt.show()

    if(graph_model == 'random'):
        g = graph_making.Graphs(a,b) #品種数,areaの品種数
        g.randomGraph(g, n=node, k=5, seed=kurikaesi, number_of_area =1, number_of_areanodes = node, area_height=retu) 
        # nx.draw(g, with_labels=True)
        # plt.show()

    r_kakai = list(enumerate(g.edges())) #グラフのエッジと番号を対応付けたもの
    # print(r_kakai)
    edge_dic_kakai = {(i,j):m for m,(i,j) in r_kakai} #{(0, 1): 0, (0, 9): 1, (0, 2): 2,.....}
    num_dic_kakai = {m:(i,j) for m,(i,j) in r_kakai} #{0: (0, 1), 1: (0, 9), 2: (0, 2),.....}
    
    # r_kakaiの中身は
    # [
    #   enumerateによる番号
    #   (node,node)
    # ]
    # の２段構成
    # print(edge_dic_kakai)
    # print("\n")
    # print(num_dic_kakai)

    #品種数
    k = g.k
    #1管理領域の品種数
    area_k = g.area_k

    #ELBのパス保存用dict
    ELB_paths = dict()

    All_commodity_list = []

    # # #start時間を知るための保存リスト（0番目が処理開始時間）
    # OPTtimes = []
    # LPtimes_frac = []
    # LPtimes_donyoku_frac = []
    init_times = []
    part_times = []

    # # #statusを知るための保存リスト（0番目が処理開始時間）
    # statuss = []
    # statuss_frac = []
    # statuss_donyoku_frac = []
    init_status = []
    part_status_list = []

    max_L_optimize = 0

    commodity_count = 0

    #-------------------------------------
    # 管理領域ごとに品種を用意する
    #-------------------------------------
    for area in range(g.number_of_area):

        area_nodes_list = list(g.area_nodes_dict[area])

        #始点と終点の集合S,Tをつくる
        tuples = []
        flows_list = []
        area_commodity_count = 0
        all_paths = []  #全ての各品種が取り得る経路情報
        number_of_paths = []    #全ての各品種が取り得る経路の数

        while(len(tuples)<area_k):
            if(graph_model == 'grid'):
                s = random.randrange(area_nodes_list[0],area_nodes_list[g.number_of_areanodes-1]+1,1)
                t = random.randrange(area_nodes_list[0],area_nodes_list[g.number_of_areanodes-1]+1,1)
                demand  = random.randrange(50, 101, 10)
                # demand = 50
            
            if(graph_model == 'random'):#ノードの中から始点と終点を1つずつランダムで選ぶ
                s = random.randrange(area_nodes_list[0]-1,area_nodes_list[g.number_of_areanodes-1],1)
                t = random.randrange(area_nodes_list[0]-1,area_nodes_list[g.number_of_areanodes-1],1)
                # if len(tuples) == 0:
                #     s = 0
                #     t = 2
                # elif len(tuples) == 1:
                #     s = 0
                #     t = 1
                # else:
                #     s = 3
                #     t = 2
                # demand = random.randrange(100, 201, 10)
                demand = 10

            if(s!=t and ((s,t) not in All_commodity_list)): #流し始めのノードからそのノードに戻ってくる組み合わせは考えていない 
                tuples.append((s,t))
                All_commodity_list.append((s,t))
                # print((s,t))
                # print(All_commodity_list)


                f = Flow(g,commodity_count,s,t,demand)
                f.set_area_flow_id(area_commodity_count)
                flows_list.append(f)
                g.all_flows.append(f)
                ####print(f.get_update_s_area())
                

                # 経路を求める
                paths = find_all_paths(g, s, t)
                all_paths.append(paths)
                number_of_paths.append(len(paths))      

        
                commodity_count +=1
                area_commodity_count += 1
                

        
        g.all_flow_dict[area] = flows_list
        ####print(g.get_i_area_flows(area))
        



#     #容量制限
    capacity = nx.get_edge_attributes(g,'capacity')#全辺のcapacityの値を辞書で取得。エッジ(node,node)をキーとしてcapacityの値を取得できる。 {(0, 1): 580, (0, 9): 860, (0, 2): 980,....}

    
#     #終了時間の計測
#     elapsed_time_kakai = time.time()-start_kakai
#     status = UELB_kakai2.status #実行結果のステータスコード

    de = [] #各品種の需要量
    for l in g.all_flows:
        de.append(l.get_demand())

    
    # # print(result_flow_var_kakai)
    print(f'各品種の需要量：{de}')


    # 辺ごとの需要量を累積
    total_demand = defaultdict(float)

    select_flow = []
    
    for n,path in enumerate(all_paths,start=0):  #品種毎に経路情報取り出す
        demand = de[n]
        r = random.randint(0, len(path)-1)
        select_flow.append(path[r])
        for i in range(len(path[r]) - 1):   #経路情報から1つランダムに経路選択
            edge = (path[r][i], path[r][i + 1])
            total_demand[edge] += demand

    # 負荷率を計算し、最大負荷率とその辺を求める
    max_load_ratio = 0
    max_edge = None

    for edge, total_flow in total_demand.items():
        load_ratio = total_flow / capacity[edge]
        if load_ratio > max_load_ratio:
            max_load_ratio = load_ratio
            max_edge = edge
    
    
    
    # 辺ごとの負荷率を計算
    edge_load_ratios = {}
    for edge, total_flow in total_demand.items():
        edge_load_ratios[edge] = total_flow / capacity[edge]

    # 各フロー毎の最大負荷率を計算
    flow_max_ratios = []  # 各フローの最大負荷率を記録

    for n,path in enumerate(select_flow,start=0):
        max_ratio = 0
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            # 全体負荷率から現在の辺の負荷率を取得
            current_ratio = edge_load_ratios[edge]
            max_ratio = max(max_ratio, current_ratio)
        flow_max_ratios.append(max_ratio)

    
    
    
    
    
    
    
    
    # 結果を表示
    # print("すべての経路:", all_paths)
    print(f'品種：{All_commodity_list}') #全ての品種
    print("経路数:", number_of_paths)  
    print("選択した経路")
    for i, flow in enumerate(select_flow):
        print(f"品種 {i + 1}:  {flow}")    
    print("最大負荷率の辺:", max_edge)
    print("最大負荷率:", max_load_ratio)
    print("辺ごとの負荷率:")
    for edge, ratio in edge_load_ratios.items():
        print(f" {edge}: 負荷率 {ratio}")

    print("\n各品種の最大負荷率:")
    for i, ratio in enumerate(flow_max_ratios):
        print(f"品種 {i + 1}: 最大負荷率 {ratio}")





#     # #実行結果の書き込み
#     # with open('mip.csv', 'a', newline='') as f:
#     #     out = csv.writer(f)
#     #     if a == 1:
#     #         out.writerow(['Objective','time','graph'])
#     #     out.writerow([UELB_kakai.objective_value, elapsed_time_kakai, graph_model])


#     print(a, kurikaesi)
#     # # print(area_k)

    kurikaesi += 1
    
    
    
    
    
# # pos = nx.circular_layout(g)
pos = nx.kamada_kawai_layout(g)
nx.draw(g, pos, with_labels=True)
plt.show()









# # 切除平面可視化（微妙）
# # fig, ax = plt.subplots()
# # t = np.linspace(-0.01, 0.01, 10)
# # y = []
# # for i in range(len(lagran_list)):
# #     y.append(lagran_list[i] + Load_factor_list[i] * (t - multiplier_list[i]))

# # for i in range(len(lagran_list)):
# #     ax.plot(t,y[i], label = i)
# # ax.legend(loc=0)    # 凡例
# # fig.tight_layout()  # レイアウトの設定
# # plt.show()