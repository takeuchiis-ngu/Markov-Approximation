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

# すべてのパスの組合せの中から，負荷率が最小になる組合せを見つける関数
# O(|Pl|^k)になるので，指数時間になる

kurikaesi = 0
a = 10
b = a #int (input('area品種数>>'))
c = 1 #int (input('エリア数>>'))
#d = 16 #int (input('エリアのノード数>>'))
retu = 3
d = retu*retu
node = 10
graph_model = "random"
seed_value = 42
random.seed(seed_value) #ランダムの固定化
elapsed_time_kakai = 0

# logの削除
# if isinstance(pulp.LpSolverDefault, pulp.PULP_CBC_CMD):
#     pulp.LpSolverDefault.msg = False



while(kurikaesi < 1):

    if(graph_model == 'grid'):
        g = graph_making.Graphs(a,b) #品種数,areaの品種数
        g.gridMaker(g,1,d,retu,retu,0.1) #G,エリア数,エリアのノード数,列数,行数,ε浮動小数点

    if(graph_model == 'random'):
        g = graph_making.Graphs(a,b) #品種数,areaの品種数
        g.randomGraph(g, n=node, k=5, seed=seed_value, number_of_area =1, number_of_areanodes = node, area_height=retu)
        # nx.draw(g)
        # plt.show()

    r_kakai = list(enumerate(g.edges())) #グラフのエッジと番号を対応付けたもの
    # print(r_kakai)
    edge_dic_kakai = {(i,j):m for m,(i,j) in r_kakai} #{(0, 1): 0, (0, 9): 1, (0, 2): 2,.....}
    num_dic_kakai = {m:(i,j) for m,(i,j) in r_kakai} #{0: (0, 1), 1: (0, 9), 2: (0, 2),.....}
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
    
    demand_list = []

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

        for _ in range(a):
            while True: 
                if(graph_model == 'grid'):
                    s = random.randrange(area_nodes_list[0],area_nodes_list[g.number_of_areanodes-1]+1,1)
                    t = random.randrange(area_nodes_list[0],area_nodes_list[g.number_of_areanodes-1]+1,1)
                    demand  = random.randrange(1, 41, 1)
                    # demand = 50
                
                if(graph_model == 'random'):#ノードの中から始点と終点を1つずつランダムで選ぶ
                    s, t = random.sample(area_nodes_list, 2)  # s と t を異なるノードとして選択
                    # s = random.randrange(area_nodes_list[0]-1,area_nodes_list[g.number_of_areanodes-1],1)
                    # t = random.randrange(area_nodes_list[0]-1,area_nodes_list[g.number_of_areanodes-1],1)
                    # if len(tuples) == 0 or len(tuples) == 2:
                    #     s = 0
                    #     t = 2
                    # elif len(tuples) == 1:
                    #     s = 0
                    #     t = 2
                    # else:
                    #     s = 3
                    #     t = 2
                    demand  = random.randrange(5, 15)
                    # demand = 10

                if(s!=t and ((s,t) not in All_commodity_list)): #流し始めのノードからそのノードに戻ってくる組み合わせは考えていない 
                    break
                
            tuples.append((s,t))
            All_commodity_list.append((s,t))
            demand_list.append(demand)
            # print((s,t))
            # print(All_commodity_list)


            f = Flow(g,commodity_count,s,t,demand)
            f.set_area_flow_id(area_commodity_count)
            flows_list.append(f)
            g.all_flows.append(f)
            # print(g.all_flows)
            ####print(f.get_update_s_area())

            commodity_count +=1
            area_commodity_count += 1

        g.all_flow_dict[area] = flows_list
        ####print(g.get_i_area_flows(area))


    #-------------------------------------
    #グラフ全体の目的関数値の下界を出す
    #-------------------------------------

    #問題の定義(全体の下界)
    UELB_kakai = Model('UELB_kakai')#モデルの名前

    L_kakai = UELB_kakai.add_var('L_kakai',lb = 0, ub = 1)
    flow_var_kakai = []
    for l in range(len(g.all_flows)):
        x_kakai = [UELB_kakai.add_var('x{}_{}'.format(l,m), var_type = BINARY) for m, (i,j) in r_kakai]#enumerate関数のmとエッジ(i,j)
        flow_var_kakai.append(x_kakai) #品種エル(l)に対して全ての辺のIDを表す番号mがついている。中身は0-1

    #目的関数
    UELB_kakai.objective = minimize(L_kakai)

    UELB_kakai += (-L_kakai) >= -1 #負荷率1以下
    capacity = nx.get_edge_attributes(g,'capacity')#全辺のcapacityの値を辞書で取得
    for e in range(len(g.edges())): #容量制限
        UELB_kakai += 0 <= L_kakai - ((xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for l in g.all_flows])) / capacity[r_kakai[e][1]])
    for l in g.all_flows: #フロー保存則
        UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][0] == l.get_update_s()]) == l.get_demand()
        UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][1] == l.get_update_s()]) == 0
        UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][0] == l.get_update_t()]) == 0
        UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][1] == l.get_update_t()]) == l.get_demand()
    for l in g.all_flows: #フロー保存則
        for v in g.nodes():
            if(v != l.get_update_s() and v != l.get_update_t()):
                UELB_kakai += xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][0] == v])\
                ==xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][1] == v])
    # #制約式
    # #1-負荷率は1を超えない
    # UELB_kakai += (-L_kakai)>=-1

    # #容量制限
    # capacity = nx.get_edge_attributes(g,'capacity')#全辺のcapacityの値を辞書で取得
    # for e in range(len(g.edges())):
    #     UELB_kakai += 0 <= L_kakai-((xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for l in g.all_flows])+xsum(g.edges[r_kakai[e][1]]['flow_demand_init']))/capacity[r_kakai[e][1]])
    #     # print(capacity[r[e][1]])

    # #ソースから出るフロー,シンクに入るフローはDemands[(S[l],T[l])]
    # for l in g.all_flows:
    #     UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][0] == l.get_update_s()]) == l.get_demand()
    #     UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][1] == l.get_update_s()]) == 0
    #     UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][0] == l.get_update_t()]) == 0
    #     UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][1] == l.get_update_t()]) == l.get_demand()

    # #フロー保存則
    # for l in g.all_flows:
    #     for v in g.nodes():
    #         if(v != l.get_update_s() and v != l.get_update_t()):
    #             UELB_kakai += xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][0] == v])\
    #             ==xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][1] == v])

    # # UELB_kakai.writeLP("lp_relaxation_grid33.lp")

    #処理時間の計測
    if __name__ == '__main__':
        start_kakai = time.time()

    #線形計画問題を解く
    UELB_kakai.optimize()

    #終了時間の計測
    elapsed_time_kakai = time.time()-start_kakai
    status = UELB_kakai.status

    result_flow_var_kakai = []
    for l in range(len(g.all_flows)):
        result_x = []
        for m in range(len(r_kakai)):    
            result_x.append(flow_var_kakai[l][m].x)# 各辺のリスト
        result_flow_var_kakai.append(result_x)# 品種ごとの各辺のリスト
    
    all_path = []
    for l in range(len(g.all_flows)):
        commodity_path = []
        for m,(i,j) in r_kakai:
            if(result_flow_var_kakai[l][m]==1):
                commodity_path.append((i,j))# 品種ごとに割り当て変数が1の辺を追加
        all_path.append(commodity_path)# 品種ごとのパスを出力
        

    # print(f'目的関数値：{value(UELB_kakai.objective)}')
    print('Objective :', UELB_kakai.objective_value) #最小の最大負荷率
    print(All_commodity_list) #全ての品種
    print("需要量：",demand_list)
    # print(result_x)
    print(result_flow_var_kakai)
    print(all_path)
    print('time : ',elapsed_time_kakai)
    print('--------------------------------------------')
    


    #実行結果の書き込み
    with open('mip.csv', 'a', newline='') as f:
        out = csv.writer(f)
        if a == 1:
            out.writerow(['Objective','time','graph'])
        out.writerow([UELB_kakai.objective_value, elapsed_time_kakai, graph_model])


    print(a, kurikaesi)
    # # print(area_k)
    
    nx.draw(g, with_labels=True)
    plt.show()

    kurikaesi += 1