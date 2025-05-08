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
retu = 9
d = retu*retu
node = 10
graph_model = "random"
seed_value = 40
random.seed(seed_value) #ランダムの固定化
elapsed_time_kakai = 0

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
        g.randomGraph(g, n=node, k=5, seed=seed_value, number_of_area =1, number_of_areanodes = node, area_height=retu)
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
                demand = random.randint(5, 15)

                # demand = 10

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

                commodity_count +=1
                area_commodity_count += 1
        
        g.all_flow_dict[area] = flows_list
        ####print(g.get_i_area_flows(area))

    Load_factor = 0
    multiplier = 1 #int (input('ラグランジュ乗数(σ)>>'))
    edge_load_maxmin = float('inf')
    lagran_list = [] #ラグランジュ緩和した問題の解を格納する
    multiplier_list = []
    multiplier_list.append(multiplier)
    Load_factor_list = []
    multiplier_count = {} # 同じmultiplierの出現回数を格納する辞書
    
    #-------------------------------------
    #グラフ全体の目的関数値の下界を出す
    #-------------------------------------
        #処理時間の計測
    if __name__ == '__main__':
        start_kakai = time.time()
    
    for w in range(100):
        #問題の定義(全体の下界)
        UELB_kakai = Model('UELB_kakai')#モデルの名前

        L_kakai = UELB_kakai.add_var('L_kakai')
        flow_var_kakai = []
        for l in range(len(g.all_flows)):
            x_kakai = [UELB_kakai.add_var('x{}_{}'.format(l,m), var_type = BINARY) for m, (i,j) in r_kakai]#1つの品種に対して[全ての辺に対して0か1か]
            flow_var_kakai.append(x_kakai) #品種エル(l)に対して全ての辺のIDを表す番号mがついている。中身は0-1 [[品種1:全ての辺で0か1][品種2:0-1]....]

        #容量制限
        capacity = nx.get_edge_attributes(g,'capacity')#全辺のcapacityの値を辞書で取得。エッジ(node,node)をキーとしてcapacityの値を取得できる。 {(0, 1): 580, (0, 9): 860, (0, 2): 980,....}
        for e in range(len(g.edges())):
            Load_factor = Load_factor + ((xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for l in g.all_flows]))/capacity[r_kakai[e][1]]) #目的関数に追加した、σ以外の部分


        #目的関数
        UELB_kakai.objective = minimize(L_kakai + multiplier*(Load_factor - L_kakai * len(g.edges())))

        #制約式
        #1-負荷率は1を超えない
        UELB_kakai += (-L_kakai)>=-1

        #ソースから出るフロー,シンクに入るフローはDemands[(S[l],T[l])]
        for l in g.all_flows:
            #(flow_var_kakai[l.get_id()][e]) = 0か1
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][0] == l.get_update_s()]) == l.get_demand()
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][1] == l.get_update_s()]) == 0
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][0] == l.get_update_t()]) == 0
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][1] == l.get_update_t()]) == l.get_demand()

            #フロー保存則
        for l in g.all_flows:
            for v in g.nodes():
                if(v != l.get_update_s() and v != l.get_update_t()):
                    UELB_kakai += xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][0] == v])\
                    == xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][1] == v])

        #線形計画問題を解く
        UELB_kakai.optimize() 
        
        # 双対問題の定義
        UELB_dual = Model('UELB_dual')
        
        z = UELB_dual.add_var('z', lb = 0) # 目的関数値となる変数
        siguma = UELB_dual.add_var('siguma', lb = 0, ub = 1) #上限2は適当
        lagran_list.append(UELB_kakai.objective_value)
        Load_factor_list.append(Load_factor.x - (L_kakai.x * len(g.edges())))
        
        print("list")
        print(lagran_list)
        print(Load_factor_list)
        print(multiplier_list)
        
        Load_factor = 0 # 初期化
        
        #目的関数
        UELB_dual.objective = maximize(z)
        #制約式
        for x in range(len(lagran_list)):
            UELB_dual += lagran_list[x] + Load_factor_list[x] * (siguma - multiplier_list[x]) >= z
        # z <= UELB_kakai.objective_value + Load_factor*(siguma - multiplier)
        
        # 線形計画問題を解く
        UELB_dual.optimize()
        
        max_solution = UELB_dual.objective_value
        print(f'zの値：{max_solution}')
        # search = []
        
        # for i in range(len(lagran_list)):
        #     i = len(lagran_list) - i - 1 #最新のものから
        #     siguma_sab = (max_solution - lagran_list[i] + Load_factor_list[i] * multiplier_list[i]) / Load_factor_list[i]
        #     if siguma_sab in search:
        #         multiplier = siguma_sab
        #         # print("multiplier")
        #         # print(multiplier)
        #         break
        #     else:
        #         search.append(siguma_sab)
        # multiplier_list.append(multiplier)
        # print("その他σ")
        # print(search)
        
        multiplier = siguma.x
        multiplier_list.append(multiplier)
        
        # 最大値zには二つの接線が関わっているため、全ての接線に最大値zの値を代入したときのsigumaの値は重複するものがあり、それがzが最大の時のsigumaの値
        # そのsigumaの値を次のmultiplierにする
        # 新しく追加された接線の方が関わっている可能性高そうなので、そちらから探索した方が計算時間速くなると思ったためこのようにした
        
        # --終了条件--
        # if w > 1:
        #     if Load_factor_list[w-2] == Load_factor_list[w-1] == Load_factor_list[w] :
        #         break

        # ---σの値が3回一緒なものが出てきたら終了させる---
        # if multiplier in multiplier_count:
        #     multiplier_count[multiplier] += 1
        #     if multiplier_count[multiplier] == 3:
        #         break
        # else:
        #     multiplier_count[multiplier] = 1

        # ---同じ近似解が3回出てきたら終了---
        if Load_factor_list[w] in multiplier_count:
            multiplier_count[Load_factor_list[w]] += 1
            if multiplier_count[Load_factor_list[w]] == 3:
                break
        else:
            multiplier_count[Load_factor_list[w]] = 1
    
    # 最大の目的関数値をとるmultiplierの値を求めたので、最後にそのmultiplierの値を使って緩和問題を解く。その解が、緩和する前の問題の最適解に一番近い解となる。
    # multiplier = 0.001
    Load_factor2  = 0
    #問題の定義(全体の下界)
    UELB_kakai2 = Model('UELB_kakai2')#モデルの名前

    L_kakai2 = UELB_kakai2.add_var('L_kakai2')
    flow_var_kakai2 = []
    for l in range(len(g.all_flows)):
        x_kakai2 = [UELB_kakai2.add_var('x{}_{}'.format(l,m), var_type = BINARY) for m, (i,j) in r_kakai]#enumerate関数のmとエッジ(i,j) ソルバーを開始すると、経路を見つけるために毎回これを参照し、辺の0-1を換えながら試している？
        flow_var_kakai2.append(x_kakai2)

    #容量制限
    capacity = nx.get_edge_attributes(g,'capacity')#全辺のcapacityの値を辞書で取得。エッジ(node,node)をキーとしてcapacityの値を取得できる。 {(0, 1): 580, (0, 9): 860, (0, 2): 980,....}
    # for e in range(len(g.edges())):
    #     Load_factor = sum((((xsum([(flow_var_kakai2[l.get_id()][e])*(l.get_demand()) for l in g.all_flows]))/capacity[r_kakai[e][1]]) - L_kakai)) #目的関数に追加した、σ以外の部分
    
    # #目的関数
    # UELB_kakai.objective = minimize(L_kakai + multiplier*Load_factor)
    for e in range(len(g.edges())):
        Load_factor2 = Load_factor2 + (((xsum([(flow_var_kakai2[l.get_id()][e])*(l.get_demand()) for l in g.all_flows]))/capacity[r_kakai[e][1]])) #目的関数に追加した、σ以外の部分


    #目的関数
    UELB_kakai2.objective = minimize(L_kakai2 + multiplier * (Load_factor2 - L_kakai2 * len(g.edges())))

    #制約式
    #1-負荷率は1を超えない
    UELB_kakai2 += (-L_kakai2)>=-1
    
    #ソースから出るフロー,シンクに入るフローはDemands[(S[l],T[l])]
    for l in g.all_flows:
        #(flow_var_kakai[l.get_id()][e]) = 0か1
        UELB_kakai2 += xsum([flow_var_kakai2[l.get_id()][e]*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][0] == l.get_update_s()]) == l.get_demand()
        UELB_kakai2 += xsum([flow_var_kakai2[l.get_id()][e]*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][1] == l.get_update_s()]) == 0
        UELB_kakai2 += xsum([flow_var_kakai2[l.get_id()][e]*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][0] == l.get_update_t()]) == 0
        UELB_kakai2 += xsum([flow_var_kakai2[l.get_id()][e]*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][1] == l.get_update_t()]) == l.get_demand()
    
    #フロー保存則
    for l in g.all_flows:
        for v in g.nodes():
            if(v != l.get_update_s() and v != l.get_update_t()):
                UELB_kakai2 += xsum([(flow_var_kakai2[l.get_id()][e])*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][0] == v])\
                ==xsum([(flow_var_kakai2[l.get_id()][e])*(l.get_demand()) for e in range(len(g.edges())) if r_kakai[e][1][1] == v])

    UELB_kakai2.optimize()
    
    #終了時間の計測
    elapsed_time_kakai = time.time()-start_kakai
    status = UELB_kakai2.status #実行結果のステータスコード

    result_flow_var_kakai = []
    de = [] #各品種の需要量
    for l in g.all_flows:
        de.append(l.get_demand())

    for l in range(len(g.all_flows)):
        result_x = []
        for m in range(len(r_kakai)):    
            result_x.append(flow_var_kakai2[l][m].x)# 各辺の0か1かのリスト。双方向グラフなのでlen=54。需要があるとこだけ1でそれ以外は0
        result_flow_var_kakai.append(result_x)# 品種ごとの各辺の0か1かのリスト
    # print("iiiiiiiiii")
    # print(result_x)
    # print(result_flow_var_kakai)

    print(result_flow_var_kakai)
    
    edge_demand = []
    edge_load = []
    for e in range(len(g.edges())):
        ed = 0
        for l in range(len(g.all_flows)):
            ed += de[l] * result_flow_var_kakai[l][e]# 各辺のフロー量の総和
        edge_demand.append(ed)
        edge_load.append(ed/capacity[r_kakai[e][1]])# 各辺の負荷率
    print(z for z in edge_load)    
    edge_load_max = max(edge_load)

    if(edge_load_max > 1):
        print('overflow!!!')

    if(edge_load_max < edge_load_maxmin):
        edge_load_maxmin = edge_load_max

    all_path = []
    for l in range(len(g.all_flows)):
        commodity_path = []
        for m,(i,j) in r_kakai:
            if(result_flow_var_kakai[l][m]==1):
                commodity_path.append((i,j))# 品種ごとに、辺に対して割り当てられた変数が1の辺を追加
        all_path.append(commodity_path)# 品種ごとのパスを出力    


    # print(f'目的関数値：{value(UELB_kakai.objective)}')
    print(f'Lの値：{L_kakai2.x}')
    print(f'品種：{All_commodity_list}') #全ての品種
    # print(f'品種：{tuples}') #全ての品種
    print("品種ごとのパス：{}".format(all_path))
    # print(result_x)
    # print(result_flow_var_kakai)
    print(f'最大目的関数値：{UELB_kakai2.objective_value}')
    print("最大負荷率：{}".format(edge_load_max))# 最大負荷率
    print(f'最強ラグランジュ乗数：{multiplier}')
    print('time : ',elapsed_time_kakai)
    
    print('--------------------------------------------')
    


    # #実行結果の書き込み
    # with open('mip.csv', 'a', newline='') as f:
    #     out = csv.writer(f)
    #     if a == 1:
    #         out.writerow(['Objective','time','graph'])
    #     out.writerow([UELB_kakai.objective_value, elapsed_time_kakai, graph_model])


    print(a, kurikaesi)
    # # print(area_k)

    kurikaesi += 1
# pos = nx.circular_layout(g)
pos = nx.kamada_kawai_layout(g)
nx.draw(g, pos, with_labels=True)
plt.show()

# 切除平面可視化（微妙）
# fig, ax = plt.subplots()
# t = np.linspace(-0.01, 0.01, 10)
# y = []
# for i in range(len(lagran_list)):
#     y.append(lagran_list[i] + Load_factor_list[i] * (t - multiplier_list[i]))

# for i in range(len(lagran_list)):
#     ax.plot(t,y[i], label = i)
# ax.legend(loc=0)    # 凡例
# fig.tight_layout()  # レイアウトの設定
# plt.show()