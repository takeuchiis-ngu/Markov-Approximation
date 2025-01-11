#単純kspを求めるためのプログラムファイル
from mip import *
import networkx as nx
import matplotlib.pyplot as plt
import time
import csv
from flow import Flow
import copy
from itertools import product
import pulp
# from pyscipopt import Model as SCIPModel, quicksum
import random
import numpy as np


class Solve_simple_ksp():
    def __init__(self, episode, solver_type, simpleksp_file_name, K, objective_value):
        self.episode = episode
        self.solver_type = solver_type
        self.simpleksp_file_name = simpleksp_file_name
        self.K = K #パスの個数
        self.elapsed_time = 0
        self.objective_value = objective_value
        self.G = nx.read_gml("./value/graph.gml",destringizer=int) # グラフの定義
        self.G.all_flows = list()

        with open('./value/commodity_data.csv',newline='') as f: # 品種の読み込み
            self.commodity=csv.reader(f)
            self.commodity=[row for row in self.commodity]
            self.commodity=[[int(item) for item in row]for row in self.commodity]

        self.r_kakai = list(enumerate(self.G.edges()))
        self.commodity_count = 0
        self.tuples = []
        self.capacity = nx.get_edge_attributes(self.G, 'capacity')

        while(len(self.tuples)<len(self.commodity)): 
            s = self.commodity[self.commodity_count][0] # source
            t = self.commodity[self.commodity_count][1] # sink
            demand = self.commodity[self.commodity_count][2] # demand
            self.tuples.append((s,t))
            f = Flow(self.G,self.commodity_count,s,t,demand)
            self.G.all_flows.append(f)
            self.commodity_count +=1

    def search_ksps(self): # 各品種のkspsを求める
        print("start search ksp")
        allcommodity_ksps = []
        start = time.time()
        for i in range(self.commodity_count):
            X = nx.shortest_simple_paths(self.G, self.commodity[i][0], self.commodity[i][1]) # Yen's algorithm
            ksps_list = []
            for counter, path in enumerate(X):            
                ksps_list.append(path)
                if counter == self.K - 1: 
                    break
            allcommodity_ksps.append(ksps_list)
        self.elapsed_time = self.elapsed_time + (time.time()-start)
        return allcommodity_ksps
  
    def searh_combination(self, allcommodity_ksps): # 組み合わせを求める
        print("start search combination")
        combination = []
        zerone_combination = []
        rr_kakai = []
        start = time.time()
        for i in range(len(self.r_kakai)):
            rr_kakai.append(self.r_kakai[i][1])

        q = [*product(*allcommodity_ksps)] 
        for conbi in q:
            combination.append(conbi)
        # print("conbination",conbination)
        # print(len(conbination))

        for c in range(len(combination)): # 経路をもとに辺の01変換処理
            flow_var_kakai = []
            for l in self.G.all_flows:
                x_kakai = len(self.G.edges())*[0]
                for a in range(len(combination[c][l.get_id()])):
                    if a == len(combination[c][l.get_id()])-1:
                        break
                    set = (combination[c][l.get_id()][a],combination[c][l.get_id()][a+1])
                    idx = rr_kakai.index(set)
                    # print(set)
                    x_kakai[idx] = 1
                flow_var_kakai.append(x_kakai)
            zerone_combination.append(flow_var_kakai)
        self.elapsed_time = self.elapsed_time + (time.time()-start)
        return zerone_combination

    def solve_simple_ksp_to_env(self):
        allcommodity_ksps = self.search_ksps()
        combination = self.searh_combination(allcommodity_ksps)
        Maxload = np.empty(0)

        if (self.solver_type == 'mip'): # mip+CBC
            for c in range(len(combination)):
                # load=[]
                #### 問題の定義 ####
                Kshortestpath = Model('Kshortestpath') #モデルの名前

                #### 変数Lの生成 ####
                L = Kshortestpath.add_var('L',lb = 0, ub = 1)

                #### 目的関数 ####
                Kshortestpath.objective = minimize(L)

                #### 制約式 ####
                Kshortestpath += L <= 1 #負荷率1以下
                for e in range(len(self.G.edges())): #容量制限
                    Kshortestpath += 0 <= L - ((xsum([(combination[c][l.get_id()][e])*(l.get_demand()) for l in self.G.all_flows])) / self.capacity[self.r_kakai[e][1]])
                    # load.append(sum((combination[c][l.get_id()][e])*(l.get_demand())for l in self.G.all_flows)/self.capacity[self.r_kakai[e][1]])

                print("start optimize")
                #線形計画問題を解く
                start = time.time()
                Kshortestpath.optimize() #L.x == Kshortestpath.objective_value
                self.elapsed_time = self.elapsed_time + (time.time()-start)
                Maxload = np.append(Maxload,Kshortestpath.objective_value)
                
                if self.objective_value != None :
                    rate = float(self.objective_value) / float(Kshortestpath.objective_value)
                    if rate == 1.0: #近似率100％の近似解計算終了
                        break      
            min_maxload = Maxload.min()
        
        if (self.solver_type == 'pulp'): # pulp+CBC
            for c in range(len(combination)):
                # load=[]
                Kshortestpath = pulp.LpProblem('Kshortestpath',pulp.LpMinimize) #モデルの名前
                L = pulp.LpVariable('L', 0, 1, 'Continuous')
                Kshortestpath += ( L , "Objective" ) # 目的関数値
                Kshortestpath += L <= 1 #負荷率1以下

                for e in range(len(self.G.edges())): #容量制限
                    Kshortestpath += 0 <= L - ((sum([(combination[c][l.get_id()][e])*(l.get_demand()) for l in self.G.all_flows])) / self.capacity[self.r_kakai[e][1]])
                    # load.append(sum((combination[c][l.get_id()][e])*(l.get_demand())for l in self.G.all_flows)/self.capacity[self.r_kakai[e][1]])

                print("start optimize")
                #線形計画問題を解く
                start = time.time()
                status = Kshortestpath.solve() # 線形計画問題を解く                elapsed_time = elapsed_time + time.time()-start
                self.elapsed_time = self.elapsed_time + (time.time()-start)
                Maxload = np.append(Maxload,L.value())
                
                if self.objective_value != None :
                    rate = float(self.objective_value) / float(L.value())
                    if rate == 1.0: #近似率100％の近似解計算終了
                        break      
            min_maxload = Maxload.min()

        if (self.solver_type == 'SCIP'): # PySCIPOpt+SCIP
            for c in range(len(combination)):
                model = SCIPModel("Kshortestpath")
                L = model.addVar(vtype="C", name="L", lb=0, ub=1)
                model.setObjective(L, "minimize")
                
                model.addCons((L) <= 1)# 負荷率1以下
                for e in range(len(self.G.edges())): # 容量制限
                    model.addCons(L - (quicksum([(combination[c][l.get_id()][e])*(l.get_demand()) for l in self.G.all_flows]) / self.capacity[self.r_kakai[e][1]]) >= 0)

                print("start optimize")
                #線形計画問題を解く
                start = time.time()
                model.optimize()
                self.elapsed_time = self.elapsed_time + (time.time()-start)
                Maxload = np.append(Maxload,model.getObjVal())

                if self.objective_value != None :
                    rate = float(self.objective_value) / float(model.getObjVal())
                    # rate = float(self.objective_values) / Maxload.min()
                    if rate == 1.0: #近似率100％の近似解計算終了
                        break      
            min_maxload = Maxload.min()

        with open(self.simpleksp_file_name, 'a', newline='') as f:
            out = csv.writer(f)
            out.writerow([self.episode, min_maxload, self.elapsed_time]) 
        return min_maxload,self.elapsed_time