#厳密解を求めるためのプログラムファイル
from mip import *
import time
import csv
import networkx as nx
import matplotlib.pyplot as plt
from flow import Flow
import pulp
# from pyscipopt import Model as SCIPModel, quicksum

class Solve_exact_solution():
    def __init__(self, episode, solver_type, exact_file_name,limittime):
        self.episode = episode
        self.solver_type = solver_type
        self.exact_file_name = exact_file_name
        self.limittime = limittime
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
        
    def solve_exact_solution_to_env(self):
        if (self.solver_type == 'mip'): # mip+CBC
            # 問題の定義(全体の下界)
            UELB_kakai = Model('UELB_kakai') # モデルの名前

            L_kakai = UELB_kakai.add_var('L_kakai',lb = 0, ub = 1)
            flow_var_kakai = []
            for l in range(len(self.G.all_flows)):
                x_kakai = [UELB_kakai.add_var('x{}_{}'.format(l,m), var_type = BINARY) for m, (i,j) in self.r_kakai]#enumerate関数のmとエッジ(i,j)
                flow_var_kakai.append(x_kakai) #品種エル(l)に対して全ての辺のIDを表す番号mがついている。中身は0-1

            UELB_kakai.objective = minimize(L_kakai) # 目的関数

            UELB_kakai += (-L_kakai) >= -1 # 負荷率1以下
            
            # print("容量制限")
            for e in range(len(self.G.edges())): #容量制限
                UELB_kakai += 0 <= L_kakai - ((xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for l in self.G.all_flows])) / self.capacity[self.r_kakai[e][1]])
            # print("フロー保存則1")
            for l in self.G.all_flows: #フロー保存則
                UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == l.get_s()]) == l.get_demand()
                UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == l.get_s()]) == 0
                UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == l.get_t()]) == 0
                UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == l.get_t()]) == l.get_demand()
            # print("フロー保存則2")
            for l in self.G.all_flows: #フロー保存則
                for v in self.G.nodes():
                    if(v != l.get_s() and v != l.get_t()):
                        UELB_kakai += xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == v])\
                        ==xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == v])
            

            print("start optimize")
            #線形計画問題を解く
            start = time.time()
            UELB_kakai.optimize(max_seconds=self.limittime)
            elapsed_time = time.time()-start

            with open(self.exact_file_name, 'a', newline='') as f:
                out = csv.writer(f)
                out.writerow([self.episode, UELB_kakai.objective_value, elapsed_time]) 
            return UELB_kakai.objective_value,elapsed_time
        
        if (self.solver_type == 'pulp'): # pulp+CBC
            UELB_problem = pulp.LpProblem('UELB', pulp.LpMinimize) # モデルの名前
            L = pulp.LpVariable('L', 0, 1, 'Continuous') # 最大負荷率　Continuous：連続値 Integer:整数値 Binary:２値変数
            flow_var_kakai = []
            for l in range(len(self.G.all_flows)): # 0,1変数
                e_01 = [pulp.LpVariable('x{}_{}'.format(l,m), cat=pulp.LpBinary) for m, (i,j) in self.r_kakai]#enumerate関数のmとエッジ(i,j)
                flow_var_kakai.append(e_01) #品種エル(l)に対して全ての辺のIDを表す番号mがついている。中身は0-1

            UELB_problem += ( L , "Objective" ) # 目的関数値
            
            UELB_problem += (-L) >= -1 # 負荷率1以下

            # print("容量制限")
            for e in range(len(self.G.edges())): # 容量制限
                UELB_problem += 0 <= L - ((sum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for l in self.G.all_flows])) / self.capacity[self.r_kakai[e][1]])

            # print("フロー保存則1")
            for l in self.G.all_flows: #フロー保存則
                UELB_problem += sum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == l.get_s()]) == l.get_demand()
                UELB_problem += sum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == l.get_s()]) == 0
                UELB_problem += sum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == l.get_t()]) == 0
                UELB_problem += sum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == l.get_t()]) == l.get_demand()
            
            # print("フロー保存則2")
            for l in self.G.all_flows: #フロー保存則
                for v in self.G.nodes():
                    if(v != l.get_s() and v != l.get_t()):
                        UELB_problem += sum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == v])\
                        ==sum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == v])

            solver = pulp.PULP_CBC_CMD(timeLimit=self.limittime,msg=False)            
            print("start optimize")
            start = time.time()
            status = UELB_problem.solve(solver) # 線形計画問題を解く
            elapsed_time = time.time()-start

            # print(status)
            # print(UELB_problem) # 制約式を全て出してくれる 
            
            if UELB_problem.status == pulp.LpStatusOptimal:
                print("optimalsolution found")
            else:
                print("not found")
            
            with open(self.exact_file_name, 'a', newline='') as f:
                out = csv.writer(f)
                out.writerow([self.episode, L.value(),elapsed_time]) 
            return L.value(),elapsed_time

        if (self.solver_type == 'SCIP'): # PySCIPOpt+SCIP

            # SCIP Modelの作成
            model = SCIPModel("UELB_problem_SCIP")

            # 変数の定義
            L = model.addVar(vtype="C", name="L", lb=0, ub=1)
    
            flow_var_kakai = []
            for l in range(len(self.G.all_flows)):
                e_01 = [model.addVar('x{}_{}'.format(l, m), vtype='B') for m, (i, j) in enumerate(self.r_kakai)]
                flow_var_kakai.append(e_01)

            model.setObjective(L, "minimize")

            model.addCons((L) <= 1)# 負荷率1以下

            for e in range(len(self.G.edges())): # 容量制限
                model.addCons(L - (quicksum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for l in self.G.all_flows]) / self.capacity[self.r_kakai[e][1]]) >= 0)

            for l in self.G.all_flows: #フロー保存則
                model.addCons( quicksum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == l.get_s()]) == l.get_demand() )
                model.addCons( quicksum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == l.get_s()]) == 0 )
                model.addCons( quicksum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == l.get_t()]) == 0 )
                model.addCons( quicksum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == l.get_t()]) == l.get_demand() )
            
            for l in self.G.all_flows: #フロー保存則
                for v in self.G.nodes():
                    if(v != l.get_s() and v != l.get_t()):
                        model.addCons( quicksum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == v])\
                        ==quicksum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == v]) )

            # print("start optimize")
            model.setParam("limits/time",self.limittime)
            start = time.time()
            model.optimize()
            elapsed_time = time.time()-start
            
            with open(self.exact_file_name, 'a', newline='') as f:
                out = csv.writer(f)
                out.writerow([self.episode, model.getObjVal(),elapsed_time]) 
            # nx.draw(self.G, with_labels=True)
            # plt.show()
            return model.getObjVal(),elapsed_time