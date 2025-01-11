#mainのプログラムファイル　最新

#state:経路の組み合わせ
#action:経路の組み替え
#observation:各辺の負荷率

import gym.spaces
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
import copy
import time
import csv
import numpy as np
import pandas as pd
import random
# from Exact_Solution import Solve_exact_solution
from deepexactsolution import Solve_exact_solution
from LP import Solve_LP_solution
from lowerlimit1 import Solve_lowerlimit1
# from lowerlimit2 import Solve_lowerlimit2
# from LR import Solve_LR_solution
from Simple_ksp import Solve_simple_ksp
from flow import Flow
import graph_making
import os
from keras.models import Sequential,Model
from keras.layers import Dense, Activation, Flatten,Input,InputLayer,Concatenate,Masking,LeakyReLU,Lambda
from keras.optimizers import Adam
import rl.callbacks
import datetime
import gym


class min_maxload_KSPs_Env(gym.core.Env): # クラスの定義
    def __init__(self, K, n_action, obs_low, obs_high, max_step, node_l, node_h, range_commodity_l, range_commodity_h, sample_size,capa_l,capa_h,demand_l,demand_h,graph_model, degree, initialstate, rewardstate, countlimit):
        self.K = K # kspsの本数
        self.node = random.randint(node_l, node_h)
        self.commodity = random.randint(range_commodity_l, range_commodity_h)
        self.sample_size = sample_size
        self.capa_l = capa_l
        self.capa_h = capa_h
        self.demand_l = demand_l
        self.demand_h = demand_h
        self.graph_model = graph_model
        self.degree = degree
        self.initialstate = initialstate
        self.rewardstate = rewardstate
        self.countlimit = countlimit

        self.n_action = n_action # 行動の数
        self.action_space = gym.spaces.Discrete(self.n_action) # actionの取りうる値 gym.spaces.Discrete(N):N個の離散値の空間
        #可能なactionの集合に対してactionにしたがって組み替えたときのコストを計算し、コストの小さい10通りのうち何番目を選ぶか
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, shape=(self.n_action,)) # 観測データの取りうる値
        # n_action通りの組み替えコストを観測データ
        self.time = 0 # ステップ
        self.max_step = max_step # ステップの最大数
        self.candidate_list = [] # 経路の組み替えの候補

    def render(self, mode): # 画面への描画・可視化
        pass
    def close(self): # 終了時の処理
        pass
    def seed(self): # 乱数の固定
        pass
    def check_is_done(self): # 終了条件を判定する関数

        # 最大ステップ数に達したら終了
        if self.time >= self.max_step:
            return True
    
        # 観測変数が全て-100.0なら終了
        if all(val == -100.0 for val in self.observation):
            return True

        # どの条件にも該当しなければ終了しない
        return False

    def generate_commodity(self, commodity): # 品種の定義(numpy使用)
        determin_st = []
        commodity_list = []
        for i in range(commodity): # commodity generate
            commodity_dict = {}
            s , t = tuple(random.sample(self.G.nodes, 2)) # source，sink定義
            demand = random.randint(self.demand_l, self.demand_h) # demand設定
            tentative_st = [s,t]
            while True:
                if tentative_st in determin_st:
                    s , t = tuple(random.sample(self.G.nodes, 2)) # source，sink再定義
                    tentative_st = [s,t]
                else:
                    break
            determin_st.append(tentative_st) # commodity決定
            commodity_dict["id"] = i
            commodity_dict["source"] = s
            commodity_dict["sink"] = t
            commodity_dict["demand"] = demand

            commodity_list.append([s,t,demand])
        commodity_list.sort(key=lambda x: -x[2]) # demand大きいものから降順

        return commodity_list

    def search_ksps(self, K, G, commodity,commodity_list): # 各品種のkspsを求める
        allcommodity_ksps = []
        random_ksps = []
        allcommodity_notfstksps = []
        for i in range(commodity):
            X = nx.shortest_simple_paths(G, commodity_list[i][0], commodity_list[i][1]) # Yen's algorithm
            ksps_list = []
            for counter, path in enumerate(X):
                ksps_list.append(path)
                if counter == K - 1:
                    break
            # 経路のサンプリング
            # random_sample = random.sample(ksps_list, self.sample_size)
            # random_ksps.append(random_sample)

            allcommodity_ksps.append(ksps_list)

            subksps_list = copy.deepcopy(ksps_list)
            subksps_list.pop(0)
            allcommodity_notfstksps.append(subksps_list)
        # return random_ksps,allcommodity_ksps,allcommodity_notfstksps
        return allcommodity_ksps,allcommodity_notfstksps

    def searh_combination(self, allcommodity_ksps): # 最短の組み合わせを求める
        comb = []
        L = len(allcommodity_ksps)
        for z in range(L):
            comb.append(allcommodity_ksps[z][0])
            random.choice(allcommodity_ksps[z]) #randomの数をそろえる応急処置

        combination = [comb]
        return combination

    def zero_one(self, grouping): # 経路をもとに辺の01変換処理
        zo_combination = []
        for l in range(self.commodity):
            x_kakai = len(self.edges_notindex)*[0]
            for a in range(len(grouping[l])):
                if a == len(grouping[l])-1:
                    break
                set = (grouping[l][a],grouping[l][a+1])
                # print(set)
                idx = self.edges_notindex.index(set)
                x_kakai[idx] = 1
            zo_combination.append(x_kakai)
        return zo_combination

    def get_pair_list(self): # 品種と経路のペアを列挙
        pair_list = []
        for c in range(self.commodity):
            for p in range(len(self.allcommodity_ksps[c])):
                path = self.allcommodity_ksps[c][p]
                pair_list.append([c, path])
        # print("pair_list",pair_list)
        return pair_list

    def searh_randomcombination(self, allcommodity_ksps): # ランダムで組み合わせを求める
        comb = []
        L = len(allcommodity_ksps)
        for z in range(L):
            comb.append(random.choice(allcommodity_ksps[z]))
        combination = [comb]
        # print(combination)
        return combination

    def LoadFactor(self, grouping): # 負荷率の計算
        loads = []
        zo_combination = self.zero_one(grouping)
        for e in range(len(self.edge_list)): #容量制限
            load = sum((zo_combination[l][e])*(self.commodity_list[l][2])for l in range(self.commodity)) / self.capacity_list[self.edge_list[e][1]]
            loads.append(load)
            # load.append(sum((zo_combination[l][e])*(self.commodity_list[l][2])for l in range(self.commodity)) / self.capacity_list[self.edge_list[e][1]])
        # print("load",load)
        return loads

    def MaxLoadFactor(self, grouping): # 最大負荷率を計算する関数
        # return sum([self.LoadFactor(grouping, group_id) for group_id in self.group_id_list])
        self.maxload = max(self.LoadFactor(grouping))
        return self.maxload

    def exchange_path_action(self, grouping, action): # actionの値に応じて経路を交換する
        new_grouping = grouping.copy()
        # if len(self.candidate_list) == 0:

        # print("self.candidate_list",self.candidate_list)
        # print(action)
        c, path, cost = self.candidate_list[action]
        new_grouping[c] = path
        # print(grouping)
        # print(new_grouping)
        return new_grouping

    def exchange_path_pair(self, grouping, c, path): # candidate_listを作成するために経路を交換させる
        new_grouping = grouping.copy() # grouping:[[1,2,3],[2,4]]
        new_grouping[c] = path
        return new_grouping

    ###### reward ######
    def get_reward_maxload(self, grouping): # self.rewardstate==1 : 最大負荷率*(-1)
        return -1 * self.MaxLoadFactor(grouping)

    def get_reward_difference(self, grouping, bfmaxloadfactor): # self.rewardstate==2 : 最大負荷率の変化*(-1)
        new_maxloadfactor = self.MaxLoadFactor(grouping)
        difference = new_maxloadfactor - bfmaxloadfactor
        return -1 * difference

    def get_difference(self, grouping): # 差:costを計算する関数
        new_maxloadfactor = self.MaxLoadFactor(grouping)
        difference = new_maxloadfactor - self.old_maxloadfactor
        return -1 * difference

    ###### observation ######
    def get_observation(self): # 観測データを計算する関数　経路の組み替えによる負荷率をみたいから、pairでのobservationを確認する
        grouping = self.grouping.copy()
        candidate_list = []
        self.old_maxloadfactor = self.MaxLoadFactor(grouping)
        # print("in obs",self.old_maxloadfactor)

        for pair in self.pair_list:
            c, path = pair # 品種と経路のペアリストから一つずつ取り出す
            if(self.rewardstate==1):
                cost = self.get_reward_maxload(self.exchange_path_pair(grouping, c, path)) # 今のstateから対象の品種cに対して経路をpathに変更して報酬を計算する
                if cost>=-1: # オーバーフローじゃない場合にエントリー
                    candidate_list.append([c, path, cost])

            elif(self.rewardstate==2):
                cost = self.get_difference(self.exchange_path_pair(grouping, c, path))
                candidate_list.append([c, path, cost])

        self.candidate_list = sorted(candidate_list, key=lambda x:-x[2])[0:self.n_action] # costの大きい順に並び替えて、self.n_action個取り出す 最大負荷率が小さい順
        mask = [cand[2] for cand in self.candidate_list] # cost(最大負荷率)をリスト化して返す　観測データを返したいから
        if len(mask)<self.n_action: # エントリー数が足りない場合
            i = self.n_action - len(mask) # 足りない数を取得 i
            for n in range(i):
                mask.append(-100.0) # i回観測変数のリストに-100.0を追加
        return mask

    def step(self, action): # 各ステップで実行される操作
        self.time += 1
        observation = self.get_observation() # 入れ替え後の状態についての観測データを得る
        oldmaxload = self.MaxLoadFactor(self.grouping) # 現状の最大負荷率
        # print("instep obs:",observation)
        if all(val == -100.0 for val in observation):
            done = True  # エピソード終了
            info = {}
            # 報酬(reward)の計算
            if(self.rewardstate == 1):
                self.reward = self.get_reward_maxload(self.grouping)
            elif(self.rewardstate == 2):
                self.reward = self.get_reward_difference(self.grouping,oldmaxload) 
            self.observation = self.get_observation()
        else:
            self.grouping = self.exchange_path_action(self.grouping, action).copy()
            # 報酬(reward)の計算
            if(self.rewardstate == 1):
                self.reward = self.get_reward_maxload(self.grouping) # 今のobservationの中の報酬が最大のものではなく、candidateからationの値で選ばれたもののcostが出てくる
            elif(self.rewardstate == 2):
                self.reward = self.get_reward_difference(self.grouping,oldmaxload)
                # print("instep self.reward:",self.reward)

            # 観測データ(observation)の計算
            self.observation = self.get_observation() # 入れ替え後の状態についての観測データを得る
            # step4: 終了時刻を満たしているかの判定
            done = self.check_is_done()
            info = {}
        maxload = self.MaxLoadFactor(self.grouping)
        return self.observation, self.reward, done, info, maxload

    def reset(self): # 変数の初期化。エピソードの初期化。doneがTrueになったときに呼び出される。
        self.time = 0 # ステップ数の初期化
        self.grouping = self.get_random_grouping() # 条件の初期化
        self.pair_list = self.get_pair_list() # アクションの範囲を求める
        self.observation = self.get_observation() # 初期状態の観測変数
        count=0
        while True: # 初期パターンを変更
            if all(val == -100.0 for val in self.observation):
                # break
                self.combination = self.searh_randomcombination(self.allcommodity_ksps)
                self.grouping = self.combination[0] #初期パターン設定
                count=count+1
                self.observation = self.get_observation()
                if count==self.countlimit:
                    break
            else:
                break

        return self.observation

    def get_random_grouping(self):
        self.node = random.randint(node_l, node_h) # ランダムでgridgraphの行列数を決定
        self.commodity = random.randint(range_commodity_l, range_commodity_h) # ランダムで品種数を決定

        if(graph_model == 'grid'):
            self.G = graph_making.Graphs(self.commodity) #品種数,areaの品種数　この時点では何もグラフが作られていない　インスタンスの作成？
            self.G.gridMaker(self.G,self.node*self.node,self.node,self.node,0.1,self.capa_l,self.capa_h) #G,エリア数,エリアのノード数,列数,行数,ε浮動小数点
        if(graph_model == 'random'):
            self.G = graph_making.Graphs(self.commodity) #品種数,areaの品種数
            self.G.randomGraph(self.G, self.degree , self.node, self.capa_l, self.capa_h) # 5 is each node is joined with its k nearest neighbors in a ring topology. 5はノード数に関係しそう　次数だから
        if(graph_model == 'nsfnet'):
            self.G = graph_making.Graphs(self.commodity) #品種数,areaの品種数　この時点では何もグラフが作られていない　インスタンスの作成？
            self.G.nsfnet(self.G, self.capa_l, self.capa_h)
        self.capacity_list = nx.get_edge_attributes(self.G,'capacity') # 全辺のcapacityの値を辞書で取得
        self.edge_list = list(enumerate(self.G.edges()))
        self.edges_notindex = []
        for z in range(len(self.edge_list)):
            self.edges_notindex.append(self.edge_list[z][1]) 
        nx.write_gml(self.G, "./deepvalue/graph.gml") # グラフの保存

        self.commodity_list = self.generate_commodity(self.commodity) # 品種作成
        with open('./deepvalue/commodity_data.csv','w') as f:
            writer=csv.writer(f,lineterminator='\n')
            writer.writerows(self.commodity_list) # 品種の保存

        self.exact_file_name = f'./deepvalue/exactsolution_training.csv' #training条件の厳密解一時保存先
        while True: #厳密解が存在するグラフで検証するための条件作成
            E = Solve_exact_solution(0,solver_type,self.exact_file_name,limittime) # Exact_Solution.pyの厳密解クラスの呼び出し
            objective_value,objective_time = E.solve_exact_solution_to_env()
            if objective_value>1.0:
                self.commodity_list = self.generate_commodity(self.commodity) # 品種作成
                with open('./deepvalue/commodity_data.csv','w') as f:
                    writer=csv.writer(f,lineterminator='\n')
                    writer.writerows(self.commodity_list) # 品種の保存
            else:
                break

        # self.random_ksps, self.allcommodity_ksps, self.allcommodity_notfstksps = self.search_ksps(self.K, self.G, self.commodity, self.commodity_list) # kspsの探索 経路ランダム抽出
        self.allcommodity_ksps, self.allcommodity_notfstksps = self.search_ksps(self.K, self.G, self.commodity, self.commodity_list) # kspsの探索
        
        if(self.initialstate==1):
            # 初期状態最短経路の場合
            # self.combination = self.searh_combination(self.random_ksps) # 抽出バージョン
            self.combination = self.searh_combination(self.allcommodity_ksps)
            self.grouping = self.combination[0] #初期パターン設定
        elif(self.initialstate==2):
            # 初期状態ランダムの場合
            self.combination = self.searh_randomcombination(self.allcommodity_ksps)
            self.grouping = self.combination[0] #初期パターン設定

        return self.grouping

def DNNmodel(env):
    model = Sequential()
    model.add(Dense(64, input_dim=env.observation_space.shape[0], activation='relu'))  # 入力層 + 中間層
    model.add(Dense(32, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))
    model.summary()
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def NNmodel(env):
    model = Sequential()
    model.add(Dense(64, input_dim=env.observation_space.shape[0], activation='relu'))  # 入力層 + 中間層
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))
    model.summary()
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def choose_action(model, state):
    q_values = model.predict(state)  # 各行動に対するQ値を予測
    filtered_q_values = np.where(state[0] == -100.0, -np.inf, q_values)  # 観測変数が-2.0の場合は、その行動を選択しない
    action = np.argmax(filtered_q_values)  # 最大のQ値を持つ行動を選択
    # print("select action",action)
    # print("state",state)
    return action

def train(env, model, episodes):
    gamma = 0.95  # 割引率
    for e in range(episodes):
        total_reward = 0
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        done = False
        while not done:
            action = choose_action(model, state)
            next_state, reward, done, _ , Maxload= env.step(action)
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            # 報酬が最大になるようにQ値を更新
            target = reward
            total_reward += reward
            if not done:
                target = reward + gamma * np.amax(model.predict(next_state)[0])
            target_f = model.predict(state)
            target_f[0][action] = target
            model.fit(state, target_f, epochs=1, verbose=0)
            state = next_state
            if done:
                print(f"Episode {e+1}/{episodes} finished with totalreward: {total_reward}")

def test_agent(env, model, nb_episodes, nb_max_episode_steps=None, callbacks=None):
    for e in range(nb_episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        done = False
        total_reward = 0
        steps = 0
        
        # エピソード開始時のコールバック
        if callbacks:
            for callback in callbacks:
                callback.on_episode_begin(e)
                
        while not done:
            action = choose_action(model, state)
            next_state, reward, done, _ , Maxload= env.step(action)
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])

            total_reward += reward
            steps += 1

            state = next_state
            
            # ステップ終了時のコールバック
            if callbacks:
                for callback in callbacks:
                    callback.on_step_end(steps, logs={"state": state, "reward": reward, "done": done})
                    
            if nb_max_episode_steps and steps >= nb_max_episode_steps:
                break
        
        # エピソード終了時のコールバック
        if callbacks:
            for callback in callbacks:
                callback.on_episode_end(e, logs={"totalreward": total_reward, "steps": steps})
                
        print(f"Episode {e+1}/{nb_episodes} finished with totalreward: {total_reward} and maxload: {Maxload} and steps: {steps}")

class CustomEpisodeLogger(rl.callbacks.Callback):
    def __init__(self,env):
        self.env = env
        self.episode = 0
        self.start_time = 0 # エピソードごとの処理時間
        self.rewards = {}  # エピソードごとの報酬を保存する辞書
        self.objective_values = []
        self.objective_time = []
        self.LP_values = []
        self.LP_time = []
        self.lowerlimit1_values = []
        self.lowerlimit1_time = []
        self.lowerlimit2_values = []
        self.lowerlimit2_time = []
        self.LR_values = []
        self.LR_time = []
        self.objective_values_ksp = []
        self.objective_time_ksp = []
        self.apploximatesolutions = []
        self.apploximatetime = []
        self.totalrewards = []

    def on_episode_begin(self, episode, logs=None):
        self.episode = episode
        filename = f'./deepvalue/commodity_file_{self.episode}.csv'
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.env.commodity_list)
        filename = f'./deepvalue/graph_{self.episode}.gml'
        nx.write_gml(self.env.G, filename)
        self.start_time = time.time() # エピソード開始時間の取得
        # 新しいエピソードが始まるたびに新しいリストを作成
        self.rewards[self.episode] = []

    def on_step_end(self, step, logs):
        reward = logs['reward']
        self.rewards[self.episode].append(reward)
        if logs.get('action') == -1:  # -1は終了フラグとして設定された場合
            self.episode_ended = True
            self.on_episode_end(self.episode, logs=None)  # 自動的にエピソード終了処理を呼ぶ

    def on_episode_end(self, episode, logs=None):
        # エピソードが終了した際に呼ばれる
        # エピソード終了時に1回だけログを表示
        end_time = time.time() # エピソード終了時間の取得
        elapsed_time = end_time - self.start_time # 処理時間計算
        apploximate_solution = env.old_maxloadfactor
        self.apploximatesolutions.append(apploximate_solution)
        self.apploximatetime.append(elapsed_time)
        # steps = logs['nb_steps']

        # ファイルにデータを書き込む
        with open('./deepvalue/commodity_data.csv','w') as f:
            writer=csv.writer(f,lineterminator='\n')
            writer.writerows(self.env.commodity_list) # 品種の保存

        filename = f'./deepvalue/pathdata_{self.episode}.csv'
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.env.grouping)

        nx.write_gml(self.env.G, "./deepvalue/graph.gml") # グラフの保存

        # approximateの書き込み
        with open(approx_file_name, 'a', newline='') as f:
            out = csv.writer(f)
            out.writerow([self.episode, apploximate_solution, elapsed_time])

        if(initialstate == 1):
            # 厳密解を求める
            E = Solve_exact_solution(self.episode,solver_type,exact_file_name,limittime) # Exact_Solution.pyの厳密解クラスの呼び出し
            objective_value,objective_time = E.solve_exact_solution_to_env() # 厳密解を計算
            self.objective_values.append(objective_value) # 厳密解情報を格納
            self.objective_time.append(objective_time)

def Graph1():
        now = datetime.datetime.now()
        # グラフ保存先ディレクトリ
        save_dir = 'episode_graphs_exact_deep/'
        # ディレクトリが存在しない場合は作成
        os.makedirs(save_dir, exist_ok=True)
        epi = episode_logger.rewards

        # 削除するキーを収集する
        keys_to_remove = [key for key, value in epi.items() if isinstance(value, list) and len(value) < test_max_step-1]

        # 収集したキーを削除する
        for key in keys_to_remove:
            del epi[key]

        # stepごとの平均reward推移
        mean_reward_list = []
        # 全てのキーをチェックし、処理をスキップする
        for i in range(test_max_step):
            heikin = 0
            for j in range(len(epi)):
                if j in epi:  # キーの存在をチェック
                    heikin += epi[j][i]  # キーが存在する場合のみ処理を実行
            mean_reward_list.append((heikin / len(epi)))

        x = list(range(1, test_max_step + 1))
        plt.plot(x, mean_reward_list, label='N={}'.format(len(epi)))
        plt.xlabel('step', fontsize=15)
        plt.ylabel('mean reward', fontsize=15)
        plt.legend(loc='upper right', fontsize=15)
        plt.xticks(x)

        # グラフをエピソードごとに保存
        filename = f"episode_{kurikaeshi + 1}_{range_commodity_l}_{now.strftime('%Y-%m-%d_%H-%M-%S')}_step_reward.png"

        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.clf()

        # 厳密解と近似解の比較
        y1 = episode_logger.objective_values # 厳密解
        y2 = episode_logger.apploximatesolutions # 近似解
        x = np.arange(len(y1)) # x軸の設定
        valid_indices1 = [i for i, value in enumerate(y1) if value is not None]
        valid_indices2 = [i for i, value in enumerate(y2) if value is not None]
        # 有効なデータのみを抽出
        valid_data1 = [y1[i] for i in valid_indices1]
        valid_data2 = [y2[i] for i in valid_indices2]
        valid_x1 = [x[i] for i in valid_indices1]
        valid_x2 = [x[i] for i in valid_indices2]
        # プロット
        plt.plot(valid_x1, valid_data1, marker='o', linestyle='-', label='exactsolution')
        plt.plot(valid_x2, valid_data2, marker='o', linestyle='-', label='approximatesolution')
        # ラベルや凡例の追加
        plt.xlabel('episode')
        plt.ylabel('value')
        # plt.title('二つのリストのプロット')
        plt.legend() # 凡例を表示

        # グラフをエピソードごとに保存
        filename = f"episode_{kurikaeshi + 1}_{range_commodity_l}_{now.strftime('%Y-%m-%d_%H-%M-%S')}_value.png"

        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.clf()

        #近似率の算出
        apploximate_rate = []
        for i in range(nb_episodes):
            if y1[i] is None:
                apploximaterate = 110
            elif y2[i] is None:
                apploximaterate = 0
            else:
                apploximaterate = y1[i]/y2[i]*100
            apploximate_rate.append(apploximaterate)
        x = list(range(1, nb_episodes + 1))
        plt.plot(x, apploximate_rate, label='approximate rate')
        # ラベルや凡例の追加
        plt.xlabel('episode')
        plt.ylabel('value')
        # plt.title('二つのリストのプロット')
        plt.legend() # 凡例を表示

        # グラフをエピソードごとに保存
        filename = f"episode_{kurikaeshi + 1}_{range_commodity_l}_{now.strftime('%Y-%m-%d_%H-%M-%S')}_rate.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.clf()

        # 計算時間の比較
        y1 = episode_logger.objective_time # 厳密解の処理時間
        y2 = episode_logger.apploximatetime # 近似解の処理時間
        x = np.arange(len(y1)) # x軸の設定
        valid_indices1 = [i for i, value in enumerate(y1) if value is not None]
        valid_indices2 = [i for i, value in enumerate(y2) if value is not None]
        # 有効なデータのみを抽出
        valid_data1 = [y1[i] for i in valid_indices1]
        valid_data2 = [y2[i] for i in valid_indices2]
        valid_x1 = [x[i] for i in valid_indices1]
        valid_x2 = [x[i] for i in valid_indices2]
        # プロット
        plt.plot(valid_x1, valid_data1, marker='o', linestyle='-', label='exactsolution time')
        plt.plot(valid_x2, valid_data2, marker='o', linestyle='-', label='approximatesolution time')
        # ラベルや凡例の追加
        plt.xlabel('episode')
        plt.ylabel('s')
        # plt.title('二つのリストのプロット')
        plt.legend() # 凡例を表示

        # グラフをエピソードごとに保存
        filename = f"episode_{kurikaeshi + 1}_{range_commodity_l}_{now.strftime('%Y-%m-%d_%H-%M-%S')}_time.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.clf()
def Graph2(): #編集が必要
    now = datetime.datetime.now()
    # グラフ保存先ディレクトリ
    save_dir = 'episode_graphs_simple_deep/'
    # ディレクトリが存在しない場合は作成
    os.makedirs(save_dir, exist_ok=True)

    #近似率の算出
    approximate_rate = []
    approximate_rate_LR = []
    y1 = episode_logger.objective_values # 厳密解
    y2 = episode_logger.apploximatesolutions # LRksp
    y3 = episode_logger.objective_values_ksp # 単純ksp

    for i in range(nb_episodes):
        if y1[i] is None:
            approximaterate = approximaterate_LR = 110
        elif y2[i] is None:
            approximaterate_LR = 0
        elif y3[i] is None:
            approximaterate = 0
        else:
            approximaterate_LR = y1[i]/y2[i]*100
            approximaterate = y1[i]/y3[i]*100
        approximate_rate.append(approximaterate)
        approximate_rate_LR.append(approximaterate_LR)
    x = np.arange(len(y1)) # x軸の設定


    valid_indices1 = [i for i, value in enumerate(approximate_rate) if value is not None]
    valid_indices2 = [i for i, value in enumerate(approximate_rate_LR) if value is not None]
    # 有効なデータのみを抽出
    valid_data1 = [approximate_rate[i] for i in valid_indices1]
    valid_data2 = [approximate_rate_LR[i] for i in valid_indices2]
    valid_x1 = [x[i] for i in valid_indices1]
    valid_x2 = [x[i] for i in valid_indices2]
    # プロット
    plt.plot(valid_x1, valid_data1, marker='o', linestyle='-', label='Simple ksp')
    plt.plot(valid_x2, valid_data2, marker='o', linestyle='-', label='LP ksp')


    # plt.plot(x, approximate_rate, label='Simple ksp')
    # plt.plot(x, approximate_rate_LR, label='LR ksp')

    # ラベルや凡例の追加
    plt.xlabel('episode')
    plt.ylabel('approximate rate[%]')
    # plt.title('二つのリストのプロット')
    plt.legend() # 凡例を表示
    # グラフをエピソードごとに保存
    filename = f"episode_{kurikaeshi + 1}_{range_commodity_l}_{now.strftime('%Y-%m-%d_%H-%M-%S')}_compare_rate.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.clf()

    # 計算時間の比較
    y1 = episode_logger.objective_time_ksp # 単純kspの処理時間
    y2 = episode_logger.apploximatetime # 近似解の処理時間
    x = np.arange(len(y1)) # x軸の設定
    valid_indices1 = [i for i, value in enumerate(y1) if value is not None]
    valid_indices2 = [i for i, value in enumerate(y2) if value is not None]
    # 有効なデータのみを抽出
    valid_data1 = [y1[i] for i in valid_indices1]
    valid_data2 = [y2[i] for i in valid_indices2]
    valid_x1 = [x[i] for i in valid_indices1]
    valid_x2 = [x[i] for i in valid_indices2]
    # プロット
    plt.plot(valid_x1, valid_data1, marker='o', linestyle='-', label='Simple ksp')
    plt.plot(valid_x2, valid_data2, marker='o', linestyle='-', label='LP ksp')
    # ラベルや凡例の追加
    plt.xlabel('episode')
    plt.ylabel('time[s]')
    # plt.title('二つのリストのプロット')
    plt.legend() # 凡例を表示
    # グラフをエピソードごとに保存
    filename = f"episode_{kurikaeshi + 1}_{range_commodity_l}_{now.strftime('%Y-%m-%d_%H-%M-%S')}_compare_time.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.clf()
# ---------------------------------------------------------------------------------------

random.seed(1) #ランダムの固定化 #2024/6/27以前1

solver_type = "pulp"
result_model = "graph1" # graph1:厳密解との比較　graph2:単純kspとの比較
graph_model = "nsfnet"
K = 10 # パスの個数
node_l = 20 # gridgraphの列数範囲
node_h = 20 # gridgraphの列数範囲
range_commodity_l = 10 # 品種の範囲
range_commodity_h = 10 # 品種の範囲
sample_size = 5  # 抽出する要素の数
capa_l = 500 # capacityの範囲
capa_h = 5000 # capacityの範囲
demand_l = 100 # 需要量の範囲
demand_h = 500 # 需要量の範囲
degree = 3

n_action = 20 # candidateの個数
obs_low = -20 # 観測変数のスペース　下限
obs_high = 20 # 観測変数のスペース　上限

initialstate = 1 #初期状態　1:最短経路　2:ランダム
initialstate_list = [1,2]
rewardstate = 2 #報酬の定義 1:最大負荷率　2:最大負荷率の差
rewardstate_list = [2,1]
housyu = 3 #1:累積 2:即時 3:test

train_max_step = 20 # 訓練時の最大step数
test_max_step = 20 # テスト時のstep数

ln_episodes =  300 # 訓練エピソード数
# ln_episodes_list = [5000,10000,15000,20000,25000,30000,35000,40000,45000,50000]
ln_episodes_list = [10000,10000]
nb_episodes = 100 # テストエピソード数
countlimit = 100000 # 初期状態の更新回数
limittime = 1800 # solverの制限時間(s)

for kurikaeshi in range(len(ln_episodes_list)):
    # ln_episodes = ln_episodes_list[kurikaeshi]
    # initialstate = initialstate_list[kurikaeshi]
    # rewardstate = rewardstate_list[kurikaeshi]
    print("rewardstate",rewardstate)

    print("set env")
    env = min_maxload_KSPs_Env(K, n_action, obs_low, obs_high, train_max_step, node_l, node_h, range_commodity_l, range_commodity_h, sample_size,capa_l,capa_h,demand_l,demand_h,graph_model,degree,initialstate,rewardstate,countlimit) # 環境の定義
    
    print("training start-------------------")
    # model = NNmodel(env) # 強化学習
    model = DNNmodel(env) # 深層強化学習
    train(env, model, ln_episodes)


    # 結果書き込みファイルの準備
    exact_file_name = f'./deepvalue/exactsolution_{kurikaeshi}_{range_commodity_l}.csv' # 厳密解のファイルを用意
    with open(exact_file_name, 'w') as f:
        out = csv.writer(f)
    approx_file_name = f'./deepvalue/approximatesolution_{kurikaeshi}_{range_commodity_l}.csv' # 近似解のファイルを用意
    with open(approx_file_name, 'w') as f:
        out = csv.writer(f)

    # LP_file_name = f'./value/LPsolution_{kurikaeshi}_{range_commodity_l}.csv' # LP解のファイルを用意
    # with open(LP_file_name, 'w') as f:
    #     out = csv.writer(f)
    # lowerlimit1_file_name = f'./value/lowerlimit1_{kurikaeshi}_{range_commodity_l}.csv' # lowerlimit1のファイルを用意
    # with open(lowerlimit1_file_name, 'w') as f:
    #     out = csv.writer(f)
    # lowerlimit2_file_name = f'./value/lowerlimit2_{kurikaeshi}_{range_commodity_l}.csv' # lowerlimit2のファイルを用意
    # with open(lowerlimit2_file_name, 'w') as f:
    #     out = csv.writer(f)
    # LR_file_name = f'./value/LRsolution_{kurikaeshi}_{range_commodity_l}.csv' # LR解のファイルを用意
    # with open(LR_file_name, 'w') as f:
    #     out = csv.writer(f)
    # simpleksp_file_name = f'./value/simpleksp_{kurikaeshi}_{range_commodity_l}.csv' # 単純kspのファイルを用意
    # with open(simpleksp_file_name, 'w') as f:
    #     out = csv.writer(f)
    
    print("test start")
    random.seed(7) #ランダムの固定化　テスト条件の固定 #2024/6/27以前7
    episode_logger = CustomEpisodeLogger(env) # テスト時にカスタムコールバックを使用してエピソードごとの処理時間を取得
    
    # テストの実行
    test_agent(env, model, nb_episodes=nb_episodes, nb_max_episode_steps=test_max_step,callbacks=[episode_logger])
    
    # break
    if (result_model == 'graph1'):
        Graph1()
    if (result_model == 'graph2'):
        Graph2()

    env.close()
    print("")