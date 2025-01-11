# coding: utf-8
import networkx as nx
# import matplotlib.pyplot as plt
import math
import random
import re
import collections


class Flow():
    def __init__(self,g,flow_id,s,t,demand):
        self.g = g
        # フロー番号
        self.flow_id = flow_id

        # フローがエリアフローとして属しているエリアのエリア番号
        self.area_id = None

        # エリア内でのフロー番号
        self.area_flow_id = None

        # 始点s
        self.s = s
        # 終点t
        self.t = t
        # 需要量
        self.demand = demand

        # sからtまでのパス
        self.paths = []

        # update_sからupdate_tまでのパス
        self.up_paths = []


        # 更新された始点s
        self.update_s = s
        # 更新された終点t
        self.update_t = t

        # エリアグラフにおけるパス
        self.area_path = []


        # 流れている辺の集合
        self.edges = []

        # s の存在する管理領域
        candidate_s_area = []
        for a in range(g.number_of_area):
            if(s in g.area_nodes_dict[a]):
                candidate_s_area.append(a)
        self.s_area = list(candidate_s_area)
    
        # t の存在する管理領域
        candidate_t_area = []
        for a in range(g.number_of_area):
            if(t in g.area_nodes_dict[a]):
                candidate_t_area.append(a)
        self.t_area = list(candidate_t_area)

        # 既に通ったエリア（s; t が同じ管理領域内に存在しないとき）
        self.passing_area = []

        #更新後の始点update_sが存在するエリア
        self.update_s_area = list(candidate_s_area)

        #更新後の終点update_tが存在するエリア
        self.update_t_area = list(candidate_t_area)


        super(Flow, self).__init__()


    #始点の取得
    def get_id(self):
        return self.flow_id

    #始点の取得
    def get_s(self):
        return self.s

    #終点の取得
    def get_t(self):
        return self.t

    #更新後の始点の取得
    def get_update_s(self):
        return self.update_s

    #始点の更新
    def set_update_s(self,update_s):
        self.update_s = update_s
        # update_sの存在する管理領域を設定
        candidate_s_area = []
        for a in range(g.number_of_area):
            if(update_s in g.area_nodes_dict[a]):
                candidate_s_area.append(a)
        self.update_s_area = list(candidate_s_area)

    #更新後の終点の取得
    def get_update_t(self):
        return self.update_t

    #終点の更新
    def set_update_t(self,update_t):
        self.update_t = update_t
        # update_sの存在する管理領域を設定
        candidate_t_area = []
        for a in range(g.number_of_area):
            if(update_t in g.area_nodes_dict[a]):
                candidate_t_area.append(a)
        self.update_t_area = list(candidate_t_area)

    #需要量の設定
    def set_demand(self,demand):
        self.demand = demand

    #需要量の取得
    def get_demand(self):
        return self.demand

    #エリアパスの取得
    def get_area_path(self):
        return self.area_path

    #エリアパスの設定
    def set_area_path(self,area_path):
        self.area_path = area_path

    #フローが流れている辺の集合の取得
    def get_edges(self):
        return self.edges

    #フローが流れている辺の集合に新たな辺を追加
    def append_edge(self,source,target):
        self.edges.append(tuple(source,target))

    #始点sが存在するエリアを取得
    def get_s_area(self):
        return self.s_area

    #終点tが存在するエリアを取得
    def get_t_area(self):
        return self.s_area

    #更新後の始点update_sが存在するエリアを取得
    def get_update_s_area(self):
        return self.update_s_area

    #更新後の終点update_tが存在するエリアを取得
    def get_update_t_area(self):
        return self.update_t_area

    #通過したエリアの取得
    def get_passing_area(self):
        return self.passing_area

    #通過したエリアの追加
    def append_passing_area(self,area):
        self.passing_area.append(area)

    #エリアにおけるフローIDの取得
    def get_area_flow_id(self):
        return self.area_flow_id

    #エリアにおけるフローIDの設定
    def set_area_flow_id(self,area_flow_id):
        self.area_flow_id = area_flow_id

    #パスの設定
    def set_paths(self,paths):
        self.paths = paths

    #パスの取得
    def get_paths(self):
        return self.paths

    # update_sからupdate_tまでのパスの設定
    def set_up_paths(self,up_paths):
        self.up_paths = up_paths


    # update_sからupdate_tまでのパスの取得
    def get_up_paths(self):
        return self.up_paths

    # update_sからupdate_tまでのパスリストの初期化
    def clear_up_paths(self):
        self.up_paths.clear()

    # area_idの取得
    def get_area_id(self):
        return self.area_id

    # area_idの取得
    def set_area_id(self,area_id):
        self.area_id = area_id








