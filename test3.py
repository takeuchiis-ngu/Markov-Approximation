import networkx as nx
import matplotlib.pyplot as plt

# --- GEANT2の接続情報 (元の無向エッジリスト) ---
# 24ノード, 37エッジ
undirected_edges = [
    (1, 2), (1, 3), (1, 6), (2, 4), (2, 5),
    (3, 4), (3, 7), (4, 10), (5, 8), (5, 11),
    (6, 7), (6, 9), (7, 12), (8, 9), (8, 14),
    (9, 10), (9, 13), (10, 11), (10, 19), (11, 12),
    (11, 14), (12, 13), (13, 15), (13, 16), (14, 15),
    (15, 17), (16, 18), (16, 21), (17, 18), (17, 20),
    (18, 19), (19, 22), (19, 23), (20, 21), (21, 24),
    (22, 23), (23, 24)
]

# ★★★ 変更点(1): 有向グラフ (Directed Graph) オブジェクトを作成 ★★★
G = nx.DiGraph()

# ★★★ 変更点(2): 各エッジを双方向の有向辺として追加 ★★★
for u, v in undirected_edges:
    G.add_edge(u, v) # u -> v の方向
    G.add_edge(v, u) # v -> u の方向

# --- グラフの描画 ---
# 描画サイズを指定
plt.figure(figsize=(12, 8))

# ノードの配置方法を決定
pos = nx.kamada_kawai_layout(G)

# グラフを描画 (DiGraphを描画すると自動的に矢印が付きます)
nx.draw(G, pos, with_labels=True, node_color='skyblue',
        node_size=700, edge_color='gray', font_size=10, font_weight='bold',
        arrowsize=15) # 矢印のサイズを調整

# グラフのタイトルを設定
plt.title("GEANT2 Network Topology (Directed Graph)", size=15)

# 描画を実行
plt.show()