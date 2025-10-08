import networkx as nx
from itertools import islice

# --- 1. GEANT2ネットワークの構築 ---
def create_geant2_graph():
    """GEANT2のグラフを生成し、リンクコストを設定する"""
    G = nx.Graph() # 今回は無向グラフとして扱う
    undirected_edges = [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 4), (2, 3), (2, 6), (3, 9),
        (4, 7), (4, 10), (5, 6), (5, 8), (6, 11), (7, 8), (7, 13), (8, 9),
        (8, 12), (9, 10), (9, 18), (10, 11), (10, 13), (11, 12), (12, 14),
        (12, 15), (13, 14), (14, 16), (15, 17), (15, 20), (16, 17), (16, 19),
        (17, 18), (18, 21), (18, 22), (19, 20), (20, 23), (21, 22), (22, 23)
    ]
    
    # 全てのリンクにコストを設定 (今回は単純に全コストを1とする)
    for u, v in undirected_edges:
        G.add_edge(u, v, cost=1)
        
    return G

# --- 2. 経路探索とフィルタリング関数 ---

def k_shortest_paths(G, source, target, k):
    """K-shortest pathを計算する"""
    return list(islice(nx.shortest_simple_paths(G, source, target, weight='cost'), k))

def get_path_cost(G, path):
    """経路の合計コストを計算する"""
    return sum(G.edges[u, v]['cost'] for u, v in zip(path[:-1], path[1:]))

def filter_for_diversity(G, paths, epsilon, theta):
    """コスト差率(ε)と経路重複率(θ)で経路候補をフィルタリングする"""
    if not paths:
        return []

    # 基準となる最短経路とそのコスト
    shortest_path = paths[0]
    shortest_cost = get_path_cost(G, shortest_path)
    
    diverse_candidates = [shortest_path]

    for path in paths[1:]:
        # a) コスト差率(ε)のチェック
        current_cost = get_path_cost(G, path)
        cost_diff_rate = (current_cost - shortest_cost) / shortest_cost
        if cost_diff_rate > epsilon:
            # print(f"  - 経路 {path} はコストが高すぎるため棄却 (コスト差率: {cost_diff_rate:.2%})")
            continue

        # b) 経路重複率(θ)のチェック
        is_too_similar = False
        path_edges = set(zip(path[:-1], path[1:]))
        for existing_path in diverse_candidates:
            existing_path_edges = set(zip(existing_path[:-1], existing_path[1:]))
            
            common_links = len(path_edges.intersection(existing_path_edges))
            # 重複率の定義：共通リンク数 / 自分のリンク数
            overlap_rate = common_links / len(path_edges) if len(path_edges) > 0 else 0
            
            if overlap_rate > theta:
                # print(f"  - 経路 {path} は {existing_path} と似すぎているため棄却 (重複率: {overlap_rate:.2%})")
                is_too_similar = True
                break
        
        if not is_too_similar:
            diverse_candidates.append(path)
            
    return diverse_candidates

# --- 3. メイン処理：フィルタリング効果の検証 ---
if __name__ == "__main__":
    G = create_geant2_graph()
    
    # --- 設定 ---
    SOURCE_NODE = 8      # 送信元 (例: ロンドン)
    TARGET_NODE = 22     # 宛先 (例: 末端ノード)
    INITIAL_K = 15       # 最初に探索する経路候補数
    EPSILON = 0.7        # コスト差率の閾値 (最短路より50%以上コストが高い経路は棄却)
    THETA = 0.5          # 経路重複率の閾値 (既存候補と50%以上リンクが重複する経路は棄却)

    # --- 実行 ---
    # 1. まず、単純なK-shortest pathを実行
    print("--- 1. 単純なK-shortest pathの結果 (上位15件) ---")
    initial_candidates = k_shortest_paths(G, SOURCE_NODE, TARGET_NODE, INITIAL_K)
    for i, path in enumerate(initial_candidates):
        cost = get_path_cost(G, path)
        print(f"  候補{i+1:<2} (コスト:{cost}): {path}")

    print("\n" + "="*50 + "\n")

    # 2. 次に、多様性フィルタリングを適用
    print(f"--- 2. 多様性フィルタリング適用後の結果 (ε={EPSILON:.0%}, θ={THETA:.0%}) ---")
    diverse_paths = filter_for_diversity(G, initial_candidates, EPSILON, THETA)
    for i, path in enumerate(diverse_paths):
        cost = get_path_cost(G, path)
        print(f"  採用{i+1:<2} (コスト:{cost}): {path}")

    print(f"\n[検証結果]")
    print(f"単純なK-shortest pathでは {len(initial_candidates)} 本の候補がリストアップされました。")
    print(f"多様性フィルタリングを適用した結果、戦略的に意味のある候補は {len(diverse_paths)} 本に絞り込まれました。")