import networkx as nx
import numpy as np

edge_path = "data\soc-Epinions1.txt" 

G = nx.read_edgelist(
    edge_path,
    comments="#",
    nodetype=int,
    create_using=nx.DiGraph()
)

print(G.number_of_nodes(), G.number_of_edges())


def prepare_graph_structures(G):
    nodes = list(G.nodes())
    N = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # lista de vizinhos de saída (out-neighbors) por índice
    out_neighbors = [[] for _ in range(N)]
    for u, v in G.edges():
        iu = node_to_idx[u]
        iv = node_to_idx[v]
        out_neighbors[iu].append(iv)

    out_degree = np.array([len(out_neighbors[i]) for i in range(N)], dtype=float)

    return nodes, node_to_idx, out_neighbors, out_degree


def pagerank_custom(G, d=0.85, tol=1e-6, max_iter=100):
    nodes, node_to_idx, out_neighbors, out_degree = prepare_graph_structures(G)
    N = len(nodes)

    # inicialização uniforme
    pr = np.full(N, 1.0 / N, dtype=float)

    for it in range(max_iter):
        # termo de teletransporte (1 - d)/N
        pr_new = np.full(N, (1.0 - d) / N, dtype=float)

        # massa dos nós sem saída (dangling)
        dangling_sum = pr[out_degree == 0].sum()
        pr_new += d * dangling_sum / N

        # contribuição dos nós com saída
        for j in range(N):
            if out_degree[j] == 0:
                continue
            contrib = d * pr[j] / out_degree[j]
            for i in out_neighbors[j]:
                pr_new[i] += contrib

        # critério de convergência: diferença máxima entre iterações
        diff = np.abs(pr_new - pr).max()
        # print(f"iter {it}, diff={diff}")
        pr = pr_new
        if diff < tol:
            # print(f"Convergiu em {it} iterações")
            break

    # volta para dict {no_original: score}
    pr_dict = {nodes[i]: float(pr[i]) for i in range(N)}
    return pr_dict



d = 0.85

pr_custom = pagerank_custom(G, d=d, tol=1e-6, max_iter=100)
pr_nx = nx.pagerank(G, alpha=d, tol=1e-08)

# garantir que estamos comparando nos mesmos nós
common_nodes = pr_custom.keys() & pr_nx.keys()

diff_l1 = sum(abs(pr_custom[n] - pr_nx[n]) for n in common_nodes)
diff_max = max(abs(pr_custom[n] - pr_nx[n]) for n in common_nodes)

print("Diferença L1 total:", diff_l1)
print("Diferença máxima:", diff_max)

def top_k(pr_dict, k=10):
    return sorted(pr_dict.items(), key=lambda x: x[1], reverse=True)[:k]

ds = [0.5, 0.85, 0.99]
results = {}

for d in ds:
    print(f"\n=== d = {d} ===")
    pr_d = pagerank_custom(G, d=d, tol=1e-6, max_iter=100)
    results[d] = pr_d
    top10 = top_k(pr_d, 10)
    for node, score in top10:
        print(f"node={node}, score={score}")

for d in ds:
    print(f"\nTop 10 para d = {d}")
    pr_d = results[d]
    top10 = top_k(pr_d, 10)
    for node, score in top10:
        print(
            f"node={node}, PR={score:.6e}, "
            f"in_deg={G.in_degree(node)}, out_deg={G.out_degree(node)}"
        )
