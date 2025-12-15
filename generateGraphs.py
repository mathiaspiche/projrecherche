import os
import random
import math
import networkx as nx



#OUT_FOLDER =

def save_graph_as_dzn(G: nx.Graph, path: str) -> None:
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    edges = list(G.edges())
    N = G.number_of_nodes()
    E = len(edges)

    for (u, v) in edges:
        if u < 0 or v < 0 or u >= N or v >= N:
            raise ValueError(f"INVALID GRAPH: edge ({u},{v}) out of range for N={N}")

    with open(path, "w") as f:
        f.write(f"nb_nodes = {N};\n")
        f.write(f"nb_edges = {E};\n")
        f.write("edges = array2d(1..nb_edges, 1..2, [\n")

        for i, (u, v) in enumerate(edges):
            if i < E - 1:
                f.write(f"  {u+1}, {v+1},\n")
            else:
                f.write(f"  {u+1}, {v+1}\n")

        f.write("]);\n")



def gen_n_colorable_graph(n: int, num_colors: int, variant: str | None = None) -> nx.Graph:

    if num_colors < 1:
        raise ValueError("num_colors must be >= 1")
    if num_colors > n:
        raise ValueError("num_colors cannot exceed number of nodes")

    if variant is None:
        variant = random.choice(["basic", "sbm", "chain", "geometric", "rewired"])

    G = nx.Graph()
    G.add_nodes_from(range(n))

    base = n // num_colors
    sizes = [base] * num_colors
    leftover = n - base * num_colors
    for i in range(leftover):
        sizes[i % num_colors] += 1

    groups: list[list[int]] = []
    current = 0
    for c in range(num_colors):
        group_nodes = list(range(current, current + sizes[c]))
        current += sizes[c]
        groups.append(group_nodes)

    for g in groups:
        random.shuffle(g)

    def add_between_groups(g1, g2, p):
        for u in g1:
            for v in g2:
                if random.random() < p:
                    G.add_edge(u, v)

    if variant == "basic":
        p = random.uniform(0.05, 0.35)
        for c1 in range(num_colors):
            for c2 in range(c1 + 1, num_colors):
                add_between_groups(groups[c1], groups[c2], p)

    elif variant == "sbm":
        ps = [[0.0] * num_colors for _ in range(num_colors)]
        for c1 in range(num_colors):
            for c2 in range(c1 + 1, num_colors):
                ps[c1][c2] = ps[c2][c1] = random.uniform(0.03, 0.40)

        for c1 in range(num_colors):
            for c2 in range(c1 + 1, num_colors):
                add_between_groups(groups[c1], groups[c2], ps[c1][c2])

    elif variant == "chain":
        for c in range(num_colors - 1):
            p = random.uniform(0.08, 0.35)
            add_between_groups(groups[c], groups[c + 1], p)

    elif variant == "geometric":
        pos = {u: (random.random(), random.random()) for u in range(n)}

        def dist2(u, v):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            return (x1 - x2) ** 2 + (y1 - y2) ** 2

        base_r2 = random.uniform(0.04, 0.12)
        for c1 in range(num_colors):
            for c2 in range(c1 + 1, num_colors):
                for u in groups[c1]:
                    for v in groups[c2]:
                        if dist2(u, v) <= base_r2 and random.random() < 0.9:
                            G.add_edge(u, v)

    elif variant == "rewired":
        p = random.uniform(0.05, 0.30)
        for c1 in range(num_colors):
            for c2 in range(c1 + 1, num_colors):
                add_between_groups(groups[c1], groups[c2], p)

        m = G.number_of_edges()
        if m > 0:
            nswap = int(0.2 * m)
            try:
                nx.double_edge_swap(G, nswap=nswap, max_tries=10 * nswap)
            except Exception:
                pass

    else:
        raise ValueError(f"Unknown variant: {variant}")

    if G.number_of_edges() == 0 and n >= 2:
        if num_colors >= 2:
            g1, g2 = groups[0], groups[1]
            if g1 and g2:
                G.add_edge(random.choice(g1), random.choice(g2))

    return G

def generate_dataset_n_colorable(
    out_folder: str = OUT_FOLDER,
    num_graphs: int = 1000,
    min_nodes: int = 20,
    max_nodes: int = 200,
    num_colors: int = 4,
    seed: int = 1234,
) -> None:
    if min_nodes < num_colors:
        raise ValueError(
            f"min_nodes ({min_nodes}) must be >= num_colors ({num_colors})"
        )

    random.seed(seed)
    os.makedirs(out_folder, exist_ok=True)

    print(
        f"🎨 Generating {num_graphs} graphs into {out_folder} "
        f"(all {num_colors}-colorable)…"
    )

    variants = ["basic", "sbm", "chain", "geometric", "rewired"]

    for i in range(num_graphs):
        n = random.randint(min_nodes, max_nodes)
        variant = random.choice(variants)

        try:
            G = gen_n_colorable_graph(n, num_colors, variant=variant)
        except Exception as e:
            print(f"⚠️ Failed to generate graph {i} (n={n}, var={variant}): {e}")
            continue

        filename = f"ncolor{num_colors}_var{variant}_n{n}_{i}.dzn"
        path = os.path.join(out_folder, filename)

        save_graph_as_dzn(G, path)

        if i % 100 == 0:
            print(f"  {i}/{num_graphs} done… (last n={n}, var={variant})")

    print("✅ Generation complete.")

if __name__ == "__main__":
    generate_dataset_n_colorable(
        out_folder=OUT_FOLDER,
        num_graphs=3000,
        min_nodes=50,
        max_nodes=150,
        num_colors=20,
        seed=124,
    )

