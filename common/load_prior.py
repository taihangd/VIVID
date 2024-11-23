import pandas as pd


def load_dist_file(filename):
    data = pd.read_csv(filename, header=0).values.tolist()
    edge_length = {}
    for idx in range(len(data)):
        edge = data[idx]
        node1 = int(edge[0])
        node2 = int(edge[1])
        dis = float(edge[2])
        if dis == -1:
            dis = 999999999
        t = float(edge[3])
        edge_length[(node1, node2)] = (dis, t)
    return edge_length


def load_pairs(filename):
    all_pairs = pd.read_csv(filename, header=0)
    pair_info = {}
    for idx, row in all_pairs.iterrows():
        node1, node2, dis, t = int(row[0]), int(row[1]), float(row[2]), int(row[3])
        pair_info[(node1, node2)] = (dis, t)
        pair_info[(node2, node1)] = (dis, t)
    return pair_info


def load_node_dict(filename):
    node_dict = {}
    with open(filename, "r") as f:
        for line in f.readlines():
            node = line[:-1].split(" ")
            node_dict[int(node[0])] = (float(node[1]), float(node[2]))
    return node_dict
