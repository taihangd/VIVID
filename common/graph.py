from collections import defaultdict


def dfs_calc_edge_weights(node, graph, leaf_count_cache):
    # check if it is already in cache
    if node in leaf_count_cache:
        return leaf_count_cache[node]

    # if a node has no children, it is a leaf node
    if not graph[node]:
        leaf_count_cache[node] = 1
        return 1

    # recursively calculate the total number of leaf nodes of all child nodes
    total_leaves = 0
    for child in graph[node]:
        total_leaves += dfs_calc_edge_weights(child, graph, leaf_count_cache)

    leaf_count_cache[node] = total_leaves
    return total_leaves

# global leaf_count_cache
def calculate_frequency_graph(graph):
    leaf_count_cache = defaultdict(int)
	
    # First traverse all nodes once and fill the leaf node count cache
    for node in graph.keys():
        dfs_calc_edge_weights(node, graph, leaf_count_cache)
	
    # Directly use cache to set the weight of each edge
    edge_weights_dict = {}
    for node in graph:
        for child in graph[node]:
            edge_weights_dict[(node, child)] = leaf_count_cache[child]
            
    return edge_weights_dict

def dfs_distribute_weights_flow_split(node, graph, leaf_count_cache, in_degree):
    # if the current node has no children, it is a leaf node
    if not graph[node]:
        if in_degree[node] == 0:
            leaf_count_cache[node] = 1
        else:
            leaf_count_cache[node] = 1 / in_degree[node]
        return leaf_count_cache[node]
    
    # if the node's leaf count is already calculated, return the cached result
    if node in leaf_count_cache:
        return leaf_count_cache[node]
    
    # recursively calculate leaf count for children
    total_leaves = 0
    for child in graph[node]:
        child_leaves = dfs_distribute_weights_flow_split(child, graph, leaf_count_cache, in_degree)
        # add the child leaves to total, but divide by the number of incoming edges to the child
        total_leaves += child_leaves
    
    # cache the leaf count for this node
    leaf_count_cache[node] = (total_leaves + 1) / in_degree[child]
    return total_leaves

def calc_freq_graph_flow_split(graph):
    # calculate the in-degree for each node
    in_degree = defaultdict(int)
    for node in graph.keys():
        for child in graph[node]:
            in_degree[child] += 1
    
    # calculate the leaf node count and distribute weights during DFS
    leaf_count_cache = defaultdict(int)
    for node in graph.keys():
        dfs_distribute_weights_flow_split(node, graph, leaf_count_cache, in_degree)
    
    # set the weights for the edges
    edge_weights_dict = {}
    for node in graph:
        for child in graph[node]:
            edge_weights_dict[(node, child)] = leaf_count_cache[child]
            
    return edge_weights_dict

