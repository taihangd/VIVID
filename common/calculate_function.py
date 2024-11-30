import math
import time
from common.load_feature import *


def node_weight(x):
    return 1 + math.log(x)

def log_function(x):
    return math.log(x + 1)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_arr(x):
    return 1 / (1 + np.exp(-x))

def cal_appear_weight(appear_times):
    # return [node_weight(x) for x in appear_times]
    return [sigmoid(x) for x in appear_times]

def cal_dis_undirected_graph(node1, node2, edge_length):
    t = time.time()
    minval = min(node1, node2)
    maxval = max(node1, node2)
    if (minval, maxval) in edge_length.keys():
        dis = edge_length[(minval, maxval)][0]
    else:
        dis = 999999999
    t = time.time() - t
    return dis, t

def cal_dis(node1, node2, edge_length):
    t = time.time()
    if (node1, node2) in edge_length.keys():
        dis = edge_length[(node1, node2)][0]
    else:
        dis = 999999999
    t = time.time() - t
    return dis, t

def get_cos_sim(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim == 1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    
    if a_norm == 0 or b_norm == 0:
        similarity = 0
    else:
        similarity = np.dot(a, b.T) / (a_norm * b_norm)
    return similarity

def get_cos_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim == 1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    
    if a_norm == 0 or b_norm == 0:
        similarity = 0
    else:
        similarity = np.dot(a, b.T) / (a_norm * b_norm)
    dist = 1.0 - similarity
    return dist

def get_euclidean_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim == 1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    
    if a_norm == 0 or b_norm == 0:
        dist = 1
    else:
        dist = np.linalg.norm(a - b)
    return dist

