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

def cal_edge_weight_delta(w1, w2):
    mean_weight = (w1 + w2) / 2
    min_weight = min(w1, w2)
    delta = 0.5 * (sigmoid(5 * mean_weight) - 0.5)
    return -delta

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

def get_cos_similar(v1, v2):
    if v1.shape != v2.shape:
        raise RuntimeError("array {} shape not match {}".format(v1.shape, v2.shape))
    if v1.ndim == 1:
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
    elif v1.ndim == 2:
        v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
        v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(v1.ndim))
    similarity = np.dot(v1, v2.T) / (v1_norm * v2_norm)
    return similarity

def cal_vision(idx1, idx2, folder, feat_dim):
    t = time.time()
    filename = folder + '/avg_features.bin'
    feature1 = load_avg_feature(filename, idx1, feat_dim)
    feature2 = load_avg_feature(filename, idx2, feat_dim)
    vision = get_cos_similar(feature1, feature2)
    # transfer to vision distance
    vis_dist = 1. - vision
    t = time.time() - t
    return vis_dist, t

def cal_score(val, max_val, min_val):
    if min_val == max_val or val <= min_val:
        return 1
    if val >= max_val:
        return 0
    score = (max_val - val) / (max_val - min_val)
    score = max(min(score, 1), 0)
    return score

def score_fusion_min(score1, score2, score3):
    if score1 != 0 and score2 != 0 and score3 != 0:
        fusion_score = fusion_score = min(min(score1, score2), score3)
    else:
        fusion_score = 0

    return fusion_score

def score_fusion_weighted_mean(score1, score2, score3, weight_list=[0.4, 0.4, 0.2]):
    fusion_score = score1 * weight_list[0] + score2 * weight_list[1] + score3 * weight_list[2]
    return fusion_score

def score_fusion_harmonic_mean(score1, score2, score3):
    if score1 != 0 and score2 != 0 and score3 != 0:
        fusion_score = 3 / (1 / score1 + 1 / score2 + 1 / score3)
    else:
        fusion_score = 0
    return fusion_score

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

def cal_adaptive_threshold(x):
    a = 1.6263676512927179
    b = -2.5025352845881614
    c = 0.06187670820910257
    return max(1 / (a * np.sqrt(x) + b), 0.45)

def get_percentile(data, low_bound, up_bound):
    data.sort()
    all_num = len(data)
    low_bound_index = max(round(all_num * low_bound), 0)
    up_bound_index = min(round(all_num * up_bound), all_num - 1)
    return data[low_bound_index], data[up_bound_index]


def get_interval_range(data):
    data.sort()
    min_val = data[0]
    sec_val = data[min(bisect.bisect_right(data, min_val), len(data)-1)] # limit the range to avoid exceeding the index
    max_val = data[round(len(data) * 0.5)]
    min_range = sec_val - min_val
    max_range = max_val - min_val
    return min_range, max_range


