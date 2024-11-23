import json
import os
import csv
import math
from collections import defaultdict
import numpy as np
import base64
import random
import pickle
from multiprocessing import Pool
from functools import partial
import itertools
import networkx as nx
from networkx import shortest_simple_paths
import osmnx as ox


def coo_dist(x1, y1, x2, y2):
    return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

def my_k_shortest_paths(u, v, k, road_graph_di):
    paths_gen = shortest_simple_paths(road_graph_di, u, v, "length")
    for path in itertools.islice(paths_gen, 0, k):
        yield path

def from_base64(s):
    return np.frombuffer(base64.b64decode(s), np.float32)

def random_seed(total_num, seed_num):
    raw = list(range(total_num))
    random.shuffle(raw)
    return raw[:seed_num]

def load_camera_info(camera_src):
    cam_pos_dict = {}
    # write camera information to cam_pos_dict
    with open(camera_src) as file:
        for l in file:
            data = json.loads(l)
            cam_id = data['camera_id']
            cam_pos_dict[cam_id] = [data['position'][0], data['position'][1]]
    
    return cam_pos_dict

def load_map(map_src):
    map = json.load(open(map_src))
    return map

def gen_road_graph(road_graph_pkl_file):
    if not os.path.exists(road_graph_pkl_file):
        # read out edge and node information
        road_graph_info = load_map()
        
        # generate road graph
        node_list = list()
        edge_list = list()
        for curr_info in road_graph_info:
            if curr_info['type'] == 'node':
                node_list.append(curr_info)
            if curr_info['type'] == 'way':
                edge_list.append(curr_info)

        # construct road network
        road_graph = nx.MultiDiGraph()

        for curr_node_info in node_list:
            road_graph.add_node(curr_node_info['id'], x=curr_node_info['xy'][0], y=curr_node_info['xy'][1])
        
        road_sec_id = 0
        for curr_edge_info in edge_list:
            oneway = curr_edge_info['oneway']
            road_level = curr_edge_info['level']
            for nodeID_idx in range(len(curr_edge_info['nodes'])-1):
                pre_node_id = curr_edge_info['nodes'][nodeID_idx]
                succ_node_id = curr_edge_info['nodes'][nodeID_idx+1]

                [x1, y1] = [road_graph.nodes[pre_node_id]['x'], road_graph.nodes[pre_node_id]['y']]
                [x2, y2] = [road_graph.nodes[succ_node_id]['x'], road_graph.nodes[succ_node_id]['y']]
                node_dist = coo_dist(x1, y1, x2, y2)
                road_graph.add_edge(pre_node_id, succ_node_id, id=road_sec_id, 
                                    node_id_list=[pre_node_id, succ_node_id], 
                                    level=road_level, oneway=oneway, length=node_dist)
                road_sec_id += 1
                if oneway == False:
                    road_graph.add_edge(succ_node_id, pre_node_id, id=road_sec_id, 
                                        node_id_list=[succ_node_id, pre_node_id], 
                                        level=road_level, oneway=oneway, length=node_dist)
                    road_sec_id += 1
        
        pickle.dump(road_graph, open(road_graph_pkl_file, "wb"))
        print("save road network graph .pkl file successfully!")
    else:
        road_graph = pickle.load(open(road_graph_pkl_file, "rb"))
        print("load road network graph pkl file successfully!")

    return road_graph

# generate correspondence between camera nodes and road nodes
def gen_cid_rid_correspondence(cid_rid_correspondence_pkl_file, road_graph, cam_pos_dict):
    # generate camera id road id correspondence
    if not os.path.exists(cid_rid_correspondence_pkl_file):
        cid_rid_correspondence_list = list()
        for curr_cam_id in cam_pos_dict.keys():
            curr_cam_coo = cam_pos_dict[curr_cam_id]

            cam_road_node = defaultdict(int)
            cam_road_node['id'] = curr_cam_id

            min_dist = np.inf
            for node in list(road_graph.nodes):
                curr_node = road_graph.nodes[node]
                node_coo = [curr_node['x'], curr_node['y']]
                dist = coo_dist(node_coo[0], node_coo[1], curr_cam_coo[0], curr_cam_coo[1])
                if dist < min_dist:
                    cam_road_node['node_id'] = node
                    min_dist = dist
            
            cid_rid_correspondence_list.append(cam_road_node)
        
        pickle.dump(cid_rid_correspondence_list, open(cid_rid_correspondence_pkl_file, "wb"))
    else:
        cid_rid_correspondence_list = pickle.load(open(cid_rid_correspondence_pkl_file, "rb"))
    
    print("generarate and save camera id to road id correspondence info .pkl file successfully!")

    return cid_rid_correspondence_list

def gen_shortest_path_results(node_sub_list, node_list, cid_to_rid_dict, road_graph_di):
    def dist(a, b):
        (x1, y1) = road_graph_di.nodes[a]['x'], road_graph_di.nodes[a]['y']
        (x2, y2) = road_graph_di.nodes[b]['x'], road_graph_di.nodes[b]['y']
        return coo_dist(x1, y1, x2, y2)

    shortest_path_dict = {}
    for node1 in node_sub_list:
        for node2 in node_list:
            if node1 == node2:
                continue
            # transfer to road node
            cam1_id = node1[0]
            cam2_id = node2[0]
            node1_id = cid_to_rid_dict[cam1_id]
            node2_id = cid_to_rid_dict[cam2_id]
            try:
                # paths = [x for x in my_k_shortest_paths(node1_id, node2_id, 10, road_graph_di)]
                # shortest_path_dict[(cam1_id, cam2_id)] = paths
                paths = nx.astar_path(road_graph_di, node1_id, node2_id, heuristic=dist)
                shortest_path_dict[(cam1_id, cam2_id)] = [paths]
            except:
                pass
        
        print('camera {} has been finished'.format(cam1_id))
    return shortest_path_dict

def gen_cam_shortest_result(cam_shortest_path_result_pkl_file, cid_rid_correspondence, road_graph, cam_pos_dict):
    if not os.path.exists(cam_shortest_path_result_pkl_file):
        # prepare data
        cid_to_rid = {x["id"]: x["node_id"] for x in cid_rid_correspondence}
        road_graph_di = ox.utils_graph.get_digraph(road_graph, "length")
        # camera node list
        node_list = [[cam_id, cam_pos[0], cam_pos[1]] for cam_id, cam_pos in cam_pos_dict.items()]

        # generate shortest path results
        num_worker = 20
        pool = Pool(num_worker)
        nodes_num = round(np.ceil(len(node_list) / float(num_worker)))
        node_sub_lists = [node_list[i:i+nodes_num] for i in range(0, len(node_list), nodes_num)]
        gen_shortest_path_results_partial = partial(gen_shortest_path_results, node_list=node_list, cid_to_rid_dict=cid_to_rid, road_graph_di=road_graph_di)
        shortest_path_dict_list = pool.map(gen_shortest_path_results_partial, node_sub_lists)
        pool.close()
        pool.join()

        shortest_path_dict = shortest_path_dict_list[0]
        for curr_shortest_path_dict in shortest_path_dict_list[1:]:
            shortest_path_dict.update(curr_shortest_path_dict)
        pickle.dump(shortest_path_dict, open(cam_shortest_path_result_pkl_file, "wb"))
        print('save camera shortest path results .pkl successfully!')
    else:
        shortest_path_dict = pickle.load(open(cam_shortest_path_result_pkl_file, "rb"))
        print('load camera shortest path results .pkl successfully!')
    
    return shortest_path_dict

def calculate_shortest_path_distance(cam1_id, cam2_id, shortest_path_results, cam_pos_dict, road_graph):
    if (cam1_id, cam2_id) in shortest_path_results:
        shortest_path = shortest_path_results[(cam1_id, cam2_id)][0]
    else:
        shortest_path = []

    dist = 0
    if len(shortest_path) == 1:
        node1_x, node1_y = cam_pos_dict[cam1_id]
        node2_x, node2_y = cam_pos_dict[cam2_id]
        dist = coo_dist(node1_x, node1_y, node2_x, node2_y)
    else:
        for path_node_pre, path_node_succ in zip(shortest_path, shortest_path[1:]):
            node1_x, node1_y = road_graph.nodes[path_node_pre]['x'], road_graph.nodes[path_node_pre]['y']
            node2_x, node2_y = road_graph.nodes[path_node_succ]['x'], road_graph.nodes[path_node_succ]['y']
            dist += coo_dist(node1_x, node1_y, node2_x, node2_y)

    return shortest_path, dist

def gen_cam_dist_files(cam_shortest_path_result_pkl_file, 
                       all_pairs_direct_distance_file_name, 
                       all_pairs_route_distance_file_name,
                       cam_pos_dict,
                       road_graph):
    # write direct distance info between node pairs
    if os.path.exists(cam_shortest_path_result_pkl_file):
        shortest_path_dict = pickle.load(open(cam_shortest_path_result_pkl_file, "rb"))

    f = open(all_pairs_direct_distance_file_name, 'w')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["node1", "node2", "distance", "time", "shortest path"])
    for cam1_id in cam_pos_dict.keys():
        for cam2_id in cam_pos_dict.keys():
            if cam1_id == cam2_id:
                continue
            
            node1_x, node1_y = cam_pos_dict[cam1_id]
            node2_x, node2_y = cam_pos_dict[cam2_id]
            dist = coo_dist(node1_x, node1_y, node2_x, node2_y)
            time = -1
            if (cam1_id, cam2_id) in shortest_path_dict.keys():
                paths = shortest_path_dict[(cam1_id, cam2_id)]
                shortest_path = paths[0]
            else:
                shortest_path = []

            curr_pair = [cam1_id, cam2_id, dist, time, shortest_path]

            csv_writer.writerow(curr_pair)
    f.close()
    print(f'save {all_pairs_direct_distance_file_name} successfully!')

    # write route distance info between node pairs
    f = open(all_pairs_route_distance_file_name, 'w')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["node1", "node2", "distance", "time", "shortest path"])
    for cam1_id in cam_pos_dict.keys():
        for cam2_id in cam_pos_dict.keys():
            if cam1_id == cam2_id:
                continue
            
            time = -1
            shortest_path, dist = calculate_shortest_path_distance(cam1_id, cam2_id, shortest_path_dict, cam_pos_dict, road_graph)
            curr_pair = [cam1_id, cam2_id, dist, time, shortest_path]
            csv_writer.writerow(curr_pair)
    f.close()
    print(f'save {all_pairs_route_distance_file_name} successfully!')
    
def gen_cam_path_neighbor_dict(cam_pos_dict, road_graph, shortest_path_results, cam_path_neighbor_dict_file):
    if not os.path.exists(cam_path_neighbor_dict_file):
        # generate camera id road id correspondence
        dist_thres = 50
        cam_neighbor_road_node_dict = defaultdict(list)
        for curr_cam_id in cam_pos_dict.keys():
            curr_cam_x, curr_cam_y = cam_pos_dict[curr_cam_id]
            for node in list(road_graph.nodes):
                curr_node = road_graph.nodes[node]
                curr_node_x, curr_node_y = curr_node['x'], curr_node['y']
                dist = coo_dist(curr_node_x, curr_node_y, curr_cam_x, curr_cam_y)
                if dist < dist_thres:
                    cam_neighbor_road_node_dict[curr_cam_id].append(node)

        road_node_neighbor_cam_dict = defaultdict(list)
        for cam_id, road_nodes in cam_neighbor_road_node_dict.items():
            for road_node in road_nodes:
                road_node_neighbor_cam_dict[road_node].append(cam_id)

        # generate camera path neighbor dictionary
        cam_path_neighbor_dict = defaultdict(list)
        for (cam1, cam2), shortest_path_list in shortest_path_results.items():
            for road_node in shortest_path_list[0]: # traverse each road node in the shortest path
                if road_node not in road_node_neighbor_cam_dict.keys():
                    continue
                curr_node_cam_list = [cam for cam in road_node_neighbor_cam_dict[road_node] if cam != cam1 and cam != cam2]
                cam_path_neighbor_dict[(cam1, cam2)] += curr_node_cam_list
        
        # remove duplicate elements
        for (cam1, cam2), neighbor_cam_list in cam_path_neighbor_dict.items():
            seen = set()
            neighbor_cam_list = [neighbor_cam for neighbor_cam in neighbor_cam_list if not (neighbor_cam in seen or seen.add(neighbor_cam))]

        # save as .pkl file
        pickle.dump(cam_path_neighbor_dict, open(cam_path_neighbor_dict_file, "wb"))
        print("generarate and save cam_path_neighbor_dict file successfully!")
    else:
        cam_path_neighbor_dict = pickle.load(open(cam_path_neighbor_dict_file, "rb"))
        print("load cam_path_neighbor_dict file successfully!")

    return cam_path_neighbor_dict

def gen_cam_path_sec_ratio(cam_path_sec_ratio_file, 
                           cam_pos_dict, 
                           shortest_path_results, 
                           cam_path_neighbor_dict, 
                           road_graph):
    # generate shortest path section ratio results
    if not os.path.exists(cam_path_sec_ratio_file):
        print("computing path section ratio")
        # generate each road section ratio in shortest path
        cam_path_sec_ratio = {}
        for cam1_id in cam_pos_dict.keys():
            for cam2_id in cam_pos_dict.keys():
                if cam1_id == cam2_id:
                    continue
                
                if (cam1_id, cam2_id) not in shortest_path_results.keys():
                    continue
                
                neighbor_cam_list = cam_path_neighbor_dict[(cam1_id, cam2_id)]
                if len(neighbor_cam_list) == 0:
                    continue
                
                road_total_len = 0
                road_len_list = []
                for path_cam1, path_cam2 in zip([cam1_id] + neighbor_cam_list, neighbor_cam_list + [cam2_id]):
                    if (path_cam1, path_cam2) not in shortest_path_results.keys():
                        road_total_len = 0
                        road_len_list = []
                        break
                
                    _, dist = calculate_shortest_path_distance(path_cam1, path_cam2, shortest_path_results, cam_pos_dict, road_graph)
                    road_total_len += dist
                    road_len_list.append(dist)

                if len(road_len_list) == 0:
                    continue

                road_len_arr = np.array(road_len_list).cumsum(axis=0)
                sec_ratio = road_len_arr / road_total_len
                sec_ratio = sec_ratio.tolist()
                sec_ratio = [0.] + sec_ratio
                cam_path_sec_ratio[(cam1_id, cam2_id)] = sec_ratio

        pickle.dump(cam_path_sec_ratio, open(cam_path_sec_ratio_file, "wb"))
        print('save camera path section ratio results successfully!')
    elif os.path.exists(cam_path_sec_ratio_file):
        cam_path_sec_ratio = pickle.load(open(cam_path_sec_ratio_file, "rb"))
        print('load camera path section ratio results successfully!')
    
    return
    
if __name__ == '__main__':
    # parameter setting
    data_path = '/mnt/data_hdd_large/dth/home/data/MMVC/'
    camera_src = data_path + "cameras.json"
    map_src = data_path + "map.json"
    map_dst_path = data_path + "mmvc_dataset/map/"

    cache_data_path = data_path + 'mmvc_dataset/cache_data/'
    road_graph_pkl_file = cache_data_path + 'road_graph.pkl'
    cid_rid_correspondence_pkl_file = os.path.join(cache_data_path, 'cid_rid_correspondence.pkl')
    cam_shortest_path_result_pkl_file = os.path.join(cache_data_path, 'cam_shortest_path_result.pkl')
    cam_shortest_path_road_sec_ratio_pkl_file = os.path.join(cache_data_path, 'cam_shortest_path_road_sec_ratio.pkl')
    
    if not os.path.exists(map_dst_path):
        os.makedirs(map_dst_path)
    if not os.path.exists(cache_data_path):
        os.makedirs(cache_data_path)

    # generate road graph
    road_graph = gen_road_graph(road_graph_pkl_file)
    print('generate road graph successfully!')

    # generate camera direct and route distance files
    cam_pos_dict = load_camera_info(camera_src)
    cid_rid_correspondence_list = gen_cid_rid_correspondence(cid_rid_correspondence_pkl_file, road_graph, cam_pos_dict)
    print('generate correspondence between camera id and road id successfully!')

    # generate shortest results between camera pairs
    shortest_path_dict = gen_cam_shortest_result(cam_shortest_path_result_pkl_file, 
                                                 cid_rid_correspondence_list, 
                                                 road_graph, 
                                                 cam_pos_dict)
    all_pairs_direct_distance_file_name = os.path.join(map_dst_path, 'all_pairs_direct_distance.csv')
    all_pairs_route_distance_file_name = os.path.join(map_dst_path, 'all_pairs_route_distance.csv')
    gen_cam_dist_files(cam_shortest_path_result_pkl_file, 
                       all_pairs_direct_distance_file_name, 
                       all_pairs_route_distance_file_name,
                       cam_pos_dict,
                       road_graph)
    print('generate shortest path results successfully!')
    
    # generate camera path neighbor information
    cam_path_neighbor_dict_file = os.path.join(cache_data_path, "cam_path_neighbor_dict.pkl")
    cam_path_neighbor_dict = gen_cam_path_neighbor_dict(cam_pos_dict, 
                                                        road_graph, 
                                                        shortest_path_dict, 
                                                        cam_path_neighbor_dict_file)
    print('generate camera path neighbor dictionary successfully!')
    # generate camera paths section ratio file
    cam_path_sec_ratio_file = os.path.join(cache_data_path, "cam_path_sec_ratio.pkl")
    cam_path_sec_ratio = gen_cam_path_sec_ratio(cam_path_sec_ratio_file, 
                                                cam_pos_dict, 
                                                shortest_path_dict, 
                                                cam_path_neighbor_dict,
                                                road_graph)
    print('generate camera path section ratio results successfully!')

    print('Done!')
