import numpy as np
from collections import defaultdict
import os
import csv
import pickle
import itertools
import networkx as nx
from networkx import shortest_simple_paths
import osmnx as ox


# genetate distance between nodes
def my_k_shortest_paths(u, v, k, road_graph_di):
    paths_gen = shortest_simple_paths(road_graph_di, u, v, "length")
    for path in itertools.islice(paths_gen, 0, k):
        yield path

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6367 * 1000

def coo_dist(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return np.sqrt(dx ** 2 + dy ** 2)

def gen_road_graph(road_graph_pkl_file, road_nodes_file, road_edges_file):
    if not os.path.exists(road_graph_pkl_file):
        # read out information about nodes and edges
        node_list = list()
        edge_list = list()

        f = open(road_nodes_file)
        line = f.readline()
        while line:
            if line == '\n':
                line = f.readline()
                continue
            node_info = line.rstrip('\n').split(' ')
            node_list.append([int(node_info[0]), float(node_info[1]), float(node_info[2])])
            line = f.readline()
        f.close()

        f = open(road_edges_file)
        line = f.readline()
        while line:
            if line == '\n':
                line = f.readline()
                continue
            edge_info = line.rstrip('\n').split(' ')
            edge_list.append([int(edge_info[0]), int(edge_info[1]), int(edge_info[2])])
            line = f.readline()
        f.close()

        # construct road network
        road_graph = nx.MultiDiGraph()

        for node in node_list:
            node_id = node[0]
            lat = node[1]
            lon = node[2]
            # add node
            road_graph.add_node(node_id, id=node_id, lon=lon, lat=lat)

        for edge in edge_list:
            # add edge
            roadID = edge[0]
            camID1 = edge[1]
            camID2 = edge[2]

            [camID1_lon, camID1_lat] = [road_graph.nodes[camID1]['lon'], road_graph.nodes[camID1]['lat']]
            [camID2_lon, camID2_lat] = [road_graph.nodes[camID2]['lon'], road_graph.nodes[camID2]['lat']]
            road_len = haversine(camID1_lon, camID1_lat, camID2_lon, camID2_lat)
            road_graph.add_edge(camID1, camID2, pre_node=camID1, succ_node=camID2, length=road_len, id=roadID)

        for node in list(road_graph.nodes):
            pred = list(road_graph.predecessors(node))
            pred = [road_graph.edges[(i, node, 0)]['id'] for i in pred]
            road_graph.nodes[node]['pred'] = pred

            succ = list(road_graph.successors(node))
            succ = [road_graph.edges[(node, i, 0)]['id'] for i in succ]
            road_graph.nodes[node]['succ'] = succ

        pickle.dump(road_graph, open(road_graph_pkl_file, "wb"))
        print('save road graph successfully!')
    elif os.path.exists(road_graph_pkl_file):
        road_graph = pickle.load(open(road_graph_pkl_file, "rb"))
        print('load road graph successfully!')

    return road_graph

# generate correspondence between camera nodes and road nodes
def gen_cid_rid_correspondence(cid_rid_correspondence_pkl_file, cam_nodes_file, road_graph):
    if not os.path.exists(cid_rid_correspondence_pkl_file):
        # read out information about nodes and edges   
        cam_pos_dict = {}
        
        f = open(cam_nodes_file)
        line = f.readline()
        while line:
            if line == '\n':
                line = f.readline()
                continue
            node_info = line.rstrip('\n').split(' ')
            cam_pos_dict[int(node_info[0])] = [float(node_info[1]), float(node_info[2])]
            line = f.readline()
        f.close()
            
        # generate camera id road id correspondence
        cid_rid_correspondence_list = list()
        for curr_cam_id in cam_pos_dict.keys():
            curr_cam_lat = cam_pos_dict[curr_cam_id][0]
            curr_cam_lon = cam_pos_dict[curr_cam_id][1]

            cam_road_node = defaultdict(int)
            cam_road_node['id'] = curr_cam_id

            min_dist = np.inf
            for node in list(road_graph.nodes):
                curr_node = road_graph.nodes[node]
                curr_node_lon, curr_node_lat = curr_node['lon'], curr_node['lat']
                dist = haversine(curr_node_lon, curr_node_lat, curr_cam_lon, curr_cam_lat)
                if dist < min_dist:
                    cam_road_node['node_id'] = node
                    min_dist = dist
            
            cid_rid_correspondence_list.append(cam_road_node)
        
        # save as .pkl file
        pickle.dump(cid_rid_correspondence_list, open(cid_rid_correspondence_pkl_file, "wb"))
        print("generarate and save camera id to road id correspondence info .pkl file successfully!")
    else:
        cid_rid_correspondence_list = pickle.load(open(cid_rid_correspondence_pkl_file, "rb"))
        print("load camera id to road id correspondence info .pkl file successfully!")

    return cid_rid_correspondence_list

# calculate shortest results between camera pairs
def gen_cam_shortest_result(road_nodes_file, road_graph, cid_rid_correspondence_list, shortest_path_results_file):
    # read out information about nodes and edges
    node_list = []
    with open(road_nodes_file, 'r') as f:
        for line in f:
            line = line.strip('\n') # remove leading/trailing whitespace and newlines
            if not line: # skip empty lines
                continue

            node_info = line.split(' ') # split by spaces
            node_list.append([int(node_info[0]), float(node_info[1]), float(node_info[2])])
    print(node_list)

    # generate cid rid mapping correspondence
    cid_to_rid_dict = {x["id"]: x["node_id"] for x in cid_rid_correspondence_list}

    # generate shortest path results
    if not os.path.exists(shortest_path_results_file):
        print("pre-computing shortest path results...")
        road_graph_di = ox.utils_graph.get_digraph(road_graph, "length")

        cam_node_list = [node[0] for node in node_list]
        cam_node_list = set(cam_node_list)
        shortest_path_results = {}
        for cam1 in cam_node_list:
            rid1 = cid_to_rid_dict[cam1]
            for cam2 in cam_node_list:
                if cam1 != cam2:
                    rid2 = cid_to_rid_dict[cam2]
                    try:
                        paths = [x for x in my_k_shortest_paths(rid1, rid2, 10, road_graph_di)]
                        shortest_path_results[(cam1, cam2)] = paths
                    except:
                        pass
            print('camera {} has been finished'.format(cam1))
        
        pickle.dump(shortest_path_results, open(shortest_path_results_file, "wb"))
        print('save shortest path results successfully!')
    elif os.path.exists(shortest_path_results_file):
        shortest_path_results = pickle.load(open(shortest_path_results_file, "rb"))
        print('load shortest path results successfully!')

    return shortest_path_results

# for cityflow dataset, the camera dictionary is the same as the road graph node dictionary
def calculate_shortest_path_distance(node1_id, node2_id, shortest_path_results, node_dict): 
    if (node1_id, node2_id) in shortest_path_results:
        shortest_path = shortest_path_results[(node1_id, node2_id)][0]
    else:
        shortest_path = []

    dist = 0
    for path_node_pre, path_node_succ in zip(shortest_path, shortest_path[1:]):
        node1_lat1, node1_lon1 = node_dict[path_node_pre]
        node2_lat1, node2_lon1 = node_dict[path_node_succ]
        dist += haversine(node1_lon1, node1_lat1, node2_lon1, node2_lat1)

    return shortest_path, dist

def gen_cam_dist_files(road_nodes_file, shortest_path_results, all_pairs_direct_distance_file, all_pairs_route_distance_file):
    # read out information about nodes and edges
    node_list = []
    with open(road_nodes_file, 'r') as f:
        for line in f:
            line = line.strip('\n') # remove leading/trailing whitespace and newlines
            if not line: # skip empty lines
                continue

            node_info = line.split(' ') # split by spaces
            node_list.append([int(node_info[0]), float(node_info[1]), float(node_info[2])])
    # print(node_list)

    # write info direct distance between node pairs
    if not os.path.exists(all_pairs_direct_distance_file):
        f = open(all_pairs_direct_distance_file, 'w')
        csv_writer = csv.writer(f)
        csv_writer.writerow(["node1", "node2", "distance", "time", "shortest path"])
        for i in range(len(node_list)):
            for j in range(len(node_list)):
                if i == j:
                    continue
                node1 = node_list[i]
                node2 = node_list[j]
                node1_id = node1[0]
                node2_id = node2[0]
                node1_lat1, node1_lon1 = node1[1], node1[2]
                node2_lat1, node2_lon1 = node2[1], node2[2]
                dist = haversine(node1_lon1, node1_lat1, node2_lon1, node2_lat1)
                time = -1
                if (node1_id, node2_id) in shortest_path_results.keys():
                    shortest_path = shortest_path_results[(node1_id, node2_id)][0]
                else:
                    shortest_path = []
                curr_pair = [node1_id, node2_id, dist, time, shortest_path]

                csv_writer.writerow(curr_pair)
        f.close()
        print('save ' + all_pairs_direct_distance_file + ' successfully!')
    else:
        print('{} already exists!'.format(all_pairs_direct_distance_file))

    # write info route distance between node pairs
    if not os.path.exists(all_pairs_route_distance_file):
        node_dict = {}
        for curr_node in node_list:
            node_dict[curr_node[0]] = curr_node[1:]

        f = open(all_pairs_route_distance_file, 'w')
        csv_writer = csv.writer(f)
        csv_writer.writerow(["node1", "node2", "distance", "time", "shortest path"])
        for i in range(len(node_list)):
            for j in range(len(node_list)):
                if i == j:
                    continue
                node1 = node_list[i]
                node2 = node_list[j]
                node1_id = node1[0]
                node2_id = node2[0]
                time = -1
                shortest_path, dist = calculate_shortest_path_distance(node1_id, node2_id, shortest_path_results, node_dict)
                curr_pair = [node1_id, node2_id, dist, time, shortest_path]
                csv_writer.writerow(curr_pair)
        f.close()
        print('save ' + all_pairs_route_distance_file + ' successfully!')
    else:
        print('{} already exists!'.format(all_pairs_route_distance_file))
    
    return

def gen_cam_path_neighbor_dict(cam_nodes_file, road_graph, shortest_path_results, cam_path_neighbor_dict_file):
    if not os.path.exists(cam_path_neighbor_dict_file):
        # read out information about nodes and edges   
        cam_pos_dict = {}
        with open(cam_nodes_file, 'r') as f:
            for line in f:
                line = line.rstrip('\n')  # remove leading/trailing whitespace and newlines
                if not line:  # skip empty lines
                    continue

                node_info = line.split(' ')
                cam_pos_dict[int(node_info[0])] = [float(node_info[1]), float(node_info[2])]
        
        # generate camera id road id correspondence
        dist_thres = 50
        cam_neighbor_road_node_dict = defaultdict(list)
        for curr_cam_id in cam_pos_dict.keys():
            curr_cam_lat = cam_pos_dict[curr_cam_id][0]
            curr_cam_lon = cam_pos_dict[curr_cam_id][1]

            for node in list(road_graph.nodes):
                curr_node = road_graph.nodes[node]
                curr_node_lon, curr_node_lat = curr_node['lon'], curr_node['lat']
                dist = haversine(curr_node_lon, curr_node_lat, curr_cam_lon, curr_cam_lat)
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
                           cam_nodes_file, 
                           road_nodes_file, 
                           shortest_path_results, 
                           cam_path_neighbor_dict):
    # generate shortest path section ratio results
    if not os.path.exists(cam_path_sec_ratio_file):
        # read out information about nodes and edges   
        cam_pos_dict = {}
        with open(cam_nodes_file, 'r') as f:
            for line in f:
                line = line.rstrip('\n')  # remove leading/trailing whitespace and newlines
                if not line:  # skip empty lines
                    continue

                node_info = line.split(' ')
                cam_pos_dict[int(node_info[0])] = [float(node_info[1]), float(node_info[2])]

        node_dict = {}
        with open(road_nodes_file, 'r') as f:
            for line in f:
                line = line.strip('\n') # remove leading/trailing whitespace and newlines
                if not line: # skip empty lines
                    continue

                node_info = line.split(' ') # split by spaces
                node_dict[int(node_info[0])] = [float(node_info[1]), float(node_info[2])]

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
                
                    _, dist = calculate_shortest_path_distance(path_cam1, path_cam2, shortest_path_results, node_dict)
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


if __name__ == "__main__":
    # parameters setting
    data_path = '/mnt/data_hdd_large/dth/home/data/cityflow_orig_data/'

    # generate road graph
    road_graph_pkl_file = os.path.join(data_path, 'cache_data/road_graph_cityflow.pkl')
    road_nodes_file = os.path.join(data_path, 'road_info', 'road_nodes.txt')
    road_edges_file = os.path.join(data_path, 'road_info', 'road_edges.txt')
    road_graph = gen_road_graph(road_graph_pkl_file, road_nodes_file, road_edges_file)
    print('generate road graph successfully!')

    # generate correspondence between camera nodes and road nodes
    cid_rid_correspondence_pkl_file = os.path.join(data_path, "cache_data/cid_rid_correspondence.pkl")
    cam_nodes_file = os.path.join(data_path, 'road_info', 'road_nodes.txt')
    cid_rid_correspondence_list = gen_cid_rid_correspondence(cid_rid_correspondence_pkl_file, 
                                                            cam_nodes_file, road_graph)
    print('generate correspondence between camera id and road id successfully!')

    # change the road graph format, genetate distance between nodes
    shortest_path_results_file = os.path.join(data_path, "cache_data/shortest_path_results_cityflow.pkl")
    shortest_path_results = gen_cam_shortest_result(road_nodes_file, 
                                                    road_graph, 
                                                    cid_rid_correspondence_list, 
                                                    shortest_path_results_file)
    all_pairs_direct_distance_file = os.path.join(data_path, 'road_info/all_pairs_direct_distance.csv')
    all_pairs_route_distance_file = os.path.join(data_path, 'road_info/all_pairs_route_distance.csv')
    gen_cam_dist_files(road_nodes_file, shortest_path_results, all_pairs_direct_distance_file, all_pairs_route_distance_file)
    print('generate shortest path results successfully!')

    # generate camera path neighbor information
    cam_path_neighbor_dict_file = os.path.join(data_path, "cache_data/cam_path_neighbor_dict.pkl")
    cam_path_neighbor_dict = gen_cam_path_neighbor_dict(cam_nodes_file, 
                                                        road_graph, 
                                                        shortest_path_results, 
                                                        cam_path_neighbor_dict_file)
    print('generate camera path neighbor dictionary successfully!')
    # generate camera paths section ratio file
    cam_path_sec_ratio_file = os.path.join(data_path, "cache_data/cam_path_sec_ratio.pkl")
    cam_path_sec_ratio = gen_cam_path_sec_ratio(cam_path_sec_ratio_file, 
                                                cam_nodes_file, 
                                                road_nodes_file, 
                                                shortest_path_results, 
                                                cam_path_neighbor_dict)
    print('generate camera path section ratio results successfully!')

    print('Done!')