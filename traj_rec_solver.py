import shutil
import json
import math
import pickle
import fastkde
import networkx as nx
import osmnx as ox
import pandas as pd
from abc import ABCMeta
from scipy.integrate import cumtrapz
from scipy.interpolate import RegularGridInterpolator
from collections import defaultdict
from topk.topk import *
from topk.camera import *
from topk.feature_gallery import *
from common.graph import *


class VIVID(object):
    __metaclass__ = ABCMeta

    def __init__(self, cfg):
        super(VIVID, self).__init__()
        # configuration file parameter assignment
        self.dataset = cfg.dataset
        self.dataset_path = cfg.dataset_path
        self.traj_len = cfg.traj_len
        self.node_num = cfg.node_num
        self.video_time = cfg.video_time
        self.fps = cfg.fps
        self.down_sample_fps = cfg.down_sample_fps
        self.k = cfg.k
        self.delta = cfg.delta
        self.neighbor_cam_num_thres = cfg.neighbor_cam_num_thres
        self.output_path = cfg.output_path
        self.cam_id_list = cfg.cam_id_list
        self.cam_cand_num = cfg.cam_cand_num
        self.time_gap = cfg.time_gap
        self.feat_dim = cfg.feat_dim
        self.frame_tpye_byte_num = cfg.frame_tpye_byte_num
        self.node_feats_path = cfg.node_feats_path
        self.time_range_file = cfg.time_range_file
        self.topk_cand_save = cfg.topk_cand_save
        self.cfg = cfg

        ## prepare output path
        self.prepare_output_folders()

        ## load road and index information
        self.load_dataset_info(cfg)

        ## ground truth information
        self.ground_truth = set()
        self.ground_truth_range = {}
        self.filtered_ground_truth = set()
        self.filtered_ground_truth_range = {}
        self.traj_gt = []
        self.gt_num = 0

        ## intermediate results
        self.selection_path = []
        self.ans_node_list = []
        self.ans_time_list = []

        ## break-down time
        # running time for each query
        self.time_app_feat_search = 0
        self.time_plate_feat_search = 0
        self.time_search_space_fusion = 0
        self.time_sim_thres_trunc = 0
        self.time_find_topk = 0
        self.time_est_velocity = 0
        self.time_query_snapshot_joint_sim = 0
        self.time_construct_motion_prob_graph = 0
        self.time_build_association_graph = 0
        self.time_path_extraction = 0
        # running time for each stage
        self.time_search_all = 0
        self.time_find_topk_all = 0
        self.time_build_graph_all = 0
        self.time_path_extraction_all = 0
        
        self.time_graph_completion = 0

        # overall evaluation metrics
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.all_p = []
        self.all_r = []
        self.all_f1 = []
        self.all_inference_time = []
        # top-k related statistics
        self.top_k_precision = 0
        self.top_k_recall = 0
        self.top_k_f1 = 0
        self.all_top_k_p = []
        self.all_top_k_r = []
        self.all_top_k_f1 = []
        self.all_top_k_t = []

        # path selection
        self.longest_path_list = []
        self.selection_path_score_list = []

        # path merge
        self.road_graph_file = os.path.join(self.dataset_path, self.folder_name, cfg.road_graph_file)
        self.road_graph = None
        self.road_graph_di = None
        self.cid_to_rid_file = os.path.join(self.dataset_path, self.folder_name, cfg.cid_to_rid_file)
        self.cid_to_rid_list = None
        self.cid_to_rid_dict = None
        self.rid_to_cid_dict = None
        self.shortest_path_results_file = os.path.join(self.dataset_path, self.folder_name, cfg.shortest_path_results_file)
        self.shortest_path_results = None
        self.cam_path_neighbor_dict_file = os.path.join(self.dataset_path, self.folder_name, cfg.cam_path_neighbor_dict_file)
        self.cam_path_neighbor_dict = None
        self.cam_path_sec_ratio_file = os.path.join(self.dataset_path, self.folder_name, cfg.cam_path_sec_ratio_file)
        self.cam_path_sec_ratio = None

    def reset(self):
        self.selection_path = []
        self.ans_node_list = []
        self.ans_time_list = []
        
        # running time for each query
        self.time_app_feat_search = 0
        self.time_plate_feat_search = 0
        self.time_search_space_fusion = 0
        self.time_sim_thres_trunc = 0
        self.time_find_topk = 0
        self.time_est_velocity = 0
        self.time_query_snapshot_joint_sim = 0
        self.time_construct_motion_prob_graph = 0
        self.time_build_association_graph = 0
        self.time_path_extraction = 0
        
        self.time_graph_completion = 0

    def reset_dataset_statistics(self):
        # running time for each stage
        self.time_search_all = 0
        self.time_find_topk_all = 0
        self.time_build_graph_all = 0
        self.time_path_extraction_all = 0
    
    def prepare_output_folders(self):
        # initialize folder names
        self.folder_name = ""
        self.scalability_analysis_folder_name = "t%02d_c%03d_len%02d" % (self.video_time, self.node_num, self.traj_len)
        base_output_path = os.path.join(self.output_path, self.dataset, self.scalability_analysis_folder_name)
        # create the base output folder if it doesn't exist
        os.makedirs(base_output_path, exist_ok=True)

        # create the top-k folder
        topk_folder = os.path.join(base_output_path, "top_%d" % self.k)
        os.makedirs(topk_folder, exist_ok=True)

        # set the output folder path
        self.output_folder = os.path.join(topk_folder, "delta_{}".format(self.delta))
        # remove existing output folder if necessary
        if not self.topk_cand_save and os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        # create the output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

        return
        
    def load_dataset_info(self, cfg):
        base_path = os.path.join(self.dataset_path, self.folder_name)
        if self.dataset == "cityflow":
            route_dist_filename = os.path.join(base_path, "road_info", "all_pairs_route_distance.csv")
            self.app_feat_partition_file = "partition.txt"
            self.plate_feat_partition_file = None
            self.time_range_dict = None
        elif self.dataset in ["uv", "uv-z"]:
            route_dist_filename = os.path.join(self.dataset_path, "map", self.folder_name, "all_pairs_route_distance.csv")
            self.app_feat_partition_file = "app_feat_partition.txt"
            self.plate_feat_partition_file = "plate_feat_partition.txt"
            self.time_range_dict = self.load_traj_gt_frame_range(self.time_range_file)
        elif self.dataset == "carla":
            route_dist_filename = os.path.join(self.dataset_path, "cache_data", "all_pairs_route_distance.csv")
            self.app_feat_partition_file = "partition.txt"
            self.plate_feat_partition_file = None
            self.time_range_dict = None
        elif self.dataset == "video-sg":
            route_dist_filename = os.path.join(base_path, "road_info", "all_pairs_route_distance.csv")
            self.app_feat_partition_file = "partition.txt"
            self.plate_feat_partition_file = None
            self.time_range_dict = None

        # load cam node dictionary and edge lengths
        self.cam_node_dict = self.load_cam_node_dict(cfg.cam_node_file)
        self.edge_length = self.load_dist_file(route_dist_filename)

        return

    def set_cam_id_list(self):
        if self.cam_id_list == None or len(self.cam_id_list) == 0:
            self.cam_id_list = list(self.cam_node_dict.keys())
        return
    
    def set_road_graph_info(self):
        road_graph_file = self.road_graph_file
        if os.path.exists(road_graph_file):
            road_graph = pickle.load(open(road_graph_file, "rb"))
            self.road_graph = road_graph
            road_graph_di = ox.utils_graph.convert.to_digraph(road_graph, "length")
            self.road_graph_di = road_graph_di
        return road_graph

    def set_shortest_path_results_info(self):
        shortest_path_results_file = self.shortest_path_results_file
        if os.path.exists(shortest_path_results_file):
            shortest_path_results = pickle.load(open(shortest_path_results_file, "rb"))
            self.shortest_path_results = shortest_path_results
        return shortest_path_results

    def set_cam_path_neighbor_dict_info(self):
        cam_path_neighbor_dict_file = self.cam_path_neighbor_dict_file
        if os.path.exists(cam_path_neighbor_dict_file):
            cam_path_neighbor_dict = pickle.load(open(cam_path_neighbor_dict_file, "rb"))

            new_cam_path_neighbor_dict = {}
            for cam_pair, neighbor_cam_list in cam_path_neighbor_dict.items():
                if len(neighbor_cam_list) < self.neighbor_cam_num_thres:
                    new_cam_path_neighbor_dict[cam_pair] = neighbor_cam_list
            
            self.cam_path_neighbor_dict = new_cam_path_neighbor_dict
        return cam_path_neighbor_dict

    def set_cam_path_sec_ratio_info(self):
        cam_path_sec_ratio_file = self.cam_path_sec_ratio_file
        if os.path.exists(cam_path_sec_ratio_file):
            cam_path_sec_ratio = pickle.load(open(cam_path_sec_ratio_file, "rb"))

            new_cam_path_sec_ratio = {}
            for cam_pair, neighbor_cam_list in cam_path_sec_ratio.items():
                if len(neighbor_cam_list) < self.neighbor_cam_num_thres:
                    new_cam_path_sec_ratio[cam_pair] = neighbor_cam_list

            self.cam_path_sec_ratio = new_cam_path_sec_ratio
        return cam_path_sec_ratio
    
    def set_cid_rid_dict(self):
        cid_to_rid_file = self.cid_to_rid_file
        if os.path.exists(cid_to_rid_file):
            cid_to_rid_list = pickle.load(open(cid_to_rid_file, "rb"))
            self.cid_to_rid_list = cid_to_rid_list
            cid_to_rid_dict = {x["id"]: x["node_id"] for x in cid_to_rid_list}
            rid_to_cid_dict = {}
            for correspondence in cid_to_rid_list:
                if correspondence['node_id'] in rid_to_cid_dict.keys():
                    rid_to_cid_dict[correspondence['node_id']].append(correspondence['id'])
                else:
                    rid_to_cid_dict[correspondence['node_id']] = [correspondence['id']]
            self.cid_to_rid_dict = cid_to_rid_dict
            self.rid_to_cid_dict = rid_to_cid_dict
        return
    
    # load trajectory gt
    def load_traj_gt(self, gt_file):
        if self.dataset == "cityflow":
            self.load_traj_gt_cityflow(gt_file)
        if self.dataset == "uv" or self.dataset == "uv-z":
            self.load_traj_gt_mmvc(gt_file)
        if self.dataset == "carla":
            self.load_traj_gt_carla(gt_file)
        if self.dataset == "video-sg":
            self.load_traj_gt_videosg(gt_file)

    def load_traj_gt_cityflow(self, gt_file):
        cam_id_list = self.cam_id_list

        ground_truth = set()
        ground_truth_range = {}
        with open(gt_file, 'r') as f:
            gt = [line.strip('\n').split(',') for line in f.readlines()]
        for content in gt:
            node, st, et = int(content[0]), int(content[1]), int(content[2])
            if node in cam_id_list and st <= et:
                if node in ground_truth_range.keys():
                    ground_truth_range[node].append((st, et))
                else:
                    ground_truth_range[node] = [(st, et)]
                for t in range(st, et + 1):
                    ground_truth.add((node, t))
        self.ground_truth = ground_truth
        self.ground_truth_range = ground_truth_range
        
        # self.gt_num = len(gt)
        gt_num = 0
        for gt_line in gt:
            if int(gt_line[0]) in cam_id_list:
                gt_num += 1
        self.gt_num = gt_num

        self.set_cid_rid_dict()
        traj_gt = self.traj_gt
        cid_to_rid_dict = self.cid_to_rid_dict
        for content in gt:
            node, st, et = int(content[0]), int(content[1]), int(content[2])
            if node in cam_id_list and st <= et:
                traj_gt.append(cid_to_rid_dict[node])

    def load_traj_gt_mmvc(self, gt_file):
        if not os.path.exists(gt_file):
            self.gt_num = 0
            return

        ground_truth = set()
        ground_truth_range = {}
        with open(gt_file, 'r') as f:
            gt = [line.strip('\n').split(',') for line in f.readlines()][:self.traj_len]
        for content in gt:
            node, st, et = int(content[0]), int(content[1]), int(content[2])
            if st <= et:
                if node in ground_truth_range.keys():
                    ground_truth_range[node].append((st, et))
                else:
                    ground_truth_range[node] = [(st, et)]
                for t in range(st, et + 1):
                    ground_truth.add((node, t))
        self.ground_truth = ground_truth
        self.ground_truth_range = ground_truth_range

        filtered_ground_truth = set()
        filtered_ground_truth_range = {}
        with open(gt_file, 'r') as f:
            filtered_gt = [line.strip('\n').split(',') for line in f.readlines()][self.traj_len:]
        for content in filtered_gt:
            node, st, et = int(content[0]), int(content[1]), int(content[2])
            if st <= et:
                if node in filtered_ground_truth_range.keys():
                    filtered_ground_truth_range[node].append((st, et))
                else:
                    filtered_ground_truth_range[node] = [(st, et)]
                for t in range(st, et + 1):
                    filtered_ground_truth.add((node, t))
        self.filtered_ground_truth = filtered_ground_truth
        self.filtered_ground_truth_range = filtered_ground_truth_range

        self.gt_num = len(gt)

        self.set_cid_rid_dict()
        traj_gt = self.traj_gt
        cid_to_rid_dict = self.cid_to_rid_dict
        for content in gt:
            node, st, et = int(content[0]), int(content[1]), int(content[2])
            if st <= et:
                traj_gt.append(cid_to_rid_dict[node])
        return

    def load_traj_gt_carla(self, gt_file):
        # load camera id list
        self.set_cam_id_list()
        cam_id_list = self.cam_id_list

        ground_truth = set()
        ground_truth_range = {}
        with open(gt_file, 'r') as f:
            gt = [line.strip('\n').split(',') for line in f.readlines()][:self.traj_len]
        for content in gt:
            node, st, et = int(content[0]), int(content[1]), int(content[2])
            if st <= et and node in cam_id_list:
                if node in ground_truth_range.keys():
                    ground_truth_range[node].append((st, et))
                else:
                    ground_truth_range[node] = [(st, et)]
                for t in range(st, et + 1):
                    ground_truth.add((node, t))
        self.ground_truth = ground_truth
        self.ground_truth_range = ground_truth_range

        filtered_ground_truth = set()
        filtered_ground_truth_range = {}
        with open(gt_file, 'r') as f:
            filtered_gt = [line.strip('\n').split(',') for line in f.readlines()][self.traj_len:]
        for content in filtered_gt:
            node, st, et = int(content[0]), int(content[1]), int(content[2])
            if st <= et and node in cam_id_list:
                if node in filtered_ground_truth_range.keys():
                    filtered_ground_truth_range[node].append((st, et))
                else:
                    filtered_ground_truth_range[node] = [(st, et)]
                for t in range(st, et + 1):
                    filtered_ground_truth.add((node, t))
        self.filtered_ground_truth = filtered_ground_truth
        self.filtered_ground_truth_range = filtered_ground_truth_range

        self.set_cid_rid_dict()
        cid_to_rid_dict = self.cid_to_rid_dict

        # get gt num in road node id
        gt_rid_set = set()
        for gt_line in gt:
            if int(gt_line[0]) in cam_id_list:
                gt_rid_set.add((cid_to_rid_dict[int(gt_line[0])], gt_line[1], gt_line[2]))
        self.gt_num = len(gt_rid_set)

        # get trajectory gt in road node id
        traj_gt = self.traj_gt
        for content in gt:
            node, st, et = int(content[0]), int(content[1]), int(content[2])
            if node in cam_id_list and st <= et:
                traj_gt.append(cid_to_rid_dict[node])

    def load_traj_gt_videosg(self, gt_file):
        # load camera id list
        self.set_cam_id_list()
        cam_id_list = self.cam_id_list

        ground_truth = set()
        ground_truth_range = {}
        with open(gt_file, 'r') as f:
            gt = [line.strip('\n').split(',') for line in f.readlines()][:self.traj_len]
        for content in gt:
            node, st, et = int(content[0]), int(content[1]), int(content[2])
            if st <= et:
                if node in ground_truth_range.keys():
                    ground_truth_range[node].append((st, et))
                else:
                    ground_truth_range[node] = [(st, et)]
                for t in range(st, et + 1):
                    ground_truth.add((node, t))
        self.ground_truth = ground_truth
        self.ground_truth_range = ground_truth_range

        filtered_ground_truth = set()
        filtered_ground_truth_range = {}
        with open(gt_file, 'r') as f:
            filtered_gt = [line.strip('\n').split(',') for line in f.readlines()][self.traj_len:]
        for content in filtered_gt:
            node, st, et = int(content[0]), int(content[1]), int(content[2])
            if st <= et and node in cam_id_list:
                if node in filtered_ground_truth_range.keys():
                    filtered_ground_truth_range[node].append((st, et))
                else:
                    filtered_ground_truth_range[node] = [(st, et)]
                for t in range(st, et + 1):
                    filtered_ground_truth.add((node, t))
        self.filtered_ground_truth = filtered_ground_truth
        self.filtered_ground_truth_range = filtered_ground_truth_range

        self.gt_num = len(gt)
    
    # load frame range of trajectory gt
    def load_traj_gt_frame_range(self, time_range_file):
        time_range_dict = {}
        with open(time_range_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line[:-1].split(',')
                time_range_dict[int(line[0])] = [float(line[1]), float(line[2])]
        return time_range_dict

    # load camera node dictionary
    def load_cam_node_dict(self, cam_node_file):
        cam_node_dict = {}
        if self.dataset == 'cityflow':
            cam_node_dict = self.load_cam_node_dict_cityflow(cam_node_file)
        if self.dataset == 'uv' or self.dataset == 'uv-z':
            cam_node_dict = self.load_cam_node_dict_mmvc(cam_node_file)
        if self.dataset == 'carla':
            cam_node_dict = self.load_cam_node_dict_carla(cam_node_file)
        if self.dataset == 'video-sg':
            cam_node_dict = self.load_cam_node_dict_videosg(cam_node_file)
        if not cam_node_dict:
            print('the camera node dictionary is null!')
        return cam_node_dict

    def load_cam_node_dict_cityflow(self, filename):
        cam_node_dict = {}
        with open(filename, "r") as f:
            for line in f.readlines():
                line = line[:-1].split(" ")
                node_id = int(line[0])
                lon = float(line[1])
                lat = float(line[2])
                cam_node_dict[node_id] = [lon, lat]

        self.cam_node_dict = cam_node_dict
        return cam_node_dict

    def load_cam_node_dict_mmvc(self, filename):
        cameras=[]
        with open(filename) as file:
            for l in file:
                cameras.append(json.loads(l))

        cam_node_dict = {}
        for cam_info in cameras:
            cam_node_dict[cam_info['camera_id']] = cam_info['position']
        
        self.cam_node_dict = cam_node_dict
        return cam_node_dict

    def load_cam_node_dict_carla(self, filename):
        cam_node_dict = {}

        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line == '\n':
                    continue
                line = line[:-1].split(' ')
                node_id = int(line[0])
                lat = float(line[1])
                lon = float(line[2])
                cam_node_dict[node_id] = [lon, lat]
        
        self.cam_node_dict = cam_node_dict
        return cam_node_dict

    def load_cam_node_dict_videosg(self, filename):
        cam_node_dict = {}

        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line == '\n':
                    continue
                line = line[:-1].split(' ')
                node_id = int(line[0])
                lon = float(line[1])
                lat = float(line[2])
                cam_node_dict[node_id] = [lon, lat]
        
        self.cam_node_dict = cam_node_dict
        return cam_node_dict

    def load_dist_file(self, filename):
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


    def filter_cands(self, I, D, car_id, down_sample_fps):
        frame_range = []
        if self.time_range_dict and car_id in self.time_range_dict.keys():
            frame_range = self.time_range_dict[car_id]
        
        cut_flag = False
        if self.dataset == 'carla':
            cut_flag = True
            max_fps = self.fps
            fps_list = list(range(down_sample_fps))
        if self.dataset == 'video-sg':
            cut_flag = True
            max_fps = self.fps
            fps_list = list(range(down_sample_fps))
        
        filtered_I = []
        filtered_D = []
        for idx, d in zip(I, D):
            if idx == -1:
                continue
            
            frame, camid, _ = get_candidate_info_by_index(self.node_feats_path, 
                                                          idx, 
                                                          self.folder_name, 
                                                          self.frame_tpye_byte_num, 
                                                          self.app_feat_partition_file)

            if frame_range:
                if frame < frame_range[0] or frame > frame_range[1]:
                    continue

            if cut_flag:
                if frame > self.video_time * max_fps * 60:
                    continue

            if cut_flag:
                if frame % max_fps not in fps_list:
                    continue

            if cut_flag and self.cam_id_list:
                if camid not in self.cam_id_list:
                    continue

            if self.filtered_ground_truth:
                if (camid, frame) in self.filtered_ground_truth:
                    continue
            
            filtered_I.append(idx)
            filtered_D.append(d)
            
        return filtered_I, filtered_D

    def find_top_k(self, 
                   save_dirs, 
                   D, 
                   I, 
                   index_to_unified_records_json_order_dict=None, 
                   unified_records_json_order_to_plate_index_dict=None, 
                   topk_cand_save=False):
        print("Temporal clustering for candidate snapshots")
        
        cand_rank = 0
        top_k = Topk(self.k, self.time_gap, self.feat_dim)
        time_find_topk = time.time()
        
         # step 1: bucket sorting - group candidate snapshots by camera ID
        camera_buckets = defaultdict(list)
        for dist, idx in zip(D, I):
            if idx == -1:
                continue
            
            frame, camid, idx_in_frame = get_candidate_info_by_index(
                self.node_feats_path, idx, self.folder_name, self.frame_tpye_byte_num, self.app_feat_partition_file
            )
            # create a Candidate object and add it to the corresponding camera bucket
            temp_cand = Candidate(idx, frame, camid, dist, cand_rank, idx_in_frame, 
                                self.node_feats_path, self.feat_dim, self.app_feat_partition_file,
                                self.plate_feat_partition_file)
            cand_rank += 1
            camera_buckets[camid].append(temp_cand)
        
        # step 2: traverse each camera bucket and sort candidates within each bucket by timestamp
        for camid, candidates in camera_buckets.items():
            # sort candidate snapshots by frame (timestamp)
            candidates.sort(key=lambda x: x.get_frame())  # x.get_frame() returns frame

            # step 3: cluster based on time gap
            curr_camera = Camera(camid)
            last_frame = None
            for candidate in candidates:
                frame = candidate.get_frame()

                if last_frame is None or (frame - last_frame) <= self.time_gap:
                    # if it’s the first candidate snapshot or within the time gap, add to the current Camera cluster
                    curr_camera.add_candidate(candidate)
                else:
                    # if time gap exceeded, end the current Camera cluster and add it to top_k
                    top_k.add_camera(curr_camera)
                    
                    # create a new Camera cluster and add the current candidate snapshot
                    curr_camera = Camera(camid)
                    curr_camera.add_candidate(candidate)
                
                # update the last processed timestamp
                last_frame = frame
            
            # add the last Camera cluster to top_k
            top_k.add_camera(curr_camera)
            
        # add the top-k plate information
        if len(top_k.camera_list) != 0:
            if (index_to_unified_records_json_order_dict and 
                unified_records_json_order_to_plate_index_dict):
                for cam in top_k.camera_list:
                    for cand in cam.candidate_list:
                        feature_raw_idx = cand.get_feature_raw_index()
                        unified_records_json_order_idx = index_to_unified_records_json_order_dict[feature_raw_idx]
                        if unified_records_json_order_idx in unified_records_json_order_to_plate_index_dict.keys():
                            plate_feature_raw_idx = unified_records_json_order_to_plate_index_dict[unified_records_json_order_idx]
                            cand.add_plate_feature(plate_feature_raw_idx, self.plate_feat_partition_file)
            top_k.camera_sort_info()
        
        time_find_topk = time.time() - time_find_topk
        self.time_find_topk = time_find_topk
        self.time_find_topk_all += time_find_topk

        # save the top_k information
        if len(top_k.camera_list) != 0 and not topk_cand_save:
            top_k.save_top_k_info(save_dirs)
            top_k.save_avg_features(save_dirs, self.folder_name)

        return top_k, time_find_topk

    def extract_topk_cand(self, top_k, query_id, car_id, query_feature, query_plate_feature, 
                            query_plate_text, topk_cand_path, folder_name):
        # export topk candidates and the corresponding features
        data = {}
        data['query_id'] = query_id
        data['car_id'] = car_id
        query_feature = np.squeeze(query_feature)
        data['query_feat'] = query_feature.tolist()
        if query_plate_feature is not None:
            data['query_plate_feat'] = query_plate_feature.tolist()
        else:
            data['query_plate_feat'] = None
        if query_plate_text is not None:
            data['query_plate_text'] = query_plate_text
        else:
            data['query_plate_text'] = None
        data['top_k_precision'] = self.top_k_precision
        data['top_k_recall'] = self.top_k_recall
        data['top_k_f1'] = self.top_k_f1
        
        top_k.camera_sort_info()
        gt = self.ground_truth_range
        candidates = []
        for camera in top_k.camera_list:
            for cand in camera.candidate_list:
                cand_dict = {}
                cand_dict['node_id'] = cand.nodeid
                cand_dict['idx_in_top_k'] = camera.idx_in_topk
                cand_dict['app_feat_dist'] = float(np.linalg.norm(cand.get_feature(self.folder_name) - query_feature))
                cand_dict['frame'] = int(cand.frame)
                cand_dict['dist'] = cand.dist
                cand_dict['rank'] = cand.rank
                cand_dict['feature'] = cand.get_feature(self.folder_name).tolist()
                if cand.get_plate_feature(self.folder_name) is not None:
                    cand_dict['plate_feature'] = cand.get_plate_feature(self.folder_name).tolist()
                else:
                    cand_dict['plate_feature'] = None
                if cand.get_plate_text() is not  None:
                    cand_dict['plate_text'] = cand.get_plate_text()
                else:
                    cand_dict['plate_text'] = None
                
                # get ground truth label
                node_id = cand_dict['node_id']
                frame = cand_dict['frame']
                label = any(
                    node_id == node and any(tm_range[0] <= frame <= tm_range[1] for tm_range in gt[node])
                    for node in gt
                )
                cand_dict['label'] = label             
                candidates.append(cand_dict)

        candidates = sorted(candidates, key=lambda x: x['app_feat_dist'])
        for i, cand in enumerate(candidates):
            cand['app_feat_dist_rank'] = i
        data['candidates'] = candidates
        
        output_path = os.path.join(topk_cand_path, self.dataset, folder_name, f"top_{self.k}")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        class JsonEncoder(json.JSONEncoder):
            """Convert numpy classes to JSON serializable objects."""
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating, np.bool_)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return super(JsonEncoder, self).default(obj)
        with open(os.path.join(output_path, '{}.json'.format(car_id)), 'w', encoding='utf8') as f:
            json.dump(data, f, ensure_ascii=False, cls=JsonEncoder)
        
        return

    def save_top_k_img(self, dirs, top_k, snapshot_path):
        print("Save top-k snapshots in the folder for testing")
        for cam in top_k.camera_list:
            for cand in cam.candidate_list:
                feat_raw_idx = cand.get_feature_raw_index()
                file_id, new_index = find_new_index(self.node_feats_path, feat_raw_idx, self.folder_name, 'partition.txt')

                curr_cam_snapshot_path = os.path.join(snapshot_path, str(file_id))
                snapshot_list = sorted(os.listdir(curr_cam_snapshot_path))
                curr_snapshot = snapshot_list[new_index]
                if not os.path.exists(os.path.join(dirs, str(file_id))):
                    os.makedirs(os.path.join(dirs, str(file_id)))
                shutil.copy(os.path.join(curr_cam_snapshot_path, curr_snapshot), os.path.join(dirs, str(file_id), curr_snapshot))

    def save_vel_distrib(self, 
                         save_dirs, 
                         time_diff_arr, 
                         dist_arr, 
                         velocities, 
                         effective_time_diff,
                         effective_dist, 
                         effective_velocities, 
                         vel_median, 
                         vel_lower_bound, 
                         vel_upper_bound, 
                         vel_std, 
                         kde_all_pairs, 
                         kde_effect_pairs):
        vel_distrib_file = os.path.join(save_dirs, 'vel_distrib_info.pkl')
        spatio_temporal_distribution = [
            time_diff_arr,
            dist_arr,
            velocities,
            effective_time_diff,
            effective_dist,
            effective_velocities,
            vel_median,
            vel_lower_bound,
            vel_upper_bound,
            vel_std,
            kde_all_pairs,
            kde_effect_pairs
        ]
        pickle.dump(spatio_temporal_distribution, open(vel_distrib_file, "wb"))
        print('save velocity distribution information successfully!')
        return
    
    def get_query_snapshot_joint_sim(self, top_k):
        time_query_snapshot_joint_sim = time.time()

        # get similarity list
        dist_lists = [cam.get_cand_dist_list() for cam in top_k.camera_list]
        snapshot_weight_arr = np.array([
            1.0 if 1.0 in dist_list else np.mean(dist_list) 
            for dist_list in dist_lists
        ])
        
        # generate the node weights according to appearing times
        node_appear_times_list = [len(cam.candidate_list) for cam in top_k.camera_list]
        # node_weight_list = cal_appear_weight(node_appear_times_list)
        node_weight_list = sigmoid_arr(np.array(node_appear_times_list))

        snapshot_weight_arr *= node_weight_list

        time_query_snapshot_joint_sim = time.time() - time_query_snapshot_joint_sim
        self.time_query_snapshot_joint_sim = time_query_snapshot_joint_sim
        self.time_build_graph_all += time_query_snapshot_joint_sim
        return snapshot_weight_arr, time_query_snapshot_joint_sim
    
    def haversine(self, lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return c * 6367 * 1000

    def calc_dist_mat(self, top_k_camera_list):
        cam_num = len(top_k_camera_list)
        dist_arr = np.zeros((cam_num, cam_num), dtype=float)
        for i in range(cam_num):
            for j in range(cam_num):
                if i == j:
                    continue
                cam_id1, cam_id2 = top_k_camera_list[i].cam_id, top_k_camera_list[j].cam_id
                if cam_id1 == cam_id2:
                    continue
                
                if self.dataset == 'cityflow' or self.dataset == 'uv' or self.dataset == 'uv-z' or self.dataset == 'video-sg':
                    dist_arr[i, j] = self.edge_length[(cam_id1, cam_id2)][0]
                elif self.dataset == 'carla':
                    node1_id, node2_id = self.cid_to_rid_dict[cam_id1], self.cid_to_rid_dict[cam_id2]
                    if node1_id != node2_id: # if the road network is sparser than the camera network
                        dist_arr[i, j] = self.edge_length[(node1_id, node2_id)][0]
                    else:
                        cam_id1_lon, cam_id1_lat = self.cam_node_dict[cam_id1][0], self.cam_node_dict[cam_id1][1]
                        cam_id2_lon, cam_id2_lat = self.cam_node_dict[cam_id2][0], self.cam_node_dict[cam_id2][1]
                        dist_arr[i, j] = self.haversine(cam_id1_lon, cam_id1_lat, cam_id2_lon, cam_id2_lat)
        return dist_arr
    
    def calc_time_diff(self, timestamps):
        cam_num = len(timestamps)
        time_diff_arr = np.zeros((cam_num, cam_num))
        for i in range(cam_num):
            for j in range(cam_num):
                if i == j:
                    continue
                time_diff_arr[i, j] = (timestamps[i][0] + timestamps[i][1]) / 2 - (timestamps[j][0] + timestamps[j][1]) / 2
        return np.abs(time_diff_arr)

    def calc_snapshot_pairs_velocity(self, time_diff_arr, distances):
        # calculate the velocity between snapshot pairs
        velocities = np.zeros_like(time_diff_arr)
        non_zero_mask = time_diff_arr != 0
        velocities[non_zero_mask] = distances[non_zero_mask] / time_diff_arr[non_zero_mask] * self.fps

        # mask out velocities where i == j (self-comparison)
        np.fill_diagonal(velocities, 0)

        return velocities
    
    def estimate_kde(self, time_diff, dist):
        """Estimate KDE for time_diff and dist."""
        try:
            return fastkde.pdf(time_diff, dist, num_points_per_sigma=20)
        except Exception as e:
            print(f"An error occurred in KDE estimation: {e}")
            return None
        
    def est_vehicle_velocity_distrib(self, top_k, snapshot_weight_arr, topk_filtering_coef, vel_range_coeff):
        print("Estimate vehicle velocity distribution")
        time_est_velocity = time.time()

        # Step 1: Extract necessary data from top_k cameras
        top_k_camera_list = top_k.camera_list
        confident_top_k_idx = np.argsort(snapshot_weight_arr)[-int(len(snapshot_weight_arr) / topk_filtering_coef):]

        # Step 2: Calculate timestamp difference and distance matrix for all top-k nodes
        timestamps = np.array([[cam.candidate_list[0].frame, cam.candidate_list[-1].frame] for cam in top_k_camera_list])
        time_diff_arr = self.calc_time_diff(timestamps)
        dist_arr = self.calc_dist_mat(top_k_camera_list)
        # Step 3: Calculate velocities between snapshots
        velocities = self.calc_snapshot_pairs_velocity(time_diff_arr, dist_arr)

        # Step 4: Extract confident velocities using confident_top_k_idx for both rows and columns
        confident_time_diff_arr = time_diff_arr[np.ix_(confident_top_k_idx, confident_top_k_idx)]
        confident_dist_arr = dist_arr[np.ix_(confident_top_k_idx, confident_top_k_idx)]
        confident_velocities = velocities[np.ix_(confident_top_k_idx, confident_top_k_idx)]

        # Step 5: Filter out zero velocities for confidence calculation
        valid_vel_mask = confident_velocities != 0
        effective_time_diff = confident_time_diff_arr[valid_vel_mask]
        effective_dist = confident_dist_arr[valid_vel_mask]
        effective_velocities = confident_velocities[valid_vel_mask]

        if effective_velocities.size == 0:
            time_est_velocity = time.time() - time_est_velocity
            self.time_est_velocity = time_est_velocity
            self.time_build_graph_all += time_est_velocity
            return (time_diff_arr, 
                    dist_arr, 
                    velocities, 
                    effective_time_diff,
                    effective_dist, 
                    effective_velocities, 
                    None, 
                    None, 
                    None, 
                    None, 
                    None, 
                    None, 
                    time_est_velocity)
        
        # Step 6: Calculate velocity statistics
        vel_median = np.median(effective_velocities)
        vel_std = np.std(effective_velocities)
        print('velocity standard deviation: {}'.format(vel_std))

        # Step 7: Define bounds and filter abnormal velocities
        lower_bound, upper_bound = 0, vel_median + vel_range_coeff * vel_std # can only constrain the upper bound of estimated speed based on the shortest path
        valid_vel_mask = (lower_bound <= effective_velocities) & (effective_velocities <= upper_bound)

        effective_time_diff = effective_time_diff[valid_vel_mask]
        effective_dist = effective_dist[valid_vel_mask]
        effective_velocities = effective_velocities[valid_vel_mask]
        effect_vel_std = np.std(effective_velocities)

        # Step 8: Generate KDE for all node pairs and effective pairs
        kde_all_pairs = self.estimate_kde(time_diff_arr.flatten(), dist_arr.flatten())
        kde_effect_pairs = self.estimate_kde(effective_time_diff, effective_dist)

        # Step 9: Record and return elapsed time
        time_est_velocity = time.time() - time_est_velocity
        self.time_est_velocity = time_est_velocity
        self.time_build_graph_all += time_est_velocity

        return (time_diff_arr, 
                dist_arr, 
                velocities, 
                effective_time_diff,
                effective_dist, 
                effective_velocities, 
                vel_median, 
                lower_bound, 
                upper_bound, 
                effect_vel_std, 
                kde_all_pairs, 
                kde_effect_pairs, 
                time_est_velocity,
        )

    def construct_motion_prob_graph(self, 
                                    time_diff_arr, 
                                    dist_arr, 
                                    velocities, 
                                    vel_lower_bound, 
                                    vel_upper_bound, 
                                    kde_all_pairs, 
                                    kde_effect_pairs,
                                    topk_filtering_coef,
                                    save_dirs):
        def compute_cdf_interpolator(method='cubic', kde_pdf=None):
            """Computes a CDF interpolator function for given time and distance arrays using fastKDE."""
            time_grid, dist_grid, density_values = kde_pdf.coords['var0'].values, kde_pdf.coords['var1'].values, kde_pdf.values

            # compute the cumulative distribution function (CDF)
            cdf_x = cumtrapz(density_values, dist_grid, initial=0, axis=0)
            cdf_2d = cumtrapz(cdf_x, time_grid, initial=0, axis=1)
            cdf_2d /= cdf_2d[-1, -1]  # Normalize to ensure CDF goes from 0 to 1

            # create RegularGridInterpolator for CDF with cubic interpolation
            return RegularGridInterpolator((dist_grid, time_grid), cdf_2d, method=method, bounds_error=False, fill_value=None)
        
        def compute_probabilities(cdf_interp_func, dist_arr, time_diff_arr, delta_dist=0.5, delta_time=0.5):
            """Computes probabilities by integrating over a specified interval using CDF interpolator."""
            upper_bound = np.column_stack([dist_arr + delta_dist, time_diff_arr + delta_time])
            lower_bound = np.column_stack([dist_arr - delta_dist, time_diff_arr - delta_time])
            
            probabilities = np.clip(cdf_interp_func(upper_bound), 0, 1) - np.clip(cdf_interp_func(lower_bound), 0, 1)
            return np.clip(probabilities, np.finfo(float).eps, 1)

        time_construct_motion_prob_graph = time.time()

        motion_prob_graph = np.zeros_like(velocities, dtype=float)
        valid_mask = (velocities != 0) & (velocities >= vel_lower_bound) & (velocities <= vel_upper_bound)

        if kde_all_pairs is not None and kde_effect_pairs is not None:
            # get the valid value
            valid_time_diff_arr = time_diff_arr[valid_mask]
            valid_dist_arr = dist_arr[valid_mask]

            # compute CDF interpolator for effective graph node pairs
            cdf_interp_func_effective = compute_cdf_interpolator('linear', kde_effect_pairs)
            effective_probabilities = compute_probabilities(cdf_interp_func_effective, valid_dist_arr, valid_time_diff_arr)

            # compute CDF interpolator for all graph node pairs
            cdf_interp_func_all = compute_cdf_interpolator('linear', kde_all_pairs)
            probabilities = compute_probabilities(cdf_interp_func_all, valid_dist_arr, valid_time_diff_arr)

            # calculate motion probability graph
            motion_prob_graph[valid_mask] = np.clip(effective_probabilities / probabilities / topk_filtering_coef ** 2, 0, 1)
        else:
            motion_prob_graph[valid_mask] = 1
        
        time_construct_motion_prob_graph = time.time() - time_construct_motion_prob_graph
        self.time_construct_motion_prob_graph = time_construct_motion_prob_graph
        self.time_build_graph_all += time_construct_motion_prob_graph

        # save association graph info
        self.save_prob_graph_info(save_dirs, motion_prob_graph)

        return motion_prob_graph, time_construct_motion_prob_graph
    
    def power_based_weight_update(self, w, strengthen_w, k):
        return w + (1 - w) * (strengthen_w ** k)

    def graph_completion(self, 
                         top_k, 
                         motion_prob_graph, 
                         merge_time_gap):
        print("complete graph nodes")
        time_graph_completion = time.time()

        # parameter setting
        cam_path_neighbor_dict = self.cam_path_neighbor_dict
        cam_path_sec_ratio = self.cam_path_sec_ratio

        # generate undirected snapshot graph
        snapshot_graph = nx.Graph()
        # add topk nodes
        snapshot_graph.add_nodes_from((camera, {"id": i}) for i, camera in enumerate(top_k.camera_list))
        # add edge weight from motion_prob_graph
        for i, camera1 in enumerate(top_k.camera_list): 
            for j, camera2 in enumerate(top_k.camera_list):
                if motion_prob_graph[i, j] == 0.:
                    continue
                snapshot_graph.add_edge(camera1, camera2, edge=(i, j), weight=motion_prob_graph[i, j])

        # generate original completed graph nodes
        cam_frames = {camera: (camera.candidate_list[0].frame + camera.candidate_list[-1].frame) / 2 for camera in top_k.camera_list}
        graph_completion_node_list = []
        graph_completion_node_dict = {}
        for camera1 in top_k.camera_list:
            cam1_id = camera1.cam_id
            cam1_frame = cam_frames[camera1]
            for camera2 in top_k.camera_list:
                cam2_id = camera2.cam_id
                if (cam1_id, cam2_id) not in cam_path_neighbor_dict or (cam1_id, cam2_id) not in cam_path_sec_ratio or not snapshot_graph.has_edge(camera1, camera2):
                    continue

                cam2_frame = cam_frames[camera2]
                neighbor_cam_list = cam_path_neighbor_dict[(cam1_id, cam2_id)]
                sec_ratio = cam_path_sec_ratio[(cam1_id, cam2_id)]
                
                # compute the timestamp corresponding to node id
                total_frame = abs(cam2_frame - cam1_frame)
                frame_list = [round(i * total_frame + cam1_frame) for i in sec_ratio]
                completed_cam_traj = list(zip(neighbor_cam_list, frame_list[1:-1]))

                graph_completion_node_list.extend(completed_cam_traj)
                graph_completion_node_dict[(camera1, camera2)] = completed_cam_traj

        ## find the points that need to be added and the points that need to be merged
        graph_completion_node_cam_id_frame_dict = defaultdict(list)
        for cam_id, frame in graph_completion_node_list:
            graph_completion_node_cam_id_frame_dict[cam_id].append(frame)
        # sort timestamps for each cam_id
        for cam_id in graph_completion_node_cam_id_frame_dict:
            graph_completion_node_cam_id_frame_dict[cam_id].sort()
        
        # group existing nodes by cam_id for easier range checks
        existing_node_list = [(camera.cam_id, camera.candidate_list[0].frame, camera.candidate_list[-1].frame)
                                for camera in top_k.camera_list]
        
        # filter the nodes that need to be added
        merge_node_set = {
            (cam_id, frame) for cam_id, frame_list in graph_completion_node_cam_id_frame_dict.items()
            for frame in frame_list
            for existing_cam_id, frame_st, frame_end in existing_node_list
            if cam_id == existing_cam_id and frame_st - merge_time_gap <= frame <= frame_end + merge_time_gap
        }
        print(f'there are total {len(merge_node_set)} merge nodes')
        
        # generate the mapping relation from new generated nodes to top-k nodes
        merge_mapping_node_dict = {} # from original node to topk index
        for merge_node_cam_id, merge_node_frame in merge_node_set:
            for (cam_id, frame_st, frame_end), camera in zip(existing_node_list, top_k.camera_list):
                if merge_node_cam_id != cam_id:
                    continue

                if frame_st - merge_time_gap <= merge_node_frame <= frame_end + merge_time_gap:
                    merge_mapping_node_dict[(cam_id, merge_node_frame)] = camera.get_idx_in_top_k()

        # update the graph edges
        def update_graph_edge(snapshot_graph, top_k_node1, top_k_node2, camera1, camera2, strengthen_factor=0.5):
            edge_data = snapshot_graph.get_edge_data(top_k_node1, top_k_node2)
            if edge_data:  # strengthen the weight of an existing edge
                weight = edge_data['weight']
                strengthen_values = snapshot_graph.edges[(camera1, camera2)]['weight'] * strengthen_factor
                snapshot_graph.edges[(top_k_node1, top_k_node2)]['weight'] = self.power_based_weight_update(weight, strengthen_values, 2)
            else:  # create a new edge
                weight = snapshot_graph.edges[(camera1, camera2)]['weight'] * strengthen_factor
                snapshot_graph.add_edge(top_k_node1, top_k_node2, edge=(top_k_node1, top_k_node2), weight=weight)
        
        for (camera1, camera2), completed_cam_traj in graph_completion_node_dict.items():
            cam1_id, frame1 = completed_cam_traj[0]
            if (cam1_id, frame1) in merge_mapping_node_dict.keys():
                idx2_in_topk = merge_mapping_node_dict[(cam1_id, frame1)]
                top_k_node2 = top_k.camera_list[idx2_in_topk]
                update_graph_edge(snapshot_graph, camera1, top_k_node2, camera1, camera2)
            cam2_id, frame2 = completed_cam_traj[-1]
            if (cam2_id, frame2) in merge_mapping_node_dict.keys():
                idx1_in_topk = merge_mapping_node_dict[(cam2_id, frame2)]
                top_k_node1 = top_k.camera_list[idx1_in_topk]
                update_graph_edge(snapshot_graph, top_k_node1, camera2, camera1, camera2)
                
            for (cam1_id, frame1), (cam2_id, frame2) in zip(completed_cam_traj, completed_cam_traj[1:]):
                # both nodes corresponding to the edge need to be preserved
                if (cam1_id, frame1) not in merge_mapping_node_dict.keys() or (cam2_id, frame2) not in merge_mapping_node_dict.keys():
                    continue

                idx1_in_topk = merge_mapping_node_dict[(cam1_id, frame1)]
                idx2_in_topk = merge_mapping_node_dict[(cam2_id, frame2)]
                top_k_node1 = top_k.camera_list[idx1_in_topk]
                top_k_node2 = top_k.camera_list[idx2_in_topk]
                update_graph_edge(snapshot_graph, top_k_node1, top_k_node2, camera1, camera2)
 
        # extract the updated motion_prob_graph
        weighted_adj_matrix = nx.adjacency_matrix(snapshot_graph, weight='weight')
        motion_prob_graph = weighted_adj_matrix.toarray()

        time_graph_completion = time.time() - time_graph_completion
        self.time_graph_completion = time_graph_completion
        self.time_build_graph_all += time_graph_completion

        return top_k, motion_prob_graph
       
    def sort_nodes_by_time(self, top_k):
        all_orig_nodes = [(cam.cam_id, int(np.mean(cam.get_timestamps()))) for cam in top_k.camera_list] # take the mean candidate timestamp as the identified timestamp
        sorted_with_idx = sorted(enumerate(all_orig_nodes), key=lambda x: x[1][1])
        sorted_idx = [elem[0] for elem in sorted_with_idx]
        all_nodes = sorted(all_orig_nodes, key=lambda x: x[1])
    
        return all_orig_nodes, sorted_idx, all_nodes
    
    def build_association_graph(self,
                                cam_num, 
                                motion_prob_graph, 
                                sorted_idx, 
                                all_orig_nodes, 
                                edge_weight_thres,
                                save_dirs):
        print("Build association graph")
        time_build_association_graph = time.time()
        
        # generate trajectory weight graph
        sorted_idx_dict = {val: idx for idx, val in enumerate(sorted_idx)}
        motion_prob_graph_dict = defaultdict(list)
        motion_prob_graph_reverse_dict = defaultdict(list)
        for key in range(cam_num):
            motion_prob_graph_dict[key]
            motion_prob_graph_reverse_dict[key]
        for u, row in enumerate(motion_prob_graph): # traverse the rows and columns of the adjacency matrix
            u_sorted_idx = sorted_idx_dict[u]
            for v, value in enumerate(row):
                v_sorted_idx = sorted_idx_dict[v]
                if not value or u_sorted_idx >= v_sorted_idx: # make sure that the graph edge connections follow timestamp order
                    continue

                motion_prob_graph_dict[u].append(v)
                motion_prob_graph_reverse_dict[v].append(u)
        edge_freq_dict1 = calc_freq_graph_flow_split(motion_prob_graph_dict)
        edge_freq_dict2 = calc_freq_graph_flow_split(motion_prob_graph_reverse_dict)
        edge_freq_dict = { (node1, node2): edge_freq_dict1.get((node1, node2), 0) + freq for (node2, node1), freq in edge_freq_dict2.items() }

        # normalize each edge flow
        avg_flow = sum(edge_freq_dict.values()) / len(edge_freq_dict)
        normalized_edge_freq_dict = {edge: flow / avg_flow for edge, flow in edge_freq_dict.items()}

        # generate the final weight graph
        association_graph = {}
        incoming_node_dict = defaultdict(list)
        for (i, j), freq in normalized_edge_freq_dict.items():
            edge = (all_orig_nodes[i][0], all_orig_nodes[i][1], all_orig_nodes[j][0], all_orig_nodes[j][1])
            # traj_weight = node_weight(freq)
            # traj_weight = log_function(freq)
            traj_weight = sigmoid(freq)
            edge_weight = traj_weight * motion_prob_graph[i, j] # fusion motion probability, trajectory weight
            if edge_weight > edge_weight_thres: # filter out some edges to form a sparse graph
                association_graph[edge] = edge_weight
                
                prev_node = (all_orig_nodes[i][0], all_orig_nodes[i][1])
                succ_node = (all_orig_nodes[j][0], all_orig_nodes[j][1])
                incoming_node_dict[succ_node].append(prev_node)

        time_build_association_graph = time.time() - time_build_association_graph
        self.time_build_association_graph = time_build_association_graph
        self.time_build_graph_all += time_build_association_graph

        # save association graph info
        self.save_association_graph_info(save_dirs, association_graph)

        return association_graph, incoming_node_dict
    
    def convert_association_graph(self,
                                cam_num, 
                                motion_prob_graph, 
                                sorted_idx, 
                                all_orig_nodes, 
                                edge_weight_thres,
                                save_dirs):
        print("Build association graph")
        time_build_association_graph = time.time()
        
        # generate trajectory weight graph
        sorted_idx_dict = {val: idx for idx, val in enumerate(sorted_idx)}
        motion_prob_graph_dict = defaultdict(list)
        for key in range(cam_num):
            motion_prob_graph_dict[key]
        for u, row in enumerate(motion_prob_graph): # traverse the rows and columns of the adjacency matrix
            u_sorted_idx = sorted_idx_dict[u]
            for v, value in enumerate(row):
                v_sorted_idx = sorted_idx_dict[v]
                if not value or u_sorted_idx >= v_sorted_idx: # make sure that the graph edge connections follow timestamp order
                    continue
                motion_prob_graph_dict[u].append(v)

        # generate the final weight graph
        association_graph = {}
        incoming_node_dict = defaultdict(list)
        for i, j_list in motion_prob_graph_dict.items():
            for j in j_list:
                edge = (all_orig_nodes[i][0], all_orig_nodes[i][1], all_orig_nodes[j][0], all_orig_nodes[j][1])
                if edge not in association_graph.keys():
                    edge_weight = motion_prob_graph[i, j] # fusion motion probability, trajectory weight
                    if edge_weight > edge_weight_thres: # filter out some edges to form a sparse graph
                        association_graph[edge] = edge_weight
                
                prev_node = (all_orig_nodes[i][0], all_orig_nodes[i][1])
                succ_node = (all_orig_nodes[j][0], all_orig_nodes[j][1])
                incoming_node_dict[succ_node].append(prev_node)

        time_build_association_graph = time.time() - time_build_association_graph
        self.time_build_association_graph = time_build_association_graph
        self.time_build_graph_all += time_build_association_graph

        # save association graph info
        self.save_association_graph_info(save_dirs, association_graph)

        return association_graph, incoming_node_dict

    def degrees_to_radians(self, degrees):
        return degrees * math.pi / 180

    def direction_vector_wgs84(self, cam_node1, cam_node2):
        lon1, lat1 = self.cam_node_dict[cam_node1]
        lon2, lat2 = self.cam_node_dict[cam_node2]
        # convert latitude and longitude to radians
        lat1_rad = self.degrees_to_radians(lat1)
        lon1_rad = self.degrees_to_radians(lon1)
        lat2_rad = self.degrees_to_radians(lat2)
        lon2_rad = self.degrees_to_radians(lon2)

        d_lat = lat2_rad - lat1_rad
        d_lon = (lon2_rad - lon1_rad) * math.cos((lat1_rad + lat2_rad) / 2)
        return (d_lon, d_lat)

    def direction_angle_cartesian(self, cam_node1, cam_node2):
        x1, y1 = self.cam_node_dict[cam_node1]
        x2, y2 = self.cam_node_dict[cam_node2]
        dx = x2 - x1
        dy = y2 - y1
        return (dx, dy)

    # define the function to detect U-turn behavior
    def is_uturn(self, node1, node2, node3):
        if self.dataset == 'uv' or self.dataset == 'uv-z':
            vec_ij = self.direction_angle_cartesian(node1, node2)
            vec_jk = self.direction_angle_cartesian(node2, node3)
        else:
            vec_ij = self.direction_vector_wgs84(node1, node2)
            vec_jk = self.direction_vector_wgs84(node2, node3)
        dot_product = vec_ij[0] * vec_jk[0] + vec_ij[1] * vec_jk[1]
        norm_ij = (vec_ij[0] ** 2 + vec_ij[1] ** 2) ** 0.5
        norm_jk = (vec_jk[0] ** 2 + vec_jk[1] ** 2) ** 0.5
        return 1 if dot_product <= -0.9 * norm_ij * norm_jk else 0
    
    def find_max_weight_path(self, 
                             car_id, 
                             sorted_idx, 
                             all_nodes, 
                             incoming_node_dict, 
                             association_graph, 
                             snapshot_weight_arr, 
                             u_turn_penalty_coeff):
        # adjust the snapshot weight order
        snapshot_weight_arr = [snapshot_weight_arr[i] for i in sorted_idx] 

        max_weight_table = {}
        max_weight_path = {}
        for idx, (curr_node, snapshot_weight) in enumerate(zip(all_nodes, snapshot_weight_arr)):
            # initialize dynamic programming state table
            max_weight_path[curr_node] = [curr_node]
            max_weight_table[curr_node] = snapshot_weight # Initialize to node weight

            if idx > 0 and curr_node in incoming_node_dict.keys():
                for pre_node in incoming_node_dict[curr_node]: # traverse the previous node weights
                    pre_edge = (pre_node[0], pre_node[1], curr_node[0], curr_node[1])
                    weight = association_graph.get(pre_edge, 0)
                    prev_path = max_weight_path[pre_node]
                    
                    if len(prev_path) > 1:
                        pre_pre_node = prev_path[-2]
                        if self.is_uturn(pre_pre_node[0], pre_node[0], curr_node[0]):
                            weight += u_turn_penalty_coeff  # apply U-turn penalty
                    
                    if pre_node not in max_weight_table:
                        print('car id {} previous node {} current node {}'.format(car_id, pre_node, curr_node))
                        new_weight_sum = weight
                        new_path = []
                    else:
                        new_weight_sum = max_weight_table[pre_node] + weight + snapshot_weight
                        new_path = prev_path + [curr_node]
                    
                    if new_weight_sum > max_weight_table[curr_node]: # get the max weight sum
                        max_weight_table[curr_node] = new_weight_sum
                        max_weight_path[curr_node] = new_path

        # find the path with the max weight sum
        max_weight_path_end_node = max(max_weight_table, key=max_weight_table.get, default=None)
        final_path = max_weight_path.get(max_weight_path_end_node, [])
        max_weight_sum = max(max_weight_table.values(), default=0)
        
        return final_path, max_weight_sum

    def final_path_extract(self, 
                           query_id, 
                           save_dirs, 
                           car_id, 
                           sorted_idx, 
                           all_nodes, 
                           incoming_node_dict, 
                           association_graph, 
                           snapshot_weight_arr, 
                           u_turn_penalty_coeff=-0.1):
        print("Extract final paths")
        time_path_extraction = time.time()

        # find the path with maximum weight sum based on dynamic programming
        final_path, max_weight_sum = self.find_max_weight_path(car_id,
                                                             sorted_idx, 
                                                             all_nodes, 
                                                             incoming_node_dict, 
                                                             association_graph, 
                                                             snapshot_weight_arr, 
                                                             u_turn_penalty_coeff)
        print(f'max weight sum of the trajectory is: {max_weight_sum}')

        time_path_extraction = time.time() - time_path_extraction
        self.time_path_extraction = time_path_extraction
        self.time_path_extraction_all += time_path_extraction

        # save the final path results
        self.save_final_path(save_dirs, query_id, final_path)
        
        return final_path, time_path_extraction
    
    def evaluate_top_k(self, top_k):
        ground_truth_range_set = {
            (node, time_range[0], time_range[1])
            for node in self.ground_truth_range
            for time_range in self.ground_truth_range[node]
        }

        # initialize hit count and removal set
        hit_num = 0
        remove_set = set()
        # iterate over camera list in top_k
        for camera in top_k.camera_list:
            avg_time = int(np.mean(camera.get_timestamps()))  # calculate the average time
            curr_gt_set = ground_truth_range_set - remove_set
            curr_gt_node_set = {x[0] for x in curr_gt_set}

            # check if the camera ID exists in the ground truth node set
            if camera.cam_id in curr_gt_node_set:
                # check if avg_time falls within any ground truth range for the current camera
                for curr_node_range in self.ground_truth_range[camera.cam_id]:
                    if curr_node_range[0] <= avg_time <= curr_node_range[1]:
                        hit_num += 1
                        remove_set.add((camera.cam_id, curr_node_range[0], curr_node_range[1]))
                        break  # stop after first hit to avoid duplicate counting

        self.top_k_precision = hit_num / len(top_k.camera_list)
        self.top_k_recall = hit_num / self.gt_num
        if self.top_k_precision + self.top_k_recall > 0:
            self.top_k_f1 = 2 * self.top_k_precision * self.top_k_recall / (self.top_k_precision + self.top_k_recall)
        else:
            self.top_k_f1 = 0

        self.all_top_k_p.append(self.top_k_precision)
        self.all_top_k_r.append(self.top_k_recall)
        self.all_top_k_f1.append(self.top_k_f1)
        self.all_top_k_t.append(self.time_app_feat_search + self.time_plate_feat_search + self.time_search_space_fusion + self.time_sim_thres_trunc + self.time_find_topk)

    def evaluate(self, path):
        ground_truth_range_set = {
            (node, time_range[0], time_range[1])
            for node in self.ground_truth_range
            for time_range in self.ground_truth_range[node]
        }

        hit_num = 0
        remove_set = set()
        for node_id, avg_time in path: # iterate over each node_id and avg_time in the path
            curr_gt_set = ground_truth_range_set - remove_set
            curr_gt_node_set = {x[0] for x in curr_gt_set}

            if node_id in curr_gt_node_set:
                for curr_node_range in self.ground_truth_range[node_id]:
                    if curr_node_range[0] <= avg_time <= curr_node_range[1]:
                        hit_num += 1
                        remove_set.add((node_id, curr_node_range[0], curr_node_range[1]))
                        break  # Exit after matching one time range
        
        # calculate metrics
        path_len = len(path) 
        precision = hit_num / path_len if path_len > 0 else 0 # calculate precision
        recall = hit_num / self.gt_num if self.gt_num > 0 else 0  # Calculate recall
        F1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0 # calculate F1 score

        # store results
        self.precision = precision
        self.recall = recall
        self.f1 = F1

        return precision, recall, F1

    def save_prob_graph_info(self, folder, motion_prob_graph):
        # create the file path for saving
        filename = os.path.join(folder, 'prob_graph.csv')

        # prepare the data, converting it into a list
        data = [
            [i, j, motion_prob_graph[i][j]]
            for i in range(len(motion_prob_graph))
            for j in range(len(motion_prob_graph[i]))
            if motion_prob_graph[i][j] != 0
        ]
        # create a DataFrame
        df = pd.DataFrame(data, columns=["row index", "column index", "edge weight"])
        # save the DataFrame to a CSV file
        df.to_csv(filename, mode='w', index=False)

    def save_association_graph_info(self, folder, association_graph):
        # create the file path for saving
        filename = os.path.join(folder, 'association_graph.csv')

        # prepare the data, converting it into a list
        data = [[cam1_id, frame1, cam2_id, frame2, edge_weight] 
                for (cam1_id, frame1, cam2_id, frame2), edge_weight in association_graph.items()]

        # create a DataFrame
        df = pd.DataFrame(data, columns=["start node", "start time", "end node", "end time", "edge weight"])
        # save the DataFrame to a CSV file
        df.to_csv(filename, mode='w', index=False)

    def save_final_path(self, save_dirs, query_id, final_path):
        # save the final recovered path
        final_path_file = os.path.join(os.path.dirname(save_dirs), "final_path.csv")
        if not os.path.exists(final_path_file):
            with open(final_path_file, 'w', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(["query", "final extracted path"])
                csv_writer.writerow([query_id, final_path])
        else:
            with open(final_path_file, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([query_id, final_path])
                
    def save_results(self, query_id, reid_time, all_time):
        # save results
        evaluation_file = os.path.join(self.output_folder, 'evaluation.csv')
        if not os.path.exists(evaluation_file):
            f = open(evaluation_file, 'w')
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                "query", 
                "precision", 
                "recall", 
                "f1", 
                "time", 
                "topk p", 
                "topk r", 
                "topk f1", 
                "topk t"
            ])
        else:
            f = open(evaluation_file, 'a')
            csv_writer = csv.writer(f)
        csv_writer.writerow([
            query_id, 
            self.precision, 
            self.recall, 
            self.f1, 
            all_time, 
            self.top_k_precision, 
            self.top_k_recall, 
            self.top_k_f1, 
            reid_time
        ])
        f.close()

    def save_time(self, query_id):
        # get time statistics
        reid_time = self.time_app_feat_search + self.time_plate_feat_search + self.time_search_space_fusion + self.time_sim_thres_trunc + self.time_find_topk
        all_time = reid_time + self.time_query_snapshot_joint_sim + self.time_est_velocity + self.time_construct_motion_prob_graph + self.time_graph_completion + self.time_build_association_graph + self.time_path_extraction
        self.all_inference_time.append(all_time)

        filename = os.path.join(self.output_folder, 'time.csv')
        if not os.path.exists(filename):
            f = open(filename, 'w')
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                "query",
                "reid",
                "estimate node weights",
                "estimate spatio-temporal distribution",
                "motion prob graph generation",
                "graph completion",
                "association graph generation",
                "path extraction",
                "sum"
            ])
        else:
            f = open(filename, 'a')
            csv_writer = csv.writer(f)
        csv_writer.writerow([
            query_id,
            reid_time,
            self.time_query_snapshot_joint_sim,
            self.time_est_velocity,
            self.time_construct_motion_prob_graph,
            self.time_graph_completion,
            self.time_build_association_graph,
            self.time_path_extraction,
            all_time
        ])
        f.close()

    def set_traj_rec_res_zero(self, query_id):
        # get time statistics
        reid_time = self.time_app_feat_search + self.time_plate_feat_search + self.time_search_space_fusion + self.time_sim_thres_trunc + self.time_find_topk
        all_time = reid_time + self.time_query_snapshot_joint_sim + self.time_est_velocity + self.time_construct_motion_prob_graph + self.time_graph_completion + self.time_build_association_graph + self.time_path_extraction

        ## set reid results
        # set top-k metrics to 0
        self.top_k_precision = 0
        self.top_k_recall = 0
        self.top_k_f1 = 0
        # append 0 to the top-k metrics lists
        self.all_top_k_p.append(0)
        self.all_top_k_r.append(0)
        self.all_top_k_f1.append(0)
        self.all_top_k_t.append(reid_time) # record the time
        
        ## set trajectory recovery results
        # set precision, recall, and F1 to 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        # append 0 to the general precision, recall, and F1 lists
        self.all_p.append(0)
        self.all_r.append(0)
        self.all_f1.append(0)
        self.all_inference_time.append(reid_time) # record the time
        
        # print the updated scores for VIVID and Top-k
        print('VIVID: ', round(self.all_p[-1], 4), round(self.all_r[-1], 4), round(self.all_f1[-1], 4), all_time)
        print('Top-k: ', round(self.all_top_k_p[-1], 4), round(self.all_top_k_r[-1], 4), round(self.all_top_k_f1[-1], 4), reid_time)
        
        # record results and time files
        self.save_results(query_id, reid_time, all_time)
        self.save_time(query_id)

    def update_traj_rec_res_from_cand_retrieve(self, query_id):
        # get time statistics
        reid_time = self.time_app_feat_search + self.time_plate_feat_search + self.time_search_space_fusion + self.time_sim_thres_trunc + self.time_find_topk
        all_time = reid_time + self.time_query_snapshot_joint_sim + self.time_est_velocity + self.time_construct_motion_prob_graph + self.time_graph_completion + self.time_build_association_graph + self.time_path_extraction

        ## set trajectory recovery results 
        # update precision, recall, and F1 score
        self.precision = self.top_k_precision
        self.recall = self.top_k_recall
        self.f1 = self.top_k_f1
        # append updated scores to corresponding lists
        self.all_p.append(self.precision)
        self.all_r.append(self.recall)
        self.all_f1.append(self.f1)
        self.all_inference_time.append(all_time)
        
        # print the updated scores for VIVID and Top-k
        print('VIVID: ', round(self.all_p[-1], 4), round(self.all_r[-1], 4), round(self.all_f1[-1], 4), all_time)
        print('Top-k: ', round(self.all_top_k_p[-1], 4), round(self.all_top_k_r[-1], 4), round(self.all_top_k_f1[-1], 4), reid_time)
        
        # record results and time files
        self.save_results(query_id, reid_time, all_time)
        self.save_time(query_id)

    def update_traj_rec_res(self, query_id, p, r, F1):
        # get time statistics
        reid_time = self.time_app_feat_search + self.time_plate_feat_search + self.time_search_space_fusion + self.time_sim_thres_trunc + self.time_find_topk
        all_time = reid_time + self.time_query_snapshot_joint_sim + self.time_est_velocity + self.time_construct_motion_prob_graph + self.time_graph_completion + self.time_build_association_graph + self.time_path_extraction

        print('VIVID: ', round(p, 4), round(r, 4), round(F1, 4), all_time)
        print('Top-k: ', round(self.all_top_k_p[-1], 4), round(self.all_top_k_r[-1], 4), round(self.all_top_k_f1[-1], 4), reid_time)
        self.all_p.append(p)
        self.all_r.append(r)
        self.all_f1.append(F1)
        self.all_inference_time.append(all_time)

        # record results and time files
        self.save_results(query_id, reid_time, all_time)
        self.save_time(query_id)

