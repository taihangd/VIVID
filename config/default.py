from yacs.config import CfgNode as CN

_C = CN()

## algorithm setting parameters, scalability analysis
_C.video_time = 100000
_C.node_num = 42
_C.traj_len = 100000

_C.fps = 10 # original fps
_C.down_sample_fps = 10 # fps after downsampling
_C.delta = 0.45

## dataset parameters
_C.dataset = "cityflow"
_C.dataset_path = "/mnt/data_hdd_large/dth/home/data/cityflow_orig_data/"
_C.traj_gt_path = "/mnt/data_hdd_large/dth/home/data/cityflow_orig_data/test_set_traj_road_network_comp_gt_range_200/"
_C.time_range_file = '/mnt/data_hdd_large/dth/home/data/MMVC/mmvc_dataset/query_time_range/time_range.txt'

_C.cam_node_file = '/mnt/data_hdd_large/dth/home/data/MMVC/cameras.json'
_C.node_feats_path = '/mnt/data_hdd_large/dth/home/data/cityflow_orig_data/feat_extract/gf_feats_bin/'
_C.orig_node_feats_path = '/mnt/data_hdd_large/dth/home/data/MMVC/mmvc_dataset/feature/records_200w/gf_feats/'
_C.query_feats_path = '/mnt/data_hdd_large/dth/home/data/cityflow_orig_data/feat_extract/query_feats/'
_C.app_feats_index_path = '/mnt/data_hdd_large/dth/home/data/MMVC/mmvc_dataset/feature/records_200w/app_feat_features.index'
_C.plate_feats_index_path = '/mnt/data_hdd_large/dth/home/data/MMVC/mmvc_dataset/feature/records_200w/plate_feat_features.index'

# index related parameters
_C.ngpu = 0 # the GPU number for index retrieval
_C.select_ratio = 10 # the select ratio for building index
_C.frame_tpye_byte_num = 8 # int format, need to be keep in line with binary format
_C.feat_dim = 3840 # feature dimension
_C.k = 5000 # faiss function search space
_C.sim_thres = 0.0

# algorithm setting parameters
_C.time_gap = 1000 # for merging adjacent nodes
_C.cam_cand_num = 10 # max number of candidates in one camera

# trajectory generation parameters
_C.topk_filtering_coef = 1.0 # the coefficient for filtering topk nodes
_C.vel_range_coeff = 1. # velocity range coefficient for standard deviation
_C.vel_std_thres = 100. # velocity standard deviation threshold
_C.neighbor_cam_num_thres = 10 # neighbor camera number threshold
_C.merge_time_gap = 100 # time range threshold for merging adjacent nodes in graph completion
_C.u_turn_penalty_coeff = -0.1 # the penalty coefficient for u-turn
_C.edge_weight_thres = 0. # the edge weight threshold for filt out some edges

# output directory
_C.output_path = "res/outputs/"

# all camera id list
_C.cam_id_list = [1, 2, 3, 4, 5, 6, 8, 9, 10, 13, 14, 17, 18, 
                    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                    32, 34, 36, 38, 39, 44, 45, 91, 92, 95, 96, 
                    106, 107, 108, 113, 114]

# only for MMVC dataset
_C.dist_file = '/mnt/data_hdd_large/dth/home/data/MMVC/mmvc_dataset/map/all_pairs_route_distance.csv'
_C.traj_gt_path = "/mnt/data_hdd_large/dth/home/data/cityflow_orig_data/test_set_traj_road_network_comp_gt_range_50/"

# cache_data about road graph and shortest path
_C.road_graph_file = "cache_data/road_graph_cityflow.pkl"
_C.shortest_path_results_file = "cache_data/shortest_path_results_cityflow.pkl"
_C.cam_path_neighbor_dict_file = "cache_data/cam_path_neighbor_dict.pkl"
_C.cam_path_sec_ratio_file = "cache_data/cam_path_sec_ratio.pkl"
_C.cid_to_rid_file = "/mnt/data_hdd_large/dth/home/data/MMVC/mmvc_dataset/cid_rid_correspondence.pkl"

# if use the plate information
_C.use_plate = False

# save topk candidates
_C.topk_cand_save = False
_C.topk_cand_path = '/home/dth/research/VIVID-main/topk_cand/'
_C.save_vel_distrib = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


def check_config(cfg):
    pass
