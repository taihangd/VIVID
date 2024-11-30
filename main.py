import argparse
import yaml
from config import cfg, update_config
from traj_rec_solver import *


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='A Python Implementation of VIVID')
    # command-line config options
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    # config file
    parser.add_argument('--cfg', help='experiment configure file name',
                        default='/home/dth/research/VIVID-main/config/cityflow.yaml')

    # parse arguments and check
    args = parser.parse_args()
    update_config(cfg, args)
    print(yaml.dump(cfg.dump(), default_flow_style=False))

    ## initialize the algorithm object
    traj_rec_obj = VIVID(cfg)
    traj_rec_obj.set_road_graph_info()
    traj_rec_obj.set_cid_rid_dict()
    traj_rec_obj.set_shortest_path_results_info()
    traj_rec_obj.set_cam_path_neighbor_dict_info()
    traj_rec_obj.set_cam_path_sec_ratio_info()
    traj_rec_obj.reset_dataset_statistics()
    
    # load and generate query info
    query_features = np.load(cfg.query_feats_path + 'query_gf.npy').astype('float32')
    query_features = np.squeeze(query_features)
    query_plate_features = None
    if os.path.exists(cfg.query_feats_path + 'query_gpf.npy'):
        query_plate_features = np.load(cfg.query_feats_path + 'query_gpf.npy').astype('float32')
    query_plate_texts = None
    if os.path.exists(cfg.query_feats_path + 'query_gpt.npy'):
        query_plate_texts = np.load(cfg.query_feats_path + 'query_gpt.npy')
    index_gallery = FeatureGallery(cfg)
    
    # the index order conversion relationship and the plate text information
    if cfg.dataset == 'uv':
        app_feats_list = np.load('/mnt/data_hdd_large/dth/home/data/MMVC/mmvc_dataset/feature/records_200w/gf_feats/gf.npy')
        plate_feats_list = np.load('/mnt/data_hdd_large/dth/home/data/MMVC/mmvc_dataset/feature/records_200w/gf_feats/gpf.npy', allow_pickle=True)
        plate_text_list = np.load('/mnt/data_hdd_large/dth/home/data/MMVC/mmvc_dataset/feature/records_200w/gf_feats/gpt.npy', allow_pickle=True)
        plate_text_list = ['' if plate_text == [] else plate_text for plate_text in plate_text_list]

        # generate the mapping relationship between index and unified records json order
        index_to_unified_records_json_order_dict, unified_records_json_order_to_index_dict = index_gallery.gen_idx_id_mapping('_gf_id.npy')
        plate_index_to_unified_records_json_order_dict, unified_records_json_order_to_plate_index_dict = index_gallery.gen_idx_id_mapping('_gpf_id.npy')
    else:
        index_to_unified_records_json_order_dict = None
        unified_records_json_order_to_index_dict = None
        plate_index_to_unified_records_json_order_dict = None
        unified_records_json_order_to_plate_index_dict = None
        plate_text_list = None

    # load feature index
    if cfg.dataset == 'video-sg' or cfg.dataset == 'carla': # organize the feature index path to facilitate script invocation
        app_feats_index_path = os.path.join(cfg.app_feats_index_path, 'features.index')
    elif cfg.dataset == 'cityflow' or cfg.dataset == 'uv' or cfg.dataset == 'uv-z':
        app_feats_index_path = cfg.app_feats_index_path
    app_feats_index = faiss.read_index(app_feats_index_path)
    if cfg.use_plate and cfg.plate_feats_index_path:
        plate_feats_index = faiss.read_index(cfg.plate_feats_index_path)
    if cfg.ngpu:
        gpu_resources = faiss.StandardGpuResources()
        app_feats_index = faiss.index_cpu_to_gpu(gpu_resources, 0, app_feats_index)
        print('load appearance feature index in GPU successfully!')
        if cfg.use_plate and cfg.plate_feats_index_path:
            plate_feats_index = faiss.index_cpu_to_gpu(gpu_resources, 0, plate_feats_index)
            print('load plate feature index in GPU successfully!')


    # query process
    for query_id, query_feature in enumerate(query_features):
        car_id = np.load(cfg.query_feats_path + 'query_carid.npy')[query_id]
        if cfg.dataset == 'carla' and (car_id <= 900): # the first 900 trajectories in Carla are reserved for training
            continue
        print("======================== Query ID: %03d, Car ID: %d ========================" % (query_id, car_id))


        ## data preparation
        traj_gt_file = os.path.join(cfg.traj_gt_path, traj_rec_obj.folder_name, str(car_id) + ".txt")
        if not os.path.exists(traj_gt_file): # if the labeled snapshots don't have a corresponding GPS trajectory
            continue
        traj_rec_obj.load_traj_gt(traj_gt_file)
        
        # initialization
        traj_rec_obj.reset()
        # time recording initialization
        time_plate_feat_search = 0
        time_search_space_fusion = 0

        save_dirs = os.path.join(traj_rec_obj.output_folder, "query_%03d" % query_id)  # folder to store the outputs
        if not os.path.exists(save_dirs):
            os.mkdir(save_dirs)
        query_feature = np.array([query_features[query_id]]) # the default query vector shape is (1, d)
        faiss.normalize_L2(query_feature) # normalize the query feature
        if query_plate_features is not None:
            query_plate_feature = np.array([query_plate_features[query_id]]) # the default query vector shape is (1, d)
            faiss.normalize_L2(query_plate_feature) # normalize the query feature
        else:
            query_plate_feature = None
        if query_plate_texts is not None:
            query_plate_text = query_plate_texts[query_id]
        else:
            query_plate_text = None


        ## candidate retrieval
        D_app, I_app, time_app_feat_search = index_gallery.cal_all_features_dis(app_feats_index, query_feature, cfg.k)
        if cfg.use_plate and cfg.plate_feats_index_path: # if plate is used, the search results are merged
            D_plate, I_plate, time_plate_feat_search = index_gallery.cal_all_features_dis(plate_feats_index, query_plate_feature, cfg.k)
            D, I, time_search_space_fusion = index_gallery.fusion_search_space(query_feature, 
                                                                               query_plate_feature, 
                                                                               app_feats_list, 
                                                                               plate_feats_list, 
                                                                               D_app, 
                                                                               I_app, 
                                                                               D_plate, 
                                                                               I_plate, 
                                                                               save_dirs,
                                                                               index_to_unified_records_json_order_dict, 
                                                                               unified_records_json_order_to_index_dict, 
                                                                               plate_index_to_unified_records_json_order_dict, 
                                                                               unified_records_json_order_to_plate_index_dict)
        else:
            D, I = D_app, I_app
        D, I, time_sim_thres_trunc = index_gallery.sim_thres_trunc(D, I, cfg.sim_thres, save_dirs)
        # update time statistics
        traj_rec_obj.time_app_feat_search = time_app_feat_search
        traj_rec_obj.time_plate_feat_search = time_plate_feat_search
        traj_rec_obj.time_search_space_fusion = time_search_space_fusion
        traj_rec_obj.time_sim_thres_trunc = time_sim_thres_trunc
        traj_rec_obj.time_search_all += (time_app_feat_search + time_plate_feat_search + time_search_space_fusion + time_sim_thres_trunc)
        
        # retrieve top-k snapshots
        if cfg.dataset == 'video-sg' or cfg.dataset == 'carla' or cfg.dataset == 'uv' or cfg.dataset == 'uv-z':
            I, D = traj_rec_obj.filter_cands(I, D, car_id, traj_rec_obj.down_sample_fps)
            if cfg.dataset == 'carla' and len(I) < 500: # make sure there is enough candidates to generate topk
                print('the candidates number is less than 500!')
        top_k, _ = traj_rec_obj.find_top_k(save_dirs, 
                                           D, 
                                           I, 
                                           index_to_unified_records_json_order_dict, 
                                           unified_records_json_order_to_plate_index_dict, 
                                           cfg.topk_cand_save)
        # save topk candidates for comparative algorithm testing
        if cfg.topk_cand_save:
            traj_rec_obj.extract_topk_cand(top_k, 
                                           query_id, 
                                           car_id, 
                                           query_feature, 
                                           query_plate_feature, 
                                            query_plate_text, 
                                            cfg.topk_cand_path, 
                                            "t%02d_c%03d_len%02d" % (cfg.video_time, cfg.node_num, cfg.traj_len))
        if len(top_k.camera_list) == 0:
            print('warning: there is top-k nodes!')
            traj_rec_obj.set_traj_rec_res_zero(query_id)
            continue
        traj_rec_obj.evaluate_top_k(top_k) # evaluate snapshot retrieve results
        if cfg.topk_cand_save:
            continue
        

        ## graph construction
        # generate the graph node weights
        snapshot_weight_arr, query_snapshot_joint_sim_time = traj_rec_obj.get_query_snapshot_joint_sim(top_k)

        # estimate vehicle velocity distribution
        (time_diff_arr, 
        dist_arr, 
        velocities, 
        effective_time_diff, 
        effective_dist, 
        effective_velocities, 
        vel_median, 
        vel_lower_bound, 
        vel_upper_bound, 
        effect_vel_std, 
        kde_all_pairs, 
        kde_effect_pairs, 
        _) = traj_rec_obj.est_vehicle_velocity_distrib(top_k, 
                                                        snapshot_weight_arr, 
                                                        cfg.topk_filtering_coef, 
                                                        cfg.vel_range_coeff)
        print('the bound of velocity is: {}, {}'.format(vel_lower_bound, vel_upper_bound))
        if cfg.save_vel_distrib:
            traj_rec_obj.save_vel_distrib(save_dirs, 
                                        time_diff_arr, 
                                        dist_arr, 
                                        velocities, 
                                        effective_time_diff,
                                        effective_dist, 
                                        effective_velocities, 
                                        vel_median, 
                                        vel_lower_bound, 
                                        vel_upper_bound, 
                                        effect_vel_std, 
                                        kde_all_pairs, 
                                        kde_effect_pairs)
        if kde_all_pairs is None or kde_effect_pairs is None or effect_vel_std is None or effect_vel_std > cfg.vel_std_thres: # select the topk result as the trajectory recovery result
            print('warning: vehicle speed estimation is inaccurate!')
            traj_rec_obj.update_traj_rec_res_from_cand_retrieve(query_id)
            continue

        # generate the motion probability graph
        motion_prob_graph, _ = traj_rec_obj.construct_motion_prob_graph(time_diff_arr, 
                                                                        dist_arr,
                                                                        velocities, 
                                                                        vel_lower_bound, 
                                                                        vel_upper_bound, 
                                                                        kde_all_pairs, 
                                                                        kde_effect_pairs,
                                                                        cfg.topk_filtering_coef,
                                                                        save_dirs)
        if np.all(motion_prob_graph == 0): # select the topk result as the trajectory recovery result
            print('warning: there is a problem with the motion probability graph construction!')
            traj_rec_obj.update_traj_rec_res_from_cand_retrieve(query_id)
            continue


        ## graph enhancement
        top_k, motion_prob_graph = traj_rec_obj.graph_completion(top_k, motion_prob_graph, cfg.merge_time_gap)
        print('complete motion probability graph successfully!')
        
        # sort all the nodes by the time order
        all_orig_nodes, sorted_idx, all_nodes = traj_rec_obj.sort_nodes_by_time(top_k)
        association_graph, incoming_node_dict = traj_rec_obj.build_association_graph(len(top_k.camera_list), 
                                                                                    motion_prob_graph, 
                                                                                    sorted_idx, 
                                                                                    all_orig_nodes, 
                                                                                    cfg.edge_weight_thres,
                                                                                    save_dirs)
        print('construct association graph successfully!')


        ## extract final path
        final_path, _ = traj_rec_obj.final_path_extract(query_id, 
                                                        save_dirs, 
                                                        car_id, 
                                                        sorted_idx, 
                                                        all_nodes, 
                                                        incoming_node_dict, 
                                                        association_graph, 
                                                        snapshot_weight_arr, 
                                                        cfg.u_turn_penalty_coeff)
        if len(final_path) == 0: # select the topk result as the trajectory recovery result
            print('warning: the extracted final path is identified as inaccurate!')
            traj_rec_obj.update_traj_rec_res_from_cand_retrieve(query_id)
            continue


        # evaluation
        p, r, F1 = traj_rec_obj.evaluate(final_path)
        traj_rec_obj.update_traj_rec_res(query_id, p, r, F1)


    print("=" * 40)
    print(f"VIVID precision: {np.mean(traj_rec_obj.all_p):.4f}")
    print(f"VIVID recall   : {np.mean(traj_rec_obj.all_r):.4f}")
    print(f"VIVID F1       : {np.mean(traj_rec_obj.all_f1):.4f}")
    print(f"VIVID time     : {np.mean(traj_rec_obj.all_inference_time):.4f}")

    print("=" * 40)
    print(f'VIVID precision/recall/F1/time: {np.mean(traj_rec_obj.all_p):.4f}/{np.mean(traj_rec_obj.all_r):.4f}/{np.mean(traj_rec_obj.all_f1):.4f}/{np.mean(traj_rec_obj.all_inference_time):.4f}')
