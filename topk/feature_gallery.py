import faiss
import random
import time
import os
import numpy as np
from common.load_feature import *
from common.calculate_function import *


class FeatureGallery(object):
    def __init__(self, cfg):
        super(FeatureGallery, self).__init__()
        self.node_num = cfg.node_num
        self.traj_len = cfg.traj_len
        self.video_time = cfg.video_time
        self.dataset = cfg.dataset
        self.node_feats_path = cfg.node_feats_path
        self.orig_node_feats_path = cfg.orig_node_feats_path
        self.feat_dim = cfg.feat_dim

        if self.dataset == "cityflow" or self.dataset == "uv" or self.dataset == "uv-z":
            self.folder_name = ""
        elif self.dataset == "carla" or self.dataset == "video-sg":
            self.folder_name = "t%02d_c%03d_len%02d" % (self.video_time, self.node_num, self.traj_len)
        if self.dataset == "cityflow" or self.dataset == "video-sg":
            self.app_feat_partition_file = 'partition.txt'
        elif self.dataset == "uv" or self.dataset == "uv-z":
            self.app_feat_partition_file = 'app_feat_partition.txt'
            self.plate_feat_partition_file = 'plate_feat_partition.txt'
        self.data_path = os.path.join(self.node_feats_path, self.folder_name)
        self.nlist = 100
        self.m = 32

    def load_gallery_data(self, i, index, d):
        gf_file = self.data_path + '/%d/%d_gf.bin' % (i, i)
        features = np.fromfile(gf_file, dtype=np.float32)
        features.shape = -1, d
        index.add(features)
        return index

    def pick_train_data(self, node, d, line_cnt, train_data, global_cnt, limit):
        # load the gallery file
        gf_file = self.data_path + '/%d/%d_gf.bin' % (node, node)
        features = np.fromfile(gf_file, dtype=np.float32)
        features.shape = -1, d
        with open(self.data_path + "/partition.txt", 'a') as f:
            content = "%d,%d,%d\n" % (line_cnt, line_cnt + features.shape[0] - 1, node)
            f.write(content)
        line_cnt += features.shape[0]

        # randomly pick train data
        num = round(features.shape[0] / 5)
        cnt = 0
        ans = set()
        while cnt < num:
            temp = random.randint(0, features.shape[0] - 1)
            if temp not in ans:
                ans.add(temp)
                cnt += 1

        local_cnt = 0
        local_train_data = np.zeros(shape=[num, d]).astype('float32')
        for i in ans:
            local_train_data[local_cnt] = features[i]
            local_cnt += 1
            if global_cnt < limit:
                train_data[global_cnt] = features[i]
                global_cnt += 1
            else:
                break

        return train_data, line_cnt, global_cnt

    def cal_train_data_num(self):
        all_feature_num = 0
        for node in range(self.node_num):
            frame_file = self.data_path + "/%d/%d_frame.bin" % (node, node)
            frames = np.fromfile(frame_file, dtype=np.int64)
            all_feature_num += len(frames)
        
        return all_feature_num // 5

    def build_index(self):
        dst = self.data_path + '/train_data.bin'
        if os.path.exists(dst):
            return 0, 0, 0

        d = self.feat_dim

        t = time.time()
        all_feature_num = self.cal_train_data_num()
        t = time.time() - t
        print("cal all feature num %d with time %f" % (all_feature_num, t))

        # pick train data
        print("Merging training features")
        time_pick_train_data = time.time()
        train_data = np.zeros(shape=[all_feature_num, d]).astype('float32')
        line_cnt = 0
        global_cnt = 0
        for node in range(self.node_num):
            train_data, line_cnt, global_cnt = self.pick_train_data(node, d, line_cnt, train_data, global_cnt, all_feature_num)
            print("node %d is ok" % node)
        time_pick_train_data = time.time() - time_pick_train_data
        print("Picking and merging time: %f" % time_pick_train_data)

        # build index
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, self.nlist, self.m, 8)
        print("Training index")
        time_train_index = time.time()
        index.train(train_data)
        time_train_index = time.time() - time_train_index
        print("Training time: %f" % time_train_index)

        print("Loading gallery features and adding index")
        time_build_index = time.time()
        for i in range(self.node_num):
            index = self.load_gallery_data(i, index, d)
            print("node %d is ok" % i)
        time_build_index = time.time() - time_build_index
        faiss.write_index(index, self.data_path + "/features.index")
        return time_pick_train_data, time_train_index, time_build_index

    def cal_all_features_dis(self, index, query_feature, k):
        print("Searching with faiss index")
        time_search = time.time()

        index.nprobe = 2
        D, I = index.search(query_feature, k)

        # sort by the distance
        D = D[0]
        I = I[0]
        sorted_indices = np.argsort(D)[::-1]
        D, I = D[sorted_indices], I[sorted_indices]
        time_search = time.time() - time_search
        
        return D, I, time_search

    def find_new_index(self, node_feats_path, raw_index, folder_name, partition_file):
        path = os.path.join(node_feats_path, folder_name, partition_file)
        with open(path, 'r') as f:
            records = [line[:-1].split(',') for line in f.readlines()]
            start = [int(line[0]) for line in records]
        
        pivot_index = bisect.bisect_right(start, raw_index) - 1
        new_index = raw_index - start[pivot_index]
        file_id = int(records[pivot_index][2])
        
        return file_id, new_index

    def gen_idx_id_mapping(self, id_file_type):
        # generate the mapping relationship from index to unified records json order
        index_to_unified_records_json_order_list = list()
        orig_gf_feats_path = self.orig_node_feats_path
        for file in sorted(os.listdir(orig_gf_feats_path)):
            if not os.path.exists(os.path.join(orig_gf_feats_path, file, file+id_file_type)):
                continue
            feats_id_list = np.load(os.path.join(orig_gf_feats_path, file, file+id_file_type))
            index_to_unified_records_json_order_list += feats_id_list.tolist()
        index_to_unified_records_json_order_dict = {}
        for i, val in enumerate(index_to_unified_records_json_order_list):
            index_to_unified_records_json_order_dict[i] = val
        unified_records_json_order_to_index_dict = dict([val, key] for key, val in index_to_unified_records_json_order_dict.items())
        
        return index_to_unified_records_json_order_dict, unified_records_json_order_to_index_dict

    def fusion_search_space(self, query_feature, query_plate_feature, app_feat_list, 
                            plate_feat_list, D_app, I_app, D_plate, I_plate, save_dirs, 
                            index_to_unified_records_json_order_dict, 
                            unified_records_json_order_to_index_dict,
                            plate_index_to_unified_records_json_order_dict, 
                            unified_records_json_order_to_plate_index_dict):
        print("Fusion search space index")
        time_fusion = time.time()

        # generate unified records json order for appearance index and plate index
        I_app_unified = {}
        for i in I_app:
            if i < 0:
                continue
            mapped_id = index_to_unified_records_json_order_dict[i]
            I_app_unified[i] = mapped_id
        I_plate_unified = {}
        for i in I_plate:
            if i < 0:
                continue
            mapped_id = plate_index_to_unified_records_json_order_dict[i]
            I_plate_unified[i] = mapped_id

        app_weight = 0.3
        plate_weight = 0.7

        D_fusion = list()
        I_fusion = list()
        
        traversed_idx_list = list()
        for i, d in enumerate(D_app):
            if I_app[i] < 0:
                continue
            unified_idx = index_to_unified_records_json_order_dict[I_app[i]]
            if unified_idx in I_fusion:
                continue
            if unified_idx in I_plate_unified.values():
                curr_plate_index_idx = unified_records_json_order_to_plate_index_dict[unified_idx]
                traversed_idx_list.append(curr_plate_index_idx)
                I_plate_i = I_plate.tolist().index(curr_plate_index_idx)
                d_fusion = app_weight * d + plate_weight * D_plate[I_plate_i]
                D_fusion.append(d_fusion)
                I_fusion.append(unified_idx)
            else:
                curr_plate_feat = plate_feat_list[unified_idx]
                if len(curr_plate_feat) == 0:
                    D_fusion.append(d)
                    I_fusion.append(unified_idx)
                else:
                    # the correspondence computation needs to be consistent with generated index
                    curr_plate_feat_dist = get_cos_sim(np.squeeze(query_plate_feature), curr_plate_feat)
                    d_fusion = app_weight * d + plate_weight * curr_plate_feat_dist
                    D_fusion.append(d_fusion)
                    I_fusion.append(unified_idx)
        
        for i, d in enumerate(D_plate):
            if I_plate[i] < 0:
                continue
            if I_plate[i] in traversed_idx_list:
                continue
            unified_idx = plate_index_to_unified_records_json_order_dict[I_plate[i]]
            if unified_idx in I_fusion:
                continue
            curr_feat = app_feat_list[unified_idx]
            # the correspondence computation needs to be consistent with generated index
            curr_app_feat_dist = get_cos_sim(np.squeeze(query_feature), curr_feat)
            d_fusion = app_weight * curr_app_feat_dist + plate_weight * d
            D_fusion.append(d_fusion)
            I_fusion.append(unified_idx)

        I_fusion_new = list()
        for i in I_fusion:
            I_fusion_new.append(unified_records_json_order_to_index_dict[i])

        D_fusion = np.array(D_fusion)
        I_fusion = np.array(I_fusion_new)

        # sort by the distance
        sorted_indices = np.argsort(D_fusion)[::-1]
        D_fusion = D_fusion[sorted_indices]
        I_fusion = I_fusion[sorted_indices]
        
        time_fusion = time.time() - time_fusion
        
        # save the results
        np.save(save_dirs + '/fusion_original_dist.npy', D_fusion)
        np.save(save_dirs + '/fusion_order_by_index.npy', I_fusion)
        
        return D_fusion, I_fusion, time_fusion

    def dist_thres_trunc(self, D_fusion, I_fusion, thres):
        # convert to unified records json order ID
        thres_trunc_idx = [i for i, dist in enumerate(D_fusion) if dist <= thres]
        D_thres_trunc = D_fusion[thres_trunc_idx]
        I_thres_trunc = I_fusion[thres_trunc_idx]
        
        return D_thres_trunc, I_thres_trunc
    
    def sim_thres_trunc(self, D_fusion, I_fusion, thres, save_dirs):
        time_sim_thres_trunc = time.time()

        mask = D_fusion >= thres
        D_thres_trunc = D_fusion[mask]
        I_thres_trunc = I_fusion[mask]
        
        time_sim_thres_trunc = time.time() - time_sim_thres_trunc
        
        # save the final retrieved results
        np.save(os.path.join(save_dirs, 'original_dist.npy'), D_thres_trunc)
        np.save(os.path.join(save_dirs, 'retrieved_index.npy'), I_thres_trunc)

        return D_thres_trunc, I_thres_trunc, time_sim_thres_trunc

    def euclidean_distance(self, qf, gf):
        m = qf.shape[0]
        n = gf.shape[0]
        dist_mat = np.broadcast_to(np.power(qf, 2).sum(axis=1, keepdims=True), (m, n)) + \
                np.broadcast_to(np.power(gf, 2).sum(axis=1, keepdims=True), (n, m)).T
        dist_mat -= 2 * np.dot(qf, gf.T)[0], 
        return dist_mat

    def cosine_similarity(self, qf, gf):
        epsilon = 0.00001
        dist_mat = qf.mm(gf.t())
        qf_norm = np.norm(qf, p=2, dim=1, keepdim=True)  # mx1
        gf_norm = np.norm(gf, p=2, dim=1, keepdim=True)  # nx1
        qg_normdot = qf_norm.mm(gf_norm.t())

        dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
        dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
        dist_mat = np.arccos(dist_mat)
        return dist_mat

    def cal_all_features_dis_traverse(self, query_feature, dirs, k):
        # camera list order is same as sorted(os.listdir(os.path.join(data_path, 'gf_feats')))
        print("Searching with faiss index")
        time_search = time.time()
        orig_gf_feats_path = self.data_path.replace('gf_feats_bin/', 'gf_feats/')
        dist_list = list()
        for cam_id in sorted(os.listdir(orig_gf_feats_path)):
            if 'features.index' in cam_id:
                continue
            if cam_id.split('_')[1] == 'frame.npy':
                local_feats_array = np.load(os.path.join(orig_gf_feats_path, cam_id.split('_')[0] + '_gf.npy'))
                dist = self.euclidean_distance(query_feature, local_feats_array)
                dist_list.append(dist)

        dist_all = dist_list[0]
        for dist in dist_list[1:]:
            dist_all = np.concatenate((dist_all, dist), axis=1)
        
        I = np.argpartition(dist_all[0], k)[:k]
        D = dist_all[0][I]

        # sort by the distance
        sorted_indices = np.argsort(D)
        D = D[sorted_indices]
        I = I[sorted_indices]
        time_search = time.time() - time_search
        
        np.save(dirs + '/original_dist.npy', D)
        np.save(dirs + '/order_by_index.npy', I)
        
        D = (D - D.min()) / (D.max() - D.min())
        np.save(dirs + '/normalized_dist.npy', D)
        
        return D, I, time_search
