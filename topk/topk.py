import pandas as pd
from common.calculate_function import *


class Topk(object):
    def __init__(self, k_limit, time_gap, feat_dim):
        super(Topk, self).__init__()
        self.k_limit = k_limit
        self.camera_list = []
        self.time_gap = time_gap
        self.feat_dim = feat_dim

    def add_camera(self, camera):
        self.camera_list.append(camera)
        camera_index = len(self.camera_list)
        camera.set_idx_in_top_k(camera_index - 1)

    def find_camera(self, candidate):
        cand_nodeid, cand_frame, cand_idx_in_frame = candidate.get_info()
        min_diff = 99999
        correct_index = -1
        for possible_camera in self.camera_list:
            if possible_camera.get_cam_id() != cand_nodeid:
                continue
            timestamps = possible_camera.get_timestamps()
            for t in timestamps:
                if min_diff > abs(t - cand_frame):
                    min_diff = abs(t - cand_frame)
                    correct_index = possible_camera.get_idx_in_top_k()
        if min_diff > self.time_gap:
            correct_index = -1
        return correct_index

    def add_candidate_to_camera(self, candidate, camera_index):
        self.camera_list[camera_index].add_candidate(candidate)

    def camera_sort_info(self):
        for cam in self.camera_list:
            cam.sort_info()

    def save_top_k_info(self, folder):
        filename = folder + '/top_k.csv'
        for cam in self.camera_list:
            cam.save_frame_and_feature(filename)
        df = pd.read_csv(filename, header=0)
        appear_time = df["appear times"].values.tolist()
        node_weight = cal_appear_weight(appear_time)
        # df.insert(3, "node weight", node_weight)
        if "node weight" not in df.columns:
            df.insert(3, "node weight", node_weight)
        else:
            df["node weight"] = node_weight
        df.to_csv(filename, mode='w', index=False)

    def save_avg_features(self, folder, folder_name):
        features = np.zeros(shape=(len(self.camera_list), self.feat_dim)).astype('float32')
        for idx, cam in enumerate(self.camera_list):
            features[idx] = cam.get_avg_feature(folder_name)
        data = features.tobytes()
        filename = folder + '/avg_features.bin'
        f = open(filename, "wb")
        f.write(data)
        f.close()
