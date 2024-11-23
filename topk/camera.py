from collections import Counter
from topk.candidate import *


class Camera(object):
    def __init__(self, cam_id):
        super(Camera, self).__init__()
        self.cam_id = cam_id
        self.candidate_list = []
        self.idx_in_topk = -1

    def add_candidate(self, candidate):
        self.candidate_list.append(candidate)

    def get_cam_id(self):
        return self.cam_id

    def set_idx_in_top_k(self, idx):
        self.idx_in_topk = idx

    def get_idx_in_top_k(self):
        return self.idx_in_topk

    def sort_info(self):
        self.candidate_list.sort(key=lambda x: x.frame)

    def get_timestamps(self):
        return [cand.get_frame() for cand in self.candidate_list]

    def get_duration_time(self):
        timeline = [cand.get_frame() for cand in self.candidate_list]
        return np.min(timeline), np.max(timeline)

    def save_frame_and_feature(self, filename):
        timestamps = []
        feature_raw_index = []
        inx_in_frame = []
        for cand in self.candidate_list:
            timestamps.append(cand.get_frame())
            feature_raw_index.append(cand.get_feature_raw_index())
            inx_in_frame.append(cand.get_idx_in_frame())
        avg_frame = sum(timestamps) // len(timestamps)
        if not os.path.exists(filename):
            f = open(filename, 'w')
            csv_writer = csv.writer(f)
            csv_writer.writerow(["camid", "avg timestamp", "appear times", "timestamps", "feature index", "index in frame"])
            csv_writer.writerow([self.cam_id, avg_frame, len(timestamps), timestamps, feature_raw_index, inx_in_frame])
        else:
            f = open(filename, 'a')
            csv_writer = csv.writer(f)
            csv_writer.writerow([self.cam_id, avg_frame, len(timestamps), timestamps, feature_raw_index, inx_in_frame])

    def get_avg_feature(self, folder_name):
        all_num = len(self.candidate_list)
        # take the feature of the middle candidate snapshot as the average feature
        avg_feature = self.candidate_list[(all_num-1) // 2].get_feature(folder_name)
        return avg_feature
    
    def get_avg_plate_feature(self, folder_name):
        valid_features = []
        for candidate in self.candidate_list:
            feature = candidate.get_plate_feature(folder_name)
            
            if feature is not None and not np.all(feature == 0): # check if the feature is not all zeros
                valid_features.append(feature)
        
        if valid_features:
            avg_feature = np.mean(valid_features, axis=0)
        else:
            avg_feature = None
        
        return avg_feature

    def get_frequent_plate_text(self):
        plate_text_list = []
        for cand in self.candidate_list:
            cand_plate_text = cand.get_plate_text()
            plate_text_list.append(cand_plate_text)
        plate_text_list = [s for s in plate_text_list if s]

        # find the most frequent license plate text
        if plate_text_list:
            counter = Counter(plate_text_list)
            frequent_plate_text, _ = counter.most_common(1)[0]
        else:
            frequent_plate_text = ''

        return frequent_plate_text
    
    def get_cand_dist_list(self):
        dist_list = [cand.dist for cand in self.candidate_list]
        return dist_list
