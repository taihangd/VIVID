import csv
import os
from common.load_feature import *


class Candidate(object):
    def __init__(self, feature_raw_idx, frame, nodeid, dist, rank, idx_in_frame, 
                node_feats_path, feat_dim, partition_file, plate_partition_file=None, 
                plate_feature_raw_idx=None, plate_feature_text=None):
        super(Candidate, self).__init__()
        self.feature = feature_raw_idx
        self.frame = frame
        self.nodeid = nodeid
        self.idx_in_frame = idx_in_frame
        self.dist = dist
        self.rank = rank
        self.node_feats_path = node_feats_path
        self.feat_dim = feat_dim
        self.partition_file = partition_file

        self.plate_feature = plate_feature_raw_idx
        self.plate_partition_file = plate_partition_file
        self.plate_text = plate_feature_text

    def add_plate_feature(self, plate_feature_raw_idx, plate_partition_file):
        self.plate_feature = plate_feature_raw_idx
        self.plate_partition_file = plate_partition_file
        return

    def add_plate_text(self, plate_feature_text):
        self.plate_text = plate_feature_text
        return

    def get_img_name(self):
        return 'c%04d_%07d_%07d.jpg' % (self.nodeid, self.frame, self.idx_in_frame)

    def save_info(self, dirs):
        if os.path.exists(dirs + '/top_k.csv'):
            f = open(dirs + '/top_k.csv', 'a')
            csv_writer = csv.writer(f)
            csv_writer.writerow([self.frame, self.nodeid, self.dist, self.rank, self.idx_in_frame])
        else:
            f = open(dirs + '/top_k.csv', 'w')
            csv_writer = csv.writer(f)
            csv_writer.writerow(["frame", "nodeid", "dist", "rank", "idx_in_frame"])
            csv_writer.writerow([self.frame, self.nodeid, self.dist, self.rank, self.idx_in_frame])

    def get_frame(self):
        return self.frame

    def get_feature_raw_index(self):
        return self.feature

    def get_plate_feature_raw_index(self):
        return self.plate_feature

    def get_feature(self, folder_name):
        return get_feaure_by_index(self.node_feats_path, self.feature, folder_name, 
                                self.feat_dim, self.partition_file, "_gf.bin")
    
    def get_plate_feature(self, folder_name):
        if self.plate_feature != None and self.plate_partition_file != None:
            plate_feature = get_feaure_by_index(self.node_feats_path, self.plate_feature, folder_name, 
                                                self.feat_dim, self.plate_partition_file, "_gpf.bin")
        else:
            plate_feature = None
        return plate_feature
    
    def get_plate_text(self):
        return self.plate_text

    def get_nodeid(self):
        return self.nodeid

    def get_idx_in_frame(self):
        return self.idx_in_frame

    def get_info(self):
        return self.nodeid, self.frame, self.idx_in_frame

    def __lt__(self, other):
        return self.frame < other.frame
