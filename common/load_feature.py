import numpy as np
import os
import bisect


def find_new_index(node_feats_path, raw_index, folder_name, partition_file):
    path = os.path.join(node_feats_path, folder_name, partition_file)
    with open(path, 'r') as f:
        records = [line[:-1].split(',') for line in f.readlines()]
        start = [int(line[0]) for line in records]
    
    pivot_index = bisect.bisect_right(start, raw_index) - 1
    new_index = raw_index - start[pivot_index]
    file_id = int(records[pivot_index][2])
    
    return file_id, new_index

def get_candidate_info_by_index(node_feats_path, raw_index, folder_name, frame_tpye_byte_num, partition_file):
    file_id, new_index = find_new_index(node_feats_path, raw_index, folder_name, partition_file)
    prefix = os.path.join(node_feats_path, folder_name, str(file_id))

    with open(os.path.join(prefix, str(file_id) + '_frame.bin'), 'rb') as f1:
        f1.seek(1 * frame_tpye_byte_num * new_index)
        data1 = f1.read(1 * frame_tpye_byte_num)
        frame = np.frombuffer(data1, dtype=np.int64)[0]

    camid = file_id

    with open(os.path.join(prefix, str(file_id) + '_idx_in_frame.bin'), 'rb') as f2:
        f2.seek(1 * frame_tpye_byte_num * new_index)
        data2 = f2.read(1 * frame_tpye_byte_num)
        idx_in_frame = np.frombuffer(data2, dtype=np.int64)[0]
    
    return frame, camid, idx_in_frame

def get_feaure_by_index(node_feats_path, raw_index, folder_name, feat_dim, partition_file, feature_type):
    file_id, new_index = find_new_index(node_feats_path, raw_index, folder_name, partition_file)
    
    filename = os.path.join(node_feats_path, folder_name, str(file_id), str(file_id)+feature_type)
    f = open(filename, "rb")
    f.seek(feat_dim * 4 * new_index)
    data = f.read(feat_dim * 4)
    feature = np.frombuffer(data, dtype=np.float32)

    return feature

def load_avg_feature(filename, index, feat_dim):
    f = open(filename, "rb")
    f.seek(feat_dim * 4 * index)
    data = f.read(feat_dim * 4)
    feature = np.frombuffer(data, dtype=np.float32)
    return feature

