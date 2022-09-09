import os
import shutil

import cv2
import pandas as pd
import numpy as np
from collections import defaultdict


class SynchronizedData:
    def __init__(self, base_path, names, init_index=None):
        self.base_path = base_path
        self.names = names
        self.num_cameras = len(names)
        self.init_index = init_index
        self.init_frames = []
        self.num_frames = []
        self.tstamps = []
        self.video_filenames = []
        self.synchronise_data()

    def get_num_cameras(self):
        return self.num_cameras

    def get_init_frame(self, index):
        return self.init_frames[index]

    def get_num_frames(self):
        return self.num_frames

    def get_timestamps(self, index):
        return self.tstamps[index]

    def get_videos_path(self):
        return self.video_filenames

    def get_video_path(self, index):
        return self.video_filenames[index]

    def get_saving_path(self, index):
        return os.path.join(self.base_path, self.names[index], 'rgb')

    def synchronise_data(self):
        data_all = defaultdict()
        # base_path: '/media/robotvision/data/datasets/MEMEX_Pilots/Lisbon/Mission_2/Sequence_3'
        num_frames = []
        video_filenames = []
        for name_id, name in enumerate(self.names):  # names: ['B1', 'B2', 'B3', 'B4', 'T1', 'T2', 'T3', 'T4']
            path_to_folder = os.path.join(self.base_path, name)
            dir_content = os.listdir(path_to_folder)
            l = len(dir_content)
            ids_to_suppress = []
            for i, elm in enumerate(dir_content):
                if not os.path.isdir(os.path.join(path_to_folder, elm)):  # keep only the directory
                    ids_to_suppress.append(i)
                    if str(elm).endswith('mp4'):
                        video_name = elm
                        video_filenames.append(os.path.join(self.base_path, name, video_name))

            # Get the csv data
            dir_content = [ele for idx, ele in enumerate(dir_content) if idx not in ids_to_suppress]
            data = []
            for elm in dir_content:
                if 'rgb' in str(elm):
                    delete_file_folder(os.path.join(path_to_folder, elm))
                    continue
                try:
                    data = pd.read_csv(os.path.join(path_to_folder, elm, f'{elm}.csv'))
                    cap = cv2.VideoCapture(os.path.join(self.base_path, f'{name}', video_name))
                    num_frames.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                except OSError:
                    print(f"couldn't load the file {os.path.join(path_to_folder, elm, f'{elm}.csv')}")

            data_all[f'{name_id}'] = data
            cap.release()

        # Get the timestamps
        tstamps = []
        tstamps_init = []
        for i, data_name in enumerate(data_all):
            tstmp_tmp = data_all[data_name].get('Unix time[nanosec]') / 1000000000
            l = min(len(tstmp_tmp), num_frames[i])
            num_frames[i] = l
            tstmp_tmp = tstmp_tmp[:l]  # keep the same number of frames and tstamps
            tstamps_init.append(tstmp_tmp[0])
            tstamps.append(tstmp_tmp)

        # Synchronize initial timestamps
        init_frames = []
        tstamps_init_np = np.array(tstamps_init)
        if (self.init_index is None) or (self.init_index == 0):
            max_val = tstamps_init_np.max()
        else:
            max_val = tstamps[0][self.init_index]
        for i, tstamp in enumerate(tstamps):
            init_frame = find_closest_index(max_val, tstamp.to_numpy())
            init_frames.append(init_frame)
            print(f"Initial frame {self.names[i]} = {tstamp[init_frame]}")

        init_frames = np.array(init_frames)
        num_frames = np.array(num_frames)
        seq_len = num_frames - init_frames
        min_seq_len = seq_len.min()
        num_frames_to_save = min_seq_len

        self.init_frames = init_frames.tolist()
        self.num_frames = num_frames_to_save
        self.tstamps = tstamps
        self.video_filenames = video_filenames


def find_closest_index(val, array):
    delta = np.abs(val - array)

    return np.argmin(delta)


def delete_file_folder(path):
    # Check if the path is a file
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    else:
        raise ValueError(f'Path {path} is neither a file or a directory.')
