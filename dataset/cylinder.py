from torch.utils.data import IterableDataset
import os, numpy as np
import os.path as osp
import h5py
from torch_geometric.data import Data
import torch
import math
import time

class GraphTrajectoryLoader():
    '''
    Iterable dataset that loads trajectories from the cylinder flow h5 files and yields graphs.
    Each trajectory is opened and read as needed, with a limit on the number of open trajectories.
    Within each trajectory, frames are read in random order, so that the model sees a variety of data.

    ------------------- Epoch 1 -------------------
    Shuffled Trajectories: T2, T0, T3, T1
    Opened Trajectories (max 2 at a time): [T2, T0]

    Random frame order per trajectory:
    T2: [f3, f0, f1, f4, f2]
    T0: [f2, f4, f0, f3, f1]

    Iteration (__next__ calls):
    Step 1: pick T0 → yield f2 (x=v2, y=v3)
    Step 2: pick T2 → yield f3 (x=v3, y=v4)
    Step 3: pick T0 → yield f4 (x=v4, y=v5)
    Step 4: pick T2 → yield f0 (x=v0, y=v1)
    ...
    Step N: trajectory frames exhausted → close trajectory
    Step N+1: open next trajectory from shuffled list

    Epoch ends when all trajectories have been opened at least once.
    Frames consumed per trajectory may vary; some trajectories may be fully seen, others only partially.
    -----------------------------------------------
    Notes:
    - Each __next__() yields one training sample (x=current velocity, y=next velocity)
    - Epoch = all trajectories opened at least once
    - Frames within a trajectory are sampled in a **random order**
    '''
    def __init__(self, max_epochs=1, files=None):
        self.max_epochs = max_epochs

        # load specified .h5 files
        self.file_handle=files
        self.shuffle_file()

        # data keys to load from each trajectory
        self.data_keys =  ("pos", "node_type", "velocity", "cells", "pressure")
        
        # add time to the output keys
        self.out_keys = list(self.data_keys)  + ['time']

        # dictionary: traj_key -> number of frames
        self.traj_lens = {}
        for traj_key in self.file_handle.keys():
            # extract number of frames from velocity dataset shape
            self.traj_lens[traj_key] = self.file_handle[traj_key]['velocity'].shape[0]

        # dataset constant time interval between frames
        self.time_interval = 0.01

        # number of trajectories to keep open simultaneously
        self.n_open_traj = 10

        # iteration and epoch initial states
        self.traj_index = 0
        self.n_epoch = 1

        # currently opened trajectories and their states
        self.opened_traj = []
        self.opened_traj_read_index = {}
        self.opened_traj_read_random_index = {}
        self.traj_data = {}
    
    def open_traj(self):
        # open new trajectories until reaching the max number of opened trajectories
        while(len(self.opened_traj) < self.n_open_traj):
            # First check epoch end before indexing self.datasets
            if self.check_if_epoch_end():
                self.epoch_end()
                print('Shuffling epoch %d ended.' % (self.n_epoch - 1))
                break

            traj_index = self.datasets[self.traj_index]

            if traj_index not in self.opened_traj:
                # number of frames in this trajectory
                n_frames = self.traj_lens[traj_index]
                # skip trajectories that don't have at least 3 frames (t-1, t and t+1)
                if n_frames < 3:
                    # Option: issue a warning or just skip
                    print(f"Skipping trajectory {traj_index}: only {n_frames} frames")
                    self.traj_index += 1
                    continue

                # append new trajectory to currently opened list
                self.opened_traj.append(traj_index)

                # allowed frames: 0 .. n_frames-2 (so model can use frame t and t+1)
                # recall frames are 0-indexed
                self.opened_traj_read_random_index[traj_index] = np.random.permutation(n_frames - 1)
                self.opened_traj_read_index[traj_index] = -1

            self.traj_index += 1
    
    def check_and_close_traj(self):
        to_del = []
        for traj in self.opened_traj:
            n_frames = self.traj_lens[traj]
            last_usable_index = n_frames - 2   # t = 0..num_frames-2 so t+1 exists
            # check if all frames in this trajectory have been read
            if self.opened_traj_read_index[traj] >= last_usable_index:
                to_del.append(traj)

        # close trajectories that have been fully read
        for traj in to_del:
            self.opened_traj.remove(traj)
            try:
                del self.opened_traj_read_index[traj]
                del self.opened_traj_read_random_index[traj]
                del self.traj_data[traj]
            except Exception as e:
                print(e)

    def shuffle_file(self):
        # shuffle the list of trajectories for this epoch
        datasets = list(self.file_handle.keys())
        np.random.shuffle(datasets)
        self.datasets = datasets

    def epoch_end(self):
        # reset state for new epoch
        self.traj_index = 0

        # call shuffle_file to reshuffle trajectories
        self.shuffle_file()
        self.n_epoch = self.n_epoch + 1

    def check_if_epoch_end(self):
        # use datasets length (the shuffled list) to determine epoch end
        return self.traj_index >= len(self.datasets)

    @staticmethod
    def datas_to_graph(datas):
        ''' 
        Convert raw data arrays to a PyG Data graph object.
        '''
        # define time as a node feature
        time_vector = np.ones((datas[0].shape[0], 1))*datas[5]

        # concatenate node features: node_type, current velocity v_t, pressure, time
        node_attr = np.hstack((datas[1], datas[2][0], datas[4][0], time_vector))
        node_attr = torch.as_tensor(node_attr, dtype=torch.float)

        # define nodes coordinates which will become graph.pos in PyG
        coord = torch.as_tensor(datas[0], dtype=torch.float)

        # target is the next velocity v_{t+1}
        target = datas[2][1]
        target = torch.from_numpy(target)

        # define graph faces (triangular mesh connectivity in datas[3])
        face = torch.as_tensor(datas[3].T, dtype=torch.long)

        # create PyG Data object
        g = Data(x=node_attr, face=face, y=target, pos=coord)

        return g

    def __next__(self):
        '''
        Yield the next graph sample for training.
        '''
        self.check_and_close_traj()
        self.open_traj()

        # when the maximum number of epochs is reached, stop iteration
        if self.n_epoch > self.max_epochs:
            raise StopIteration

        # randomly select one of the currently opened trajectories
        selected_traj = np.random.choice(self.opened_traj)

        data = self.traj_data.get(selected_traj, None)

        # load trajectory data if not already loaded
        if data is None:
            data = self.file_handle[selected_traj]
            self.traj_data[selected_traj] = data

        selected_traj_read_index = self.opened_traj_read_index[selected_traj]
        selected_frame = self.opened_traj_read_random_index[selected_traj][selected_traj_read_index + 1]
        self.opened_traj_read_index[selected_traj] += 1

        # extract data for the selected frame and the next frame
        datas = []
        for k in self.data_keys:
            if k in ["velocity", "pressure"]:
                r = np.array((data[k][selected_frame], data[k][selected_frame + 1]), dtype=np.float32)
            else:
                r = data[k][selected_frame]
                if k in ["node_type", "cells"]:
                    r = r.astype(np.int32)
            datas.append(r)

        # add time information
        datas.append(np.array([self.time_interval * selected_frame], dtype=np.float32))

        # the current structure is ("pos", "node_type", "velocity", "cells", "pressure", "time")
        g = self.datas_to_graph(datas)
  
        return g

    def __iter__(self):
        return self