import os
import numpy as np
import torch
from torch.utils import data
import h5py

class mesh_pc_dataset_normal_16384(data.Dataset):
    def __init__(self, data_path, mode='train', min_num_point=3000, max_num_point=16384, num_sample=2048, rot=False, return_file_path=False, npz_name='pdc_16384_normal.npz', noise_std=-1):
        super(mesh_pc_dataset_normal_16384, self).__init__()
        self.noise_std = noise_std
        self.origin_data_path = data_path
        self.mode = mode
        self.return_file_path = return_file_path
        assert mode in ['train', 'test', 'val']
        self.min_num_point = min_num_point
        self.max_num_point = max_num_point
        self.num_sample = num_sample
        # self.split=np.load(os.path.join(data_path,'split.npz'))[mode]
        self.rot = rot
        self.npz_name = npz_name

        self.data_path = []
        self.data_path_noprefix = []

        cat_list = os.listdir(data_path)
        for cat in cat_list:
            with open(os.path.join(data_path, cat, mode + '.lst')) as f:
                models_c = f.read().split('\n')
            if '' in models_c:
                models_c.remove('')
            self.data_path = self.data_path + [os.path.join(data_path, cat, model) for model in models_c if
                                               os.path.exists(os.path.join(data_path, cat, model))]
            self.data_path_noprefix = self.data_path_noprefix + [os.path.join(cat, model) for model in models_c if
                                               os.path.exists(os.path.join(data_path, cat, model))]

    def rotate_point_cloud_and_gt(self):
        # angles = np.random.uniform(0, 1) * np.pi * 2

        angles = np.random.choice([0, np.pi / 2, np.pi, np.pi / 2 * 3], 1)

        Ry = np.array([[np.cos(angles), 0, -np.sin(angles)],
                       [0, 1, 0],
                       [np.sin(angles), 0, np.cos(angles)]
                       ])

        return Ry

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):

        data_path = self.data_path[idx]
        data_path_noprefix = self.data_path_noprefix[idx]
        lr_pc = np.load(os.path.join(data_path, 'points.npy'))
        hr_data = np.load(os.path.join(data_path, self.npz_name))
        hr_pc = hr_data['sparse_points']
        hr_n = hr_data['sparse_normals']

        index = np.random.permutation(lr_pc.shape[0])[0:self.min_num_point]
        lr_pc = lr_pc[index, :]

        if hr_pc.shape[0] < self.max_num_point:
            add_num = self.max_num_point - hr_pc.shape[0]
            add_index = np.random.permutation(hr_pc.shape[0])[:add_num]
            add_points = hr_pc[add_index, :]
            add_normals = hr_n[add_index, :]
            hr_pc = np.concatenate((hr_pc, add_points), axis=0)
            hr_n = np.concatenate((hr_n, add_normals), axis=0)
        if hr_pc.shape[0] > self.max_num_point:
            index = np.random.permutation(hr_pc.shape[0])[:self.max_num_point]
            hr_pc = hr_pc[index, :]
            hr_n = hr_n[index, :]

        if self.mode == 'train' and self.rot:
            Rot = self.rotate_point_cloud_and_gt()
            lr_pc = np.dot(lr_pc, Rot)
            hr_pc = np.dot(hr_pc, Rot)
            hr_n = np.dot(hr_n, Rot)

        if self.noise_std > 0:
            noise = np.random.randn(*lr_pc.shape) * self.noise_std
            lr_pc = lr_pc + noise

        if self.return_file_path:
            return [
                torch.from_numpy(hr_pc).transpose(0, 1).float(),  # (3,RN)
                torch.from_numpy(lr_pc).transpose(0, 1).float(),  # (3,N)
                torch.from_numpy(hr_n).transpose(0, 1).float(),
                data_path_noprefix
            ]
        return [
            torch.from_numpy(hr_pc).transpose(0, 1).float(),  # (3,RN)
            torch.from_numpy(lr_pc).transpose(0, 1).float(),  # (3,N)
            torch.from_numpy(hr_n).transpose(0, 1).float()
        ]


    def __init__(self, data_path, query_path, mode='train', min_num_point=3000, max_num_point=48000, num_sample=2048, rot=False):
        super(mesh_pc_dataset_normal_query, self).__init__()
        self.data_path = data_path
        self.mode = mode
        assert mode in ['train', 'test', 'val']
        self.min_num_point = min_num_point
        self.max_num_point = max_num_point
        self.num_sample = num_sample
        # self.split=np.load(os.path.join(data_path,'split.npz'))[mode]
        self.rot = rot

        self.data_path = []
        self.query_path = []

        cat_list = os.listdir(data_path)
        for cat in cat_list:
            with open(os.path.join(data_path, cat, mode + '.lst')) as f:
                models_c = f.read().split('\n')
            if '' in models_c:
                models_c.remove('')
            self.data_path = self.data_path + [os.path.join(data_path, cat, model) for model in models_c if
                                               os.path.exists(os.path.join(data_path, cat, model))]

        cat_list = os.listdir(query_path)
        for cat in cat_list:
            with open(os.path.join(query_path, cat, mode + '.lst')) as f:
                models_c = f.read().split('\n')
            if '' in models_c:
                models_c.remove('')
            self.query_path = self.query_path + [os.path.join(query_path, cat, model) for model in models_c if
                                               os.path.exists(os.path.join(query_path, cat, model))]

    def rotate_point_cloud_and_gt(self):
        # angles = np.random.uniform(0, 1) * np.pi * 2

        angles = np.random.choice([0, np.pi / 2, np.pi, np.pi / 2 * 3], 1)

        Ry = np.array([[np.cos(angles), 0, -np.sin(angles)],
                       [0, 1, 0],
                       [np.sin(angles), 0, np.cos(angles)]
                       ])

        return Ry

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):

        data_path = self.data_path[idx]
        lr_pc = np.load(os.path.join(data_path, 'points.npy'))
        hr_data = np.load(os.path.join(data_path, 'pdc_48000_normal.npz'))
        hr_pc = hr_data['sparse_points']
        hr_n = hr_data['sparse_normals']

        index = np.random.permutation(lr_pc.shape[0])[0:self.min_num_point]
        lr_pc = lr_pc[index, :]

        if hr_pc.shape[0] < self.max_num_point:
            add_num = self.max_num_point - hr_pc.shape[0]
            add_index = np.random.permutation(hr_pc.shape[0])[:add_num]
            add_points = hr_pc[add_index, :]
            add_normals = hr_n[add_index, :]
            hr_pc = np.concatenate((hr_pc, add_points), axis=0)
            hr_n = np.concatenate((hr_n, add_normals), axis=0)
        if hr_pc.shape[0] > self.max_num_point:
            index = np.random.permutation(hr_pc.shape[0])[:self.max_num_point]
            hr_pc = hr_pc[index, :]
            hr_n = hr_n[index, :]

        if self.mode == 'train' and self.rot:
            Rot = self.rotate_point_cloud_and_gt()
            lr_pc = np.dot(lr_pc, Rot)
            hr_pc = np.dot(hr_pc, Rot)
            hr_n = np.dot(hr_n, Rot)

        query_path = self.query_path[idx]

        npz_data = np.load(os.path.join(query_path, 'sample_gauss.npz'))

        sample_pc = npz_data['points']
        df = npz_data['df']
        closest_points = npz_data['closest_points']

        idx1 = (np.isnan(closest_points[:, 0]) == False) & (np.isnan(closest_points[:, 1]) == False) & (
                np.isnan(closest_points[:, 2]) == False) & (
                       np.isinf(closest_points[:, 0]) == False) & (np.isinf(closest_points[:, 1]) == False) & (
                       np.isinf(closest_points[:, 2]) == False)

        sample_pc = sample_pc[idx1, :]
        df = df[idx1]
        closest_points = closest_points[idx1, :]

        idx2 = np.random.choice(np.arange(closest_points.shape[0]), self.num_sample)

        sample_pc = sample_pc[idx2, :]
        df = df[idx2]
        closest_points = closest_points[idx2, :]

        return [
            torch.from_numpy(hr_pc).transpose(0, 1).float(),  # (3,RN)
            torch.from_numpy(lr_pc).transpose(0, 1).float(),  # (3,N)
            torch.from_numpy(hr_n).transpose(0, 1).float(),
            torch.from_numpy(sample_pc).transpose(0, 1).float(),
            torch.from_numpy(df).float(),
            torch.from_numpy(closest_points).transpose(0, 1).float()
        ]


    def __init__(self, data_path, return_file_path=False):
        super(scene_net, self).__init__()

        self.data_path = []
        self.return_file_path = return_file_path

        for cat in os.listdir(data_path):
            scene_list = os.listdir(os.path.join(data_path, cat))
            for s in scene_list:
                self.data_path.append(os.path.join(os.path.join(data_path, cat), s))

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):

        data_path = self.data_path[idx]
        pc = np.loadtxt(data_path)
        pc = pc[:, 0:3]
        n = pc[:, 3:6]

        if self.return_file_path:
            return [
                torch.from_numpy(pc).transpose(0, 1).float(),  # (3,RN)
                torch.from_numpy(n).transpose(0, 1).float(),  # (3,N)
                data_path
            ]
        return [
            torch.from_numpy(pc).transpose(0, 1).float(),  # (3,RN)
            torch.from_numpy(n).transpose(0, 1).float()  # (3,N)
        ]


    def __init__(self, prefix, mode='train'):

        self.prefix = prefix  # /root/data/Lap-Net/
        self.mode = mode
        self.data_path = []
        cat_list = os.listdir(os.path.join(prefix, mode))
        for cat in cat_list:
            self.data_path.append(cat)

        with h5py.File(os.path.join(prefix, "{}_My_Edge.h5").format(mode), "r") as hf_edge:
            self.edges = np.array(hf_edge.get("label")).astype(np.compat.long)  # [B, N]
            self.edges_W = np.array(hf_edge.get("W")).astype(np.float32)

    def __getitem__(self, idx):

        ret_list = []
        data_file = os.path.join(self.prefix, self.mode)

        with h5py.File(os.path.join(data_file, self.data_path[idx]), 'r') as hf:
            points = np.array(hf.get("points"))
            labels = np.array(hf.get("labels"))
            normals = np.array(hf.get("normals"))
            primitives = np.array(hf.get("prim"))
            primitive_param = np.array(hf.get("T_param"))

        # ret_list.append(torch.from_numpy(points).float())
        ret_list.append(points.astype(np.float32))
        ret_list.append(labels.astype(int))
        ret_list.append(normals.astype(np.float32))
        ret_list.append(primitives.astype(int))
        ret_list.append(self.edges[idx].astype(int))
        ret_list.append(self.edges_W[idx].astype(np.float32))
        ret_list.append(primitive_param.astype(np.float32))

        if self.mode == 'train':
            return self.random_points_list(ret_list)
        return ret_list

    def __len__(self):
        return len(self.data_path)

    def random_points_dict(self, items):
            l = np.arange(10000)
            np.random.shuffle(l)
            ret = {}
            for key in items.keys():
                if items[key].shape[0] == 10000:
                    ret[key] = items[key][l]
                else:
                    ret[key] = items[key]
            return ret

    def random_points_list(self, items):
            l = np.arange(10000)
            np.random.shuffle(l)
            ret = []
            for item in items:
                if item.shape[0] == 10000:
                    ret.append(item[l])
                else:
                    ret.append(item)
            return ret

    def __init__(self, prefix, mode='train', sparse_num=625):

        self.prefix = prefix  # /root/data/Lap-Net/
        self.mode = mode
        self.sparse_num = sparse_num
        self.data_path = []
        cat_list = os.listdir(os.path.join(prefix, mode))
        for cat in cat_list:
            self.data_path.append(cat)

        with h5py.File(os.path.join(prefix, "{}_My_Edge.h5").format(mode), "r") as hf_edge:
            self.edges = np.array(hf_edge.get("label")).astype(np.compat.long)  # [B, N]
            self.edges_W = np.array(hf_edge.get("W")).astype(np.float32)

    def __getitem__(self, idx):

        ret_list = []
        data_file = os.path.join(self.prefix, self.mode)

        with h5py.File(os.path.join(data_file, self.data_path[idx]), 'r') as hf:
            points = np.array(hf.get("points"))
        index = np.random.permutation(points.shape[0])[0:self.sparse_num]
        sparse_pc = points[index, :]
        # ret_list.append(torch.from_numpy(points).float())
        ret_list.append(points.astype(np.float32))
        ret_list.append(sparse_pc.astype(np.float32))

        if self.mode == 'train':
            return self.random_points_list(ret_list)
        return ret_list

    def __len__(self):
        return len(self.data_path)

    def random_points_dict(self, items):
            l = np.arange(10000)
            np.random.shuffle(l)
            ret = {}
            for key in items.keys():
                if items[key].shape[0] == 10000:
                    ret[key] = items[key][l]
                else:
                    ret[key] = items[key]
            return ret

    def random_points_list(self, items):
            l = np.arange(10000)
            np.random.shuffle(l)
            ret = []
            for item in items:
                if item.shape[0] == 10000:
                    ret.append(item[l])
                else:
                    ret.append(item)
            return ret


    def __init__(self, data_path, min_num_point=1024, max_num_point=16384):
        super(HumanForPu, self).__init__()
        self.min_num_point = min_num_point
        self.max_num_point = max_num_point

        self.data_path_npz = []
        self.data_path_npy = []
        cat_list_npz = os.listdir(os.path.join(data_path, 'pdc'))
        cat_list_npy = os.listdir(os.path.join(data_path, 'rand/' + str(min_num_point) + '/'))
        for cat in cat_list_npz:
            self.data_path_npz = self.data_path_npz + [os.path.join(os.path.join(data_path, 'pdc'), cat)]
        for cat in cat_list_npy:
            self.data_path_npy = self.data_path_npy + [os.path.join(os.path.join(data_path, 'rand/' + str(min_num_point) + '/'), cat)]

    def __len__(self):
        return len(self.data_path_npz)

    def __getitem__(self, idx):
        data_path_npz = self.data_path_npz[idx]
        # data_path_npy = self.data_path_npy[idx]
        # lr_pc = np.load(data_path_npy)
        # lr_pc = lr_pc[0:self.min_num_point, :]

        hr_data = np.load(data_path_npz)
        hr_pc = hr_data['sparse_points']
        hr_n = hr_data['sparse_normals']

        lr_pc = hr_pc[0:self.min_num_point, :]

        if hr_pc.shape[0] < self.max_num_point:
            add_num = self.max_num_point - hr_pc.shape[0]
            add_index = np.random.permutation(hr_pc.shape[0])[:add_num]
            add_points = hr_pc[add_index, :]
            add_normals = hr_n[add_index, :]
            hr_pc = np.concatenate((hr_pc, add_points), axis=0)
            hr_n = np.concatenate((hr_n, add_normals), axis=0)
        if hr_pc.shape[0] > self.max_num_point:
            index = np.random.permutation(hr_pc.shape[0])[:self.max_num_point]
            hr_pc = hr_pc[index, :]
            hr_n = hr_n[index, :]
        return [
            torch.from_numpy(hr_pc).transpose(0, 1).float(),  # (3,RN)
            torch.from_numpy(lr_pc).transpose(0, 1).float(),  # (3,N)
            torch.from_numpy(hr_n).transpose(0, 1).float()
        ]


    def __init__(self, prefix, max_num_point=16384, off_prefix='/root/FS_data/PU1K/test/original_meshes_scale_off', if_return_file=False):

        self.prefix = prefix  # /root/FS_data/PU1K/test/original_sample/
        self.off_prefix = off_prefix
        self.if_return_file = if_return_file
        self.max_num_point = max_num_point
        self.data_path_xyz = []
        cat_list_xyz = os.listdir(self.prefix)
        for cat in cat_list_xyz:
            self.data_path_xyz.append(cat)

    def __getitem__(self, idx):

        data_path = os.path.join(self.prefix, self.data_path_xyz[idx])
        data = np.load(data_path)
        sparse_pc = data['sparse_points']
        hr_pc = data['dense_points']
        if hr_pc.shape[0] < self.max_num_point:
            add_num = self.max_num_point - hr_pc.shape[0]
            add_index = np.random.permutation(hr_pc.shape[0])[:add_num]
            add_points = hr_pc[add_index, :]
            hr_pc = np.concatenate((hr_pc, add_points), axis=0)
        if hr_pc.shape[0] > self.max_num_point:
            index = np.random.permutation(hr_pc.shape[0])[:self.max_num_point]
            hr_pc = hr_pc[index, :]
        ret_list = []
        ret_list.append(torch.from_numpy(hr_pc).transpose(0, 1).float())
        ret_list.append(torch.from_numpy(sparse_pc).transpose(0, 1).float())
        if self.if_return_file:
            ret_list.append(os.path.join(self.off_prefix, self.data_path_xyz[idx][:-4]))
        return ret_list

    def __len__(self):
        return len(self.data_path_xyz)