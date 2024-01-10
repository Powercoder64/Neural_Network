import torch.utils.data as data
import os
import csv
import json
import numpy as np
import torch
import pdb
import time
import random
import utils
import config


class ThumosFeature(data.Dataset):
    def __init__(self, data_path, mode, modal, feature_fps, num_segments, len_feature, sampling, seed=-1, supervision='no_weak'):
        if seed >= 0:
            utils.set_seed(seed)

        self.mode = mode
        self.modal = modal
        self.feature_fps = feature_fps
        self.num_segments = num_segments
        self.len_feature = len_feature

        if self.modal == 'all':
            self.feature_path = []
            for _modal in ['rgb', 'flow']:
                self.feature_path.append(os.path.join(data_path, 'features', self.mode, _modal))
        else:
            self.feature_path = os.path.join(data_path, 'features', self.mode, self.modal)

        #split_path = os.path.join(data_path, 'split_{}.txt'.format(self.mode))
        #split_file = open(split_path, 'r')
        #self.vid_list = []
        #for line in split_file:
            #self.vid_list.append(line.strip())
        #split_file.close()
        self.vid_list = os.listdir(self.feature_path[0])

        #anno_path = os.path.join(data_path, 'gt.json')
        #anno_file = open(anno_path, 'r')
        #self.anno = json.load(anno_file)
        #anno_file.close()

        self.class_name_to_idx = dict((v, k) for k, v in config.class_dict.items())        
        self.num_classes = len(self.class_name_to_idx.keys())

        self.supervision = supervision
        self.sampling = sampling


    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data, vid_num_seg, sample_idx = self.get_data(index)
        #label, temp_anno = self.get_label(index, vid_num_seg, sample_idx)
        return data, self.vid_list[index], vid_num_seg

    def get_data(self, index):
        vid_name = self.vid_list[index]
        #print(vid_name)

        vid_num_seg = 0

        if self.modal == 'all':
            rgb_feature = np.load(os.path.join(self.feature_path[0],
                                    vid_name)).astype(np.float32)
            flow_feature = np.load(os.path.join(self.feature_path[1],
                                    vid_name)).astype(np.float32)

            vid_num_seg = rgb_feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(rgb_feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(rgb_feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')
            #print(rgb_feature.shape)

            rgb_feature = rgb_feature[sample_idx]
            flow_feature = flow_feature[sample_idx]
            #np.save('temp_egb_features_' + str(vid_name), rgb_feature)
            #np.save('temp_flow_features_' + str(vid_name), flow_feature)


            feature = np.concatenate((rgb_feature, flow_feature), axis=1)
        else:
            feature = np.load(os.path.join(self.feature_path,
                                    vid_name + '.npy')).astype(np.float32)

            vid_num_seg = feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')

            feature = feature[sample_idx]

        return torch.from_numpy(feature), vid_num_seg, sample_idx

    #def get_label(self, index, vid_num_seg, sample_idx):
        #vid_name = self.vid_list[index]
        #print(vid_name)
        #anno_list = self.anno['database'][vid_name]['annotations']

        #label = np.zeros([self.num_classes], dtype=np.float32)
        #print(label)

        #classwise_anno = [[]] * self.num_classes

        #for _anno in anno_list:
            #print(anno_list)
            #label[self.class_name_to_idx[_anno['label']]] = 1
            #classwise_anno[self.class_name_to_idx[_anno['label']]] = np.append(classwise_anno[self.class_name_to_idx[_anno['label']]], _anno)
            #print(str(_anno) + ' ' + str(self.class_name_to_idx[_anno['label']]))
            #print([_anno['label']])
            #print(_anno['label'])
            #print(classwise_anno)

        #np.save('anno', classwise_anno)
        #if self.supervision == 'weak':

            #return label, torch.Tensor(0)

        #else:
            #temp_anno = np.zeros([vid_num_seg, self.num_classes])
            #t_factor = self.feature_fps / 16
            #print(vid_num_seg)
            #for class_idx in range(self.num_classes):
                #if label[class_idx] != 1:
                    #continue

                #for _anno in classwise_anno[class_idx]:
                    #print(str(_anno) + ' ' + str(class_idx))
                    #print((classwise_anno[0]))
                    #tmp_start_sec = float(_anno['segment'][0])
                    #print(tmp_start_sec )

                    #tmp_end_sec = float(_anno['segment'][1])
                    #tmp_start = round(tmp_start_sec * t_factor)
                    #tmp_end = round(tmp_end_sec * t_factor)
                    #print(str(tmp_start) + ' ' + str(tmp_end))
                    #temp_anno[tmp_start:tmp_end+1, class_idx] = 1
                    #print(class_idx)
                    #.save('temp_anno_' +str(vid_name), temp_anno)

            #np.save('labels', temp_anno)
            #print(sample_idx)
            #temp_anno = temp_anno[sample_idx, :]
            #np.save('temp_anno_' + str(vid_name), temp_anno)
            #print(sample_idx)
            #np.save('labels', temp_anno)

            #

            #print(temp_anno.shape)

            #return  torch.from_numpy(temp_anno), label

    def random_perturb(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)


    def uniform_sampling(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        samples = np.floor(samples)
        return samples.astype(int)




