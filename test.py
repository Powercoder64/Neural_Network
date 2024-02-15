import torch
import torch.nn as nn
import numpy as np
import utils
import os
import json
from tqdm import tqdm
from sklearn import metrics
from scipy.interpolate import interp1d
import cv2

class_dict = {0: 'Whole_Class_Activity',
              1: 'Individual_Activity',
              2: 'Small_Group_Activity',
              3: 'Book-Using_or_Holding',
              4: 'Instructional_Tool-Using_or_Holding',
              5: 'Student_Writing',
              6: 'Teacher_Writing',
              7: 'Raising_Hand',
              8: 'Presentation_with_Technology',
              9: 'Individual_Technology',
              10: 'Worksheet-Using_or_Holding',
              11: 'Notebook-Using_or_Holding',
              12: 'Student(s)_Carpet_or_Floor-Sitting',
              13: 'Student(s)_Desks-Sitting',
              14: 'Student(s)_Group_Tables-Sitting',
              15: 'Student(s)_Standing_or_Walking',
              16: 'Teacher_Sitting',
              17: 'Teacher_Standing_(T)',
              18: 'Teacher_Walking',
              19: 'Teacher_Supporting_One_Student',
              20: 'Teacher_Supporting_Multiple_with_SS_Interaction',
              21: 'Teacher_Supporting_Multiple_without_SS_Interaction',
              22: 'On_Task_Student_Talking_with_Student',
              23: 'Transition'}


def results_process(results, output_path):
    confi_thresh = np.array([0.00024,0.00024,0.00019,0.00025,0.00018,0.00020,0.00045,
    0.00042,0.00029,0.00025,0.00017,0.00023,0.00017,0.00018,0.00017,0.00021,0.00010,
    0.00020,0.00047,0.00027,0.00020,0.00022,0.00036,0.00047])

    results_all = []

    for i in range(0, 24):
    
        #print(results.shape)   
        results_i = results.reshape(results.shape[0] * results.shape[1], results.shape[2])[:, i]
        results_i[np.where(results_i < confi_thresh[i])] = 0
        results_i[np.where(results_i >= confi_thresh[i])] = 1
        results_i = results_i.reshape(results.shape[0], results.shape[1], 1)
        results_all.append(results_i)

    results_all = np.concatenate(results_all, axis=2)

    text_file = open("split_test.txt", "r")
    lines = text_file.readlines()
    lines = [s.strip() for s in lines]
    text_file.close()

    def scale_binary(signal, scale_factor):
        n = len(signal)
        x = np.arange(n)
        f = interp1d(x, signal, kind='nearest')
        xnew = np.linspace(0, n - 1, num=int(n * scale_factor))
        return np.round(f(xnew)).astype(int)


    
    
    path_vid  = './video/'
    vid_names = os.listdir(path_vid)
    #print(vid_names)
    num_frames=[]
    for k in range(0, len(vid_names)):
        cap = cv2.VideoCapture(path_vid + vid_names[k])
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames.append(length)
    
    sample_size = results.shape[1]
    #print(sample_size)
    #scale_f = 19.2

    output_dir = output_path

    for i in range(0, results.shape[0]):
        text_whole = np.zeros([results_all.shape[2] + 1, num_frames[i] + 1], dtype=object)
        text_whole[0, 0] = 'acivity'

        for k in range(0, num_frames[i]):
            text_whole[0, k + 1] = 'frame_' + str(k + 1)

        for j in range(0, results_all.shape[2]):
            class_j = results_all[i][:, j]
            text_whole[j + 1, 0] = class_dict[j]

            class_j_s = scale_binary(class_j, (1.0) * (num_frames[i]/sample_size))
            text_whole[j + 1][1:len(class_j_s) + 1] = class_j_s

        np.savetxt(output_dir + lines[i] + '.csv', text_whole, fmt='%s', delimiter=",")


def test(net, config, logger, test_loader, test_info, step, model_file=None):
    with torch.no_grad():
        net.eval()

        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        final_res = {}
        final_res['version'] = 'VERSION 1.3'
        final_res['results'] = {}
        final_res['external_data'] = {'used': True, 'details': 'Features from I3D Network'}


        load_iter = iter(test_loader)
        score_np_list = []
        label_np_list = []
        for i in range(len(test_loader.dataset)):
            _data, vid_name, vid_num_seg = next(load_iter)

            print(vid_name)

            _data = _data.cuda()
            #print(_data.shape)
            #_label = _label.cuda()

            _, cas_base, score_supp, cas_supp, fore_weights = net(_data)

            #label_np = _label.cpu().numpy()
            score_np = score_supp.cpu().data.numpy()

            score_np = score_np[:, :, 0:-1].squeeze()
            #label_np = label_np.squeeze()

            score_np_list.append(score_np)
            #label_np_list.append(label_np)
    results_process(np.array(score_np_list), config.output_path)
