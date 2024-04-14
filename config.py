import numpy as np
import os

class Config(object):
    def __init__(self, args):
        self.lr = eval(args.lr)
        self.num_iters = len(self.lr)
        self.num_classes = 24
        self.modal = args.modal
        if self.modal == 'all':
            self.len_feature = 2048
        else:
            self.len_feature = 1024
        self.batch_size = args.batch_size
        self.data_path = args.data_path
        self.model_path = args.model_path
        self.messageid = args.messageid
        self.requesttype = args.requesttype
        self.filename = args.filename
        self.output_path = '/home/project/' + args.filename + '/' + args.messageid + '/matrixfiles'
        self.log_path = args.log_path
        self.num_workers = args.num_workers
        self.alpha = args.alpha
        self.class_thresh = args.class_th
        self.act_thresh = np.arange(0.0, 0.25, 0.025)
        self.scale = 24
        self.gt_path = os.path.join(self.data_path, 'gt.json')
        self.model_file = args.model_file
        self.seed = args.seed
        self.feature_fps = 25
        self.num_segments = 5000


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
