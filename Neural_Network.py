import pdb
import numpy as np
import torch.utils.data as data
import utils
from options import *
from config import *
from test import *
from model import *
from tensorboard_logger import Logger
from thumos_features import *
import requests

def post_process_status(process_name, status, error_message=''):
    # Placeholder for the real API URL

    message_api_url = "https://<real_rc_uva_url>"
    headers = {

        'API-Key': '<your_api_key_here>',
        'API-Secret': '<your_api_secret_here>'

    }

    # Setting up the payload with process details

    payload = {

        'processName': process_name,
        'processStatus': status,  # can be: 'started', 'completed', 'error'
        'errorMessage': error_message  # Informative string for errors, empty for start/stop

    }

    # Making the POST request

    response = requests.post(message_api_url, json=payload, headers=headers)

    return response


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()

    config = Config(args)
    worker_init_fn = None

    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)

    net = BaS_Net(config.len_feature, config.num_classes, config.num_segments)
    net = net.cuda()

    test_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='test',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='random'),

        batch_size=1,
        shuffle=False, num_workers=config.num_workers,
        worker_init_fn=worker_init_fn)

    test_info = {"step": [], "test_acc": [], "average_mAP": [],
                 "mAP@0.1": [], "mAP@0.2": [], "mAP@0.3": [],
                 "mAP@0.4": [], "mAP@0.5": [], "mAP@0.6": [],
                 "mAP@0.7": [], "mAP@0.8": [], "mAP@0.9": []}

    logger = Logger(config.log_path)

    process_name = "Neural_Network"

    try:

        post_process_status(process_name, 'started')
        print(post_process_status(process_name , 'started').text)

        test(net, config, logger, test_loader, test_info, 0, model_file=config.model_file)

        post_process_status(process_name, 'completed')
        print(post_process_status(process_name, 'completed').text)
        utils.save_best_record_thumos(test_info,
                                      os.path.join(config.output_path, "best_record.txt"))
    except Exception as e:

        error_message = str(e)

        post_process_status(process_name, 'error', error_message)
        print(post_process_status(process_name, 'error', error_message).text)


