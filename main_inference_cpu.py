import logging
import os
import shutil
from itertools import groupby
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.python.saved_model.model_utils.mode_keys import is_train

from lib.config import get_cfg
from lib.dataset import build_data_loader_inference
from lib.engine import default_argument_parser, default_setup
from lib.model import SignModel
from lib.solver import build_lr_scheduler, build_optimizer
from lib.utils import AverageMeter, clean_ksl
import time
import pickle
import cv2

def setup(args):
    """
    Create configs and perform basic setups.
    """

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg = default_setup(cfg, args)
    return cfg
import glob

def arr2sen(sequences, vocab):
    # Reverse the vocab dictionary to map from index to word
    index_to_word = {index: word for word, index in vocab.items()}
    # Convert each sequence in sequences
    words_sequences = []
    for sequence in sequences:
        words_sequence = [index_to_word.get(idx, '<unk>') for idx in sequence]
        words_sequences.append(words_sequence)

    return words_sequences

def main(args):
    device = torch.device("cpu")
    start = time.time()
    logger = logging.getLogger()

    cfg = setup(args)
    cfg.freeze()
    with open('vocab.pickle', 'rb') as f:
        vocab = pickle.load(f)
    _, test_loader = build_data_loader_inference(cfg)

    vocab = {'<si>': 0, '<unk>': 1, '<pad>': 2, '차내리다': 3, '곳': 4, '버스': 5, '내리다': 6, '맞다': 7, '전': 8, '지름길': 9, '송파': 10,
     '지하철': 11, '무엇': 12, '가다': 13, '방법': 14, '여기': 15, '목적': 16, '건너다': 17, '명동': 18, '보다': 19, '시청': 20, '신호등': 21,
     '저기': 22, '다음': 23, '도착': 24, '우회전': 25, '좌회전': 26, '찾다': 27, '길': 28}
    # test1 = time.time()
    # frames_path = glob.glob(os.path.join(os.path.join(cfg.DATASET.DATA_ROOT, cfg.DATASET.VAL.IMG_PREFIX), "*.jpg"))
    # frames_path.sort()
    # try:
    #     frames = np.stack([cv2.imread(frames_path[i], cv2.IMREAD_COLOR) for i in range(len(frames_path))], axis=0)
    # except ValueError:
    #     print('non img')
    # test2 = time.time()
    # test_dataset, _ = transform_image(cfg, frames, is_train=False)
    # test3 = time.time()
    # print(test3-test2, test2-test1)

    end_data = time.time()
    print('data', end_data-start)
    model = SignModel(vocab)
    # model = model.cuda()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    end_model_set = time.time()
    print('model setting', end_model_set-end_data)
    assert os.path.isfile(cfg.RESUME), "Error: no checkpoint directory found!"
    checkpoint = torch.load(cfg.RESUME, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    current_lr = optimizer.param_groups[0]["lr"]
    logger.info(
        "Loaded checkpoint from {}.  "
        "start_epoch: {cp[epoch]} current_lr: {lr:.5f}  "
        "recoded WER: {cp[wer]:.3f} (best: {cp[best_wer]:.3f})".format(
            cfg.RESUME, cp=checkpoint, lr=current_lr
        )
    )
    model = model.to(device)
    end_model_load = time.time()
    print('model load', end_model_load-end_model_set)
    test_dataset = test_loader.dataset.__getitem__(0)[0]
    end_test_dataset = time.time()
    print('test_dataset get', end_test_dataset - end_model_set)
    validate(model, test_dataset, vocab, device)
    end = time.time()

    print('Inference time: ', end - end_test_dataset, 's')

    print('Total time: ', end - start, 's')

def validate(model, val_dataset, vocab, device):
    logger = logging.getLogger()

    model.eval()

    all_glosses = []

    with torch.no_grad():
        val_dataset = val_dataset.unsqueeze(0).to(device).detach()
        print(val_dataset.shape, len(val_dataset))
        video_lengths = np.array([val_dataset.shape[1]])

        gloss_scores = model(val_dataset)  # (B, T, C)

        gloss_probs = gloss_scores.log_softmax(2).permute(1, 0, 2)  # (T, B, C)
        gloss_probs = gloss_probs.detach().numpy()  # (T, B, C)
        gloss_probs_tf = np.concatenate(
            # (C: 1~)
            (gloss_probs[:, :, 1:], gloss_probs[:, :, 0, None]),
            axis=-1,
        )
        sequence_length = video_lengths // 4
        ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
            inputs=gloss_probs_tf,
            sequence_length=sequence_length,
            beam_width=1,
            top_paths=1,
        )
        ctc_decode = ctc_decode[0]

        # Create a decoded gloss list for each sample
        tmp_gloss_sequences = [[] for i in range(gloss_scores.shape[0])]  # (B, )
        for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
            tmp_gloss_sequences[dense_idx[0]].append(ctc_decode.values[value_idx].numpy() + 1)
        decoded_gloss_sequences = []
        for seq_idx in range(0, len(tmp_gloss_sequences)):
            decoded_gloss_sequences.append(
                [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
            )
        all_glosses.extend(decoded_gloss_sequences)

    print(len(val_dataset))
    assert len(all_glosses) == len(val_dataset)
    decoded_gls = arr2sen(all_glosses,vocab)
    print(all_glosses, decoded_gls)
    # Gloss clean-up function
    
    # Construct gloss sequences for metrics
    gls_hyp = [clean_ksl(" ".join(t)) for t in decoded_gls]

    
    # GLS Metrics
    print('Predicted gloss: ', gls_hyp)

import logging

import numpy as np
from fvcore.transforms.transform import Transform, TransformList
from torch import Tensor

from typing import List, Tuple

from lib.dataset.transforms.transform_gen import CenterCrop, RandomCrop, Resize, ToTensorGen, TransformGen

def transform_image(cfg, img: np.ndarray, is_train: bool = True) -> Tuple[np.ndarray, TransformList]:
    """
    Apply a series of transformations to an image based on the given configuration.

    Args:
        cfg: Configuration object containing transformation parameters.
        img (ndarray): Input image as a numpy array.
        is_train (bool): Flag to indicate if the transformation is for training or validation.

    Returns:
        ndarray: The transformed image.
        TransformList: A list of transforms that were applied to the image.
    """
    logger = logging.getLogger(__name__)

    resize = cfg.DATASET.TRANSFORM.RESIZE_IMG
    ts = cfg.DATASET.TRANSFORM.TEMPORAL_SCALING
    crop = cfg.DATASET.TRANSFORM.CROP_SIZE
    tc = cfg.DATASET.TRANSFORM.TEMPORAL_CROP_RATIO
    norm_params = dict(mean=cfg.DATASET.TRANSFORM.MEAN, std=cfg.DATASET.TRANSFORM.STD)
    tfm_gens = []

    # Resize transformation
    tfm_gens.append(Resize(resize, temporal_scaling=ts, interp="trilinear"))

    # Crop transformation based on training or validation mode
    if is_train:
        tfm_gens.append(RandomCrop("absolute", crop, temporal_crop_ratio=tc))
    else:
        tfm_gens.append(CenterCrop(crop, temporal_crop_ratio=tc))

    # Normalize and convert to tensor
    tfm_gens.append(ToTensorGen(normalizer=norm_params))

    # Logging the applied transformations
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    else:
        logger.info("TransformGens used in validation: " + str(tfm_gens))

    # Apply the transformations
    for g in tfm_gens:
        assert isinstance(g, TransformGen), f"Expected TransformGen, got {type(g)}"

    check_dtype(img)

    tfms = []
    for g in tfm_gens:
        tfm = g.get_transform(img)
        assert isinstance(tfm, Transform), f"TransformGen {g} must return an instance of Transform! Got {tfm} instead"
        img = tfm.apply_image(img)
        tfms.append(tfm)

    return img, TransformList(tfms)

def check_dtype(img):
    assert isinstance(img, np.ndarray), "[TransformGen] Needs an numpy array, but got a {}!".format(
        type(img)
    )
    assert not isinstance(img.dtype, np.integer) or (
        img.dtype == np.uint8
    ), "[TransformGen] Got image of type {}, use uint8 or floating points instead!".format(
        img.dtype
    )
    assert img.ndim in [3, 4], img.ndim
if __name__ == "__main__":
    
    args = default_argument_parser().parse_args()
    args.config_file='configs/config_inference.yaml'
    world_size = torch.cuda.device_count() # new

    main(args)
