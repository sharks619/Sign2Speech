import logging
import os
import shutil
from itertools import groupby
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from lib.config import get_cfg
from lib.dataset import build_data_loader_inference
from lib.engine import default_argument_parser, default_setup
from lib.model import SignModel
from lib.solver import build_lr_scheduler, build_optimizer
from lib.utils import AverageMeter, clean_ksl
import time
import pickle

def setup(args):
    """
    Create configs and perform basic setups.
    """

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg = default_setup(cfg, args)
    return cfg

def main(args):
    start = time.time()
    logger = logging.getLogger()

    cfg = setup(args)
    cfg.freeze()
    train_loader, test_loader = build_data_loader_inference(cfg)
    print(train_loader.dataset.vocab)

    end_data = time.time()
    print('data', end_data-start)
    loss_gls = nn.CTCLoss(blank=train_loader.dataset.sil_idx, zero_infinity=True).to(cfg.GPU_ID)
    model = SignModel(train_loader.dataset.vocab)
    # model = model.cuda()
    model = model.to(cfg.GPU_ID)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    end_model_set = time.time()
    print('model setting', end_model_set-end_data)
    assert os.path.isfile(cfg.RESUME), "Error: no checkpoint directory found!"
    checkpoint = torch.load(cfg.RESUME)
    best_wer=checkpoint['best_wer']
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint['state_dict'])
    # model = nn.DataParallel(model).cuda()
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
    end_model_load = time.time()
    print('model load', end_model_load-end_model_set)
    validate(cfg,model, test_loader)
    end = time.time()

    print('Inference time: ', end - end_model_load, 's')

def validate(cfg, model, val_loader, ) -> dict:
    logger = logging.getLogger()

    model.eval()

    all_glosses = []
    loader_iter = iter(val_loader)
    vocab = decoder = val_loader.dataset.vocab
    decoder = vocab.arrays_to_sentences
    for _iter in range(len(val_loader)):
        with torch.no_grad():
            (videos, video_lengths), _ = next(loader_iter)
            videos = videos.to(cfg.GPU_ID)
            video_lengths = video_lengths.to(cfg.GPU_ID)
            print(videos.shape, videos, video_lengths)
            gloss_scores = model(videos)  # (B, T, C)

            gloss_probs = gloss_scores.log_softmax(2).permute(1, 0, 2)  # (T, B, C)
            gloss_probs = gloss_probs.cpu().detach().numpy()  # (T, B, C)
            gloss_probs_tf = np.concatenate(
                # (C: 1~)
                (gloss_probs[:, :, 1:], gloss_probs[:, :, 0, None]),
                axis=-1,
            )
            sequence_length = video_lengths.long().cpu().detach().numpy() // 4
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
            decoded = decoder(arrays=decoded_gloss_sequences)
            for _dec in decoded[:4]:
                logger.info(" ".join(_dec))
                print()

    assert len(all_glosses) == len(val_loader.dataset)
    decoded_gls = val_loader.dataset.vocab.arrays_to_sentences(arrays=all_glosses)
    # Gloss clean-up function
    
    # Construct gloss sequences for metrics
    gls_hyp = [clean_ksl(" ".join(t)) for t in decoded_gls]

    
    # GLS Metrics
    print('Predicted gloss: ', gls_hyp)



def save_checkpoint(
    state_dict: dict, is_best: bool, checkpoint: str, filename='checkpoint.pth.tar'
):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state_dict, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


if __name__ == "__main__":
    
    args = default_argument_parser().parse_args()
    args.config_file='configs/config_inference.yaml'
    world_size = torch.cuda.device_count() # new

    main(args)
