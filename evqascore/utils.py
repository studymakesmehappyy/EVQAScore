import torch
#import clip
from PIL import Image
import json
import re
import cv2
import numpy as np
from tqdm import tqdm
import math
from math import log
from torch.nn.utils.rnn import pad_sequence
import sys
import time
import os
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
from itertools import chain

def compute_correlation_uniquehuman(pred, all_human_scores):
    num_workers = 3
    import scipy.stats

    pred = np.around(pred, decimals=4)

    spearman = 0
    for worker_i in range(num_workers):
        tmp, p_value = scipy.stats.spearmanr(pred, all_human_scores[:, worker_i])
        assert p_value < 0.01
        spearman += tmp
    spearman /= num_workers
    spearman = np.around(spearman, decimals=4)

    kendalltau = 0
    for worker_i in range(num_workers):
        tmp, p_value = scipy.stats.kendalltau(pred, all_human_scores[:, worker_i])
        assert p_value < 0.01
        kendalltau += tmp
    kendalltau /= num_workers
    kendalltau = np.around(kendalltau, decimals=4)

    print('kendall: {}, spear: {}'.format(kendalltau, spearman))
    return kendalltau, spearman

def normalize_matrix(A):
    return A / torch.linalg.norm(A, dim=-1, keepdim=True)


def encode_video(video_file, preprocess, model, batch_size, device):
    print(f"Processing video: {video_file}")
    # 提取文件名中的时间区间，例如 "0qOFqf_eRk_000016_000026.mp4"
    match = re.search(r'_(\d{6})_(\d{6})', video_file)
    if match:
        start_time = int(match.group(1))  # 起始时间（秒）
        end_time = int(match.group(2))    # 结束时间（秒）
    else:
        raise ValueError("无法从视频文件名中提取时间区间")

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数

 #   将时间转换为帧数
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    start_frame = 0
    end_frame = total_frames  # 解析到视频结束

  #  frame_interval = int(fps) #
    frame_interval = 10 #

    # 跳转到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    images = []
    count = start_frame
    ret = True

    # 只提取时间区间内的帧
    while count < end_frame and ret:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval != 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))
        count += 1

    cap.release()

    # 如果没有提取到帧
    if not images:
        print(f"No frames extracted for {video_file}")
        return None, None

    # 转换为 tensor 并传输到 GPU

    image_input = torch.tensor(np.stack(images)).to(device)
    image_features_list = []

    # 执行模型推理
    with torch.no_grad():
        n_inter = math.ceil(len(image_input) / batch_size)
        for i in range(n_inter):
            image_features = model.encode_image(image_input[i*batch_size: (i+1)*batch_size]).float()
            image_features_list.append(image_features)

    image_features = torch.cat(image_features_list, dim=0)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    vid_feature = normalize_matrix(torch.mean(image_features, dim=0, keepdim=True)).squeeze()

    # 保存特征到 .pt 文件
 #   save_dir = './UniformFrameFeat_10'
 #   video_name = os.path.basename(video_file).split('.')[0]  # 提取视频文件名
 #   save_path = os.path.join(save_dir, f"{video_name}.pt")   # 创建保存路径
 #   torch.save(image_features, save_path)  # 保存特征到 .pt 文件

    return image_features, vid_feature

def encode_text(vid_caps, model, tokenizer, device):
    text_input = tokenizer(vid_caps).to(device=device)
    with torch.no_grad():
        text_features = model.encode_text(text_input, local=True).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    txt_len = text_input.argmax(dim=-1)
    mask = torch.zeros_like(text_input)
    for i in range(len(mask)):
        mask[i][0:txt_len[i]+1] = 1

    return text_features, mask



def vid_greedy_cos(ref_embedding, ref_masks, hyp_embedding, hyp_masks, ori_hpy_stats_nums, return_matched_idx):

    batch_size = ref_embedding.size(0)


    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))

    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)
    masks = masks.float().to(sim.device)
    sim = sim * masks

    word_precision, matched_indices = sim.max(dim=2)
    word_recall = sim.max(dim=1)[0]

    phrase_weight = torch.tensor([torch.log(torch.tensor(num + 1, dtype=torch.float32)) for num in ori_hpy_stats_nums], device=sim.device)

    # 直接计算总单词数作为分母
    num_hyp_words = hyp_masks.sum(dim=1)  # 候选文本的总单词数
    num_ref_words = ref_masks.sum(dim=1)  # 参考文本的总单词数

    # 计算 P 和 R 时不再使用 IDF 加权
    P = word_precision.sum(dim=1) / num_hyp_words
    R = word_recall.sum(dim=1) / num_ref_words
    
    F = 2 * P * R / (P + R) * phrase_weight
    
    if return_matched_idx:
        return P, R, F, matched_indices
    else:
        return P, R, F, torch.zeros_like(P)



def evqa_cos_score(
    model, cands, ori_cands, key_cands, ori_key_cands, vids, vid_feat_cache, tokenizer, preprocess, verbose=True, batch_size=64, device="cuda:0", return_matched_idx=False
):

    vid_preds_local = []
    vid_pred_matched_idxs = []
    vid_preds_global = []

    """process text"""
    def dedup_and_sort(l):
        return sorted(list(set(l)), key=lambda x: len(x.split(" ")), reverse=True)
    
    sentences = dedup_and_sort(cands)
    print('sentences:', sentences)
    embs = []
    iter_range = range(0, len(sentences), batch_size)
    if verbose:
        print("computing text embedding.")
        iter_range = tqdm(iter_range)
    text_local_stats_dict = dict()
    text_global_stats_dict = dict()
    text_encode_time_total = 0
    for batch_start in iter_range:
        sen_batch = sentences[batch_start: batch_start + batch_size]
        embs, masks = encode_text(sen_batch, model, tokenizer, device=device)
        embs = embs.cpu()
        masks = masks.cpu()
        for i, sen in enumerate(sen_batch):
            sequence_len = masks[i].sum().item()
            
            local_emb = embs[i, 0:sequence_len]

            global_emb = embs[i, sequence_len-1]

            text_local_stats_dict[sen] = (local_emb)

            text_global_stats_dict[sen] = global_emb

    key_sentences = dedup_and_sort(key_cands)
    key_embs = []

    key_iter_range = range(0, len(key_sentences), batch_size)
    if verbose:
        print("computing text embedding for keywords.")
        key_iter_range = tqdm(key_iter_range)
    key_text_local_stats_dict = dict()
    key_text_global_stats_dict = dict()
    text_nums_stats_dict = dict()  # for key phrase
    for batch_start in key_iter_range:
        key_sen_batch = key_sentences[batch_start: batch_start + batch_size]
        key_embs, key_masks = encode_text(key_sen_batch, model, tokenizer, device=device)
        key_embs = key_embs.cpu()
        key_masks = key_masks.cpu()
        for i, sen in enumerate(key_sen_batch):
            sequence_len = key_masks[i].sum().item()
            key_local_emb = key_embs[i, 0:sequence_len]
            key_global_emb = key_embs[i, sequence_len-1]
            key_text_local_stats_dict[sen] = key_local_emb

            pharses = sen.split(",") # for key phrase
            text_nums_stats_dict[sen] = len(pharses)

            key_text_global_stats_dict[sen] = key_global_emb

    """process video"""
    if vids:
        if vid_feat_cache:
            ori_vids = vids
            vid_local_stats_dict = vid_feat_cache
            vid_global_stats_dict = dict()
            for vid in vid_local_stats_dict:
                image_features = vid_local_stats_dict[vid]
                vid_feature = normalize_matrix(torch.mean(image_features, dim=0, keepdim=True)).squeeze()
                vid_global_stats_dict[vid] = vid_feature
        else:
            ori_vids = vids # video paths list
            unique_vids = list(set(vids))
            if verbose:
                print("computing vid embedding.")
            vid_local_stats_dict = dict()
            vid_global_stats_dict = dict()
            for vid_i in tqdm(range(len(unique_vids))):
                video_file = unique_vids[vid_i]
                image_features, vid_feature = encode_video(video_file, preprocess, model, batch_size=512, device=device)
                # vid_name = video_file.split('/')[-1][:-4]
                vid_local_stats_dict[video_file] = image_features.cpu()
                vid_global_stats_dict[video_file] = vid_feature.cpu()


    def pad_local_batch_stats(sen_batch, stats_dict, device):
        stats = [stats_dict[s] for s in sen_batch]

        emb = [e.to(device) for e in stats]
        emb = [e.to(device) for e in emb]
        lens = [e.size(0) for e in emb]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=0.0)

        def length_to_mask(lens):
            lens = torch.tensor(lens, dtype=torch.long)
            max_len = max(lens)
            base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
            return base < lens.unsqueeze(1)

        pad_mask = length_to_mask(lens).to(device)
        return emb_pad, pad_mask

    def pad_vid_local_batch_stats(sen_batch, stats_dict, device):
        stats = [stats_dict[s] for s in sen_batch]
        emb = stats
        emb = [e.to(device) for e in emb]
        lens = [e.size(0) for e in emb]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=0.0)

        def length_to_mask(lens):
            lens = torch.tensor(lens, dtype=torch.long)
            max_len = max(lens)
            base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
            return base < lens.unsqueeze(1)

        pad_mask = length_to_mask(lens).to(device)
        return emb_pad, pad_mask
    
    def pad_global_batch_stats(sen_batch, stats_dict, device):
        stats = [stats_dict[s] for s in sen_batch]
        emb = stats
        emb = [e.to(device) for e in emb]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=0.0)
        return emb_pad
        
    """ if video used as ground truth """
    if vids:
        if verbose:
            print("computing greedy matching, video as ground truth.")
        iter_range = range(0, len(ori_cands), batch_size)
        with torch.no_grad():
            for batch_start in iter_range: 
                batch_ori_hyp = ori_cands[batch_start: batch_start + batch_size]
                ori_hyp_stats_global = pad_global_batch_stats(batch_ori_hyp, text_global_stats_dict, device)
                
                ori_hpy_stats_nums = [text_nums_stats_dict[s] for s in batch_ori_hyp]


                batch_ori_vids = ori_vids[batch_start: batch_start + batch_size]
                ori_vids_stats_local = pad_vid_local_batch_stats(batch_ori_vids, vid_local_stats_dict, device)
                ori_vids_stats_global = pad_global_batch_stats(batch_ori_vids, vid_global_stats_dict, device)
                
                batch_ori_key_cands = ori_key_cands[batch_start: batch_start + batch_size]  # 候选关键词的批次
                ori_key_hyp_stats_local = pad_local_batch_stats(batch_ori_key_cands, key_text_local_stats_dict, device)

                P, R, F1, matched_indices = vid_greedy_cos(*ori_vids_stats_local, *ori_key_hyp_stats_local, ori_hpy_stats_nums,return_matched_idx)
              
                vid_preds_local.append(torch.stack((P, R, F1), dim=-1).cpu())
                vid_pred_matched_idxs.append(matched_indices)

                vid_s_cogr = torch.bmm(ori_hyp_stats_global.unsqueeze(1), ori_vids_stats_global.unsqueeze(1).transpose(1, 2)).squeeze()
                vid_preds_global.append(vid_s_cogr)  
    


    results = dict()
    """ if video used as ground truth """
    if vids:
        vid_preds_local = torch.cat(vid_preds_local, dim=0).cpu()
        if len(vids) != 1:
            vid_preds_global = torch.cat(vid_preds_global, dim=0).cpu()
        else:
            vid_preds_global = vid_preds_global[0].cpu()
        results['vid_result'] = {}
        results['vid_result']['figr'] = vid_preds_local
        results['vid_result']['cogr'] = vid_preds_global
        results['vid_result']['matched_indices'] = torch.cat(vid_pred_matched_idxs)

    return results