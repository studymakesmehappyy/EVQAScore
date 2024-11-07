import os
import argparse
import pickle
import numpy as np
import json
import glob
import torch
import math
from tqdm import tqdm
from evqascore import EVQAScorer
from evqascore.utils import compute_correlation_uniquehuman
from models import open_clip
from models.clip import clip



def get_feats_dict(feat_dir_path):
    """
    Load cached features from the given directory path.
    """
    print('loding cache feats ........')
    file_path_list = glob.glob(feat_dir_path+'/*.pt')
    feats_dict = {}
    for file_path in tqdm(file_path_list):
        vid = file_path.split('/')[-1][:-3]
        data = torch.load(file_path)
        feats_dict[vid] = data
    return feats_dict

def load_json(file_path):
    """
    Load and parse a JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_path', required=True, type=str, help='The path to the VATEX-EVAL dataset storage')
    parser.add_argument('--vid_base_path', default='', type=str, help='Path to VATEX-EVAL videos (optional if using cached video features)')
    parser.add_argument('--use_feat_cache', type=bool, default=True, help='Whether to use pre-prepared video features')
    parser.add_argument('--feat_cache_dir', type=str, default='', help='Directory for cached video features')
    parser.add_argument('--cands_path', required=True, type=str, help='Path to the JSON file containing candidate sentences')
    parser.add_argument('--key_cands_path', required=True, type=str, help='Path to the JSON file containing key candidate phrases')
    parser.add_argument('--video_ids_path', required=True, type=str, help='Path to JSON file with video IDs')
    
    opt = parser.parse_args()

    # Load candidate and key candidate data
    cands = load_json(opt.cands_path)
    key_cands = load_json(opt.key_cands_path)
    video_ids = load_json(opt.video_ids_path)
    vids = [os.path.join(opt.vid_base_path, vid + '.mp4') for vid in video_ids]
    all_human_scores = pickle.load(open(os.path.join(opt.storage_path, 'human_scores.pkl'), 'rb'))
    all_human_scores = np.transpose(all_human_scores.reshape(3, -1), (1, 0))

        # Prepare video features
    if not opt.use_feat_cache:
        print("Using custom video IDs without cached features.")
        metric = EVQAScorer(vid_feat_cache=[])
    else:
        video_clip_feats_dict = get_feats_dict(opt.feat_cache_dir)
        metric = EVQAScorer(vid_feat_cache=video_clip_feats_dict)      


    results = metric.score(cands=cands, key_cands=key_cands, vids=vids)
    
    
    if 'EMScore(X,V)' in results:
        print('EVQAScore(X,V) correlation --------------------------------------')
        vid_full_res_F = results['EMScore(X,V)']['full_F']
        compute_correlation_uniquehuman(vid_full_res_F.numpy(), all_human_scores)