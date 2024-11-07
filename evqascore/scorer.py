import torch
from models.clip import clip
from PIL import Image
import json
import cv2
import numpy as np
from tqdm import tqdm
import math
import time
from collections import defaultdict
from .utils import evqa_cos_score #


_MODELS = {
    "ViT-B/32": "./checkpoints/clip_ViT-B-32.pth",
    "open_clip_ViT-L/14": "./checkpoints/openClip_ViT-L-14.pth"
}
class EVQAScorer:
    """
    EMScore Scorer Object.
    """

    def __init__(self, vid_feat_cache=None, device=None,):
        self.vid_feat_cache = vid_feat_cache
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    def score(self, cands, key_cands, vids=None, verbose=True, batch_size=64, nthreads=4, return_matched_idx=False):
        """
        Args:
            - cands (list of str): candidate sentences
            - key_cands (list of str): key candidate sentences
            - vids (optional): video information for scoring
        
        Return:
            - final_results: Dictionary with precision, recall, and F-score.
        """
       
        model, preprocess = clip.load("ViT-B/32", device=self.device) 
        
        model = model.to(self.device).float()
        checkpoint = torch.load(_MODELS["ViT-B/32"])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        self._model = model
        self._tokenizer = clip.tokenize
        self._image_preprocess = preprocess

        ref_group_boundaries = None
        ori_cands, ori_key_cands = cands, key_cands
       
        if verbose:
            print("Calculating EVQAScore scores...")
            time_start = time.perf_counter()

        results = evqa_cos_score(
            self._model,
            cands,
            ori_cands,
            key_cands,
            ori_key_cands,
            vids,
            self.vid_feat_cache,
            self._tokenizer,
            self._image_preprocess,
            verbose=verbose,
            device=self.device,
            batch_size=batch_size,
            return_matched_idx=return_matched_idx
        )
        
        final_results = {}
        if vids:
            vid_all_local_preds  = results['vid_result']['figr']
            vid_all_global_preds = results['vid_result']['cogr']
            vid_P, vid_R, vid_F  = vid_all_local_preds[..., 0], vid_all_local_preds[..., 1], vid_all_local_preds[..., 2]   # P, R, F

            vid_results = {}
            vid_results['figr_P'] = vid_P
            vid_results['figr_R'] = vid_R
            vid_results['figr_F'] = vid_F
            vid_results['cogr'] = vid_all_global_preds
            vid_results['full_P'] = (vid_results['figr_P'] + vid_results['cogr'])/2
            vid_results['full_R'] = (vid_results['figr_R'] + vid_results['cogr'])/2
            vid_results['full_F'] = (vid_results['figr_F'] + vid_results['cogr'])/2
            final_results['EVQAScore(X,V)'] = vid_results

        if verbose:
            time_diff = time.perf_counter() - time_start
            print(f"done in {time_diff:.2f} seconds, {len(cands) / time_diff:.2f} sentences/sec")

        return final_results


