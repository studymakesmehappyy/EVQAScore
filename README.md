# EVQAScore: Efficient Video Question Answering Data Evaluation

## Overview
EVQAScore is a reference-free evaluation metric designed to assess the quality of video question-answering (QA) data. By using keyword extraction, frame sampling, and rescaling techniques, it evaluates both video captions and QA data efficiently, even for long videos.

## Environment Setup
Clone the repository and create the EVQA conda environment using the `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate evqa

## Usage
Follow these steps to run `evqascore-demo.py`:

1. **Prepare your dataset**: Place video feature files in the specified `feat_cache_dir` path, ensuring the files are in `.pt` format.

2. **Run the following command**:

   ```bash
   python evqascore-demo.py --storage_path $storage_path --vid_base_path $vid_base_path --use_feat_cache True --feat_cache_dir $feat_cache_dir --cands_path $cands_path --key_cands_path $key_cands_path --video_ids_path $video_ids_path

### Arguments:
- `--storage_path`: Path to store the VATEX-EVAL dataset.
- `--vid_base_path`: Base path for VATEX-EVAL videos (optional if using cached video features).
- `--use_feat_cache`: Whether to use pre-prepared video features; default is `True`.
- `--feat_cache_dir`: Directory for cached video features.
- `--cands_path`: Path to the JSON file containing candidate sentences.
- `--key_cands_path`: Path to the JSON file containing key candidate phrases.
- `--video_ids_path`: Path to the JSON file with video IDs.
