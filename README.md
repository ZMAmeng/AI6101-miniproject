# AI6101-miniproject

### This repository contains a **minimal, fully-reproducible pipeline** for fine-tuning [AllenAI’s Longformer-base-4096] on a résumé-to-JD matching task.


The codebase is deliberately small and split into three logical modules:

| File                           | Purpose                                                          |
|--------------------------------|------------------------------------------------------------------|
| `model.py`                     | Model, tokenizer, and text-preprocessing utilities (`scrub`, `tokenise_batch`, `get_model`). |
| `training.py`                  | End-to-end training script: builds positive/negative pairs from a local CSV, tokenises, trains with 🤗 `Trainer`, and saves checkpoints. |
| `interface.py`                 | Lightweight inference layer: loads saved checkpoints and exposes `rank_resumes()` as a Python function **and** a simple CLI. |
| `Dataset\desenstive_resume.py` | Wash away personal information from various dimensions to prevent privacy leaks. |
---

### 0. Requirements

Python ≥ 3.9

```bash
pip install "transformers>=4.41" datasets evaluate pandas scikit-learn numpy
```
### 1. Prepare the dataset
Download the dataset from the website： https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset  and put it in the Dataset folder

By running the processing script in the Dataset folder, you can remove personal information from the database to protect privacy.
```bash
cd Dataset
python3 desenstive_resume.py
cd ..
```

### 2.Training


#### Note：Since this experiment needs to process a large amount of resume text, running under this parameter requires about 38GB of CUDA memory. If this memory is reached, please change MAX_LEN,ATTN_WINDOW, and batch size parameters
```bash
python train.py
```
Checkpoints, logs, and best metrics (.json) will appear under
./checkpoints


### 3.Inference
The interface.py script provides a simple inference interface to:  
Load the fine-tuned Longformer model from saved checkpoints.  
Rank multiple résumés against a given Job Description (JD) based on their predicted matching scores.
```bash
python interface.py \
  --jd "Python Developer" \
  --cv cv1.txt cv2.txt cv3.txt \
  -k 5
```


### 📜 License
This repo is distributed under the MIT License.  
 Original résumé data is © its respective creators (see Kaggle page).  
 Longformer © Allen Institute for AI, Apache-2.0.

### 🙏 Citation
If you find this template useful, please consider starring 🌟 the repo and citing:  

@misc{resume_longformer_2025,  
  author  = {Liu Zhimeng & },  
  title   = {Fine-tuning Longformer for Résumé–JD Matching},  
  year    = 2025,  
  url     = {https://github.com/ZMAmeng/AI6101-miniproject}
}
