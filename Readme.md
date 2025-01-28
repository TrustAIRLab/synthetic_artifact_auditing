


## Overview
This is the repository of the dataset and code for the paper "Synthetic Artifact Auditing: Tracing LLM-Generated Synthetic Data Usage in Downstream Applications" accepted at Usenix Security 2025.

Our artifact includes:
- Synthetic Dataset
- Training/Generation Scripts for Classifiers/Generators/Statistical Plots
- Auditing Scripts for Classifiers/Generators/Statistical Plots
- Synthetic Data Generation Scripts

## 1. Setup

### Environment

```
python >= 3.9
pip install -r requirements.txt
```

### Dataset

1. Real data will be directly downloaded from Hugging Face once the training scripts are executed.
2. Synthetic Data: `results/generated_data` If you want to generate synthetic data, please refer to the ` Synthetic Data Generation` section.


## 2. Classifiers/Generators/Statistical Plots Training/Generation

a. training classifiers

- $T_{C_{1}}$ sentiment_analysis.py
- $T_{C_{2}}$ topic_classification.py
- $T_{C_{3}}$ spam_detection.py

```
# real data only
CUDA_VISIBLE_DEVICES=0 torchrun tasks/sentiment_analysis.py --type target --epoch 5  --num_samples 3000 --syn_prop 0 --pretrained_model distillbert --seed 0

CUDA_VISIBLE_DEVICES=0 torchrun tasks/sentiment_analysis.py --type shadow --epoch 5  --num_samples 3000 --syn_prop 0 --pretrained_model distillbert --seed 0

# s1 syn_prop = 1.0
CUDA_VISIBLE_DEVICES=0 torchrun tasks/sentiment_analysis.py --type target --epoch 5 --syn_prop 1 --num_samples 3000 --pretrained_model distillbert --shot 0 --llm gpt3.5 --temperature 0.5 --seed 0

CUDA_VISIBLE_DEVICES=0 torchrun tasks/sentiment_analysis.py --type shadow --epoch 5 --syn_prop 1 --num_samples 3000 --pretrained_model distillbert --shot 0 --llm gpt3.5 --temperature 0.5 --seed 0

# s2 syn_prop = 0.x
CUDA_VISIBLE_DEVICES=0 torchrun tasks/sentiment_analysis.py --type target --epoch 5 --syn_prop 0.2 --num_samples 3000 --pretrained_model distillbert --shot 0 --llm gpt3.5 --temperature 0.5 --seed 0

CUDA_VISIBLE_DEVICES=0 torchrun tasks/sentiment_analysis.py --type shadow --epoch 5 --syn_prop 0.2 --num_samples 3000 --pretrained_model distillbert --shot 0 --llm gpt3.5 --temperature 0.5 --seed 0

# s3 syn_prop random
CUDA_VISIBLE_DEVICES=0 torchrun tasks/sentiment_analysis.py --type target --epoch 5 --llm_list gpt3.5 mistral chatglm gpt4 --dist_type random --num_samples 3000 --pretrained_model distillbert --shot 0 --temperature 0.5 --seed 0

CUDA_VISIBLE_DEVICES=0 torchrun tasks/sentiment_analysis.py --type shadow --epoch 5 --llm_list gpt3.5 mistral chatglm gpt4 --dist_type random --num_samples 3000 --pretrained_model distillbert --shot 0 --temperature 0.5 --seed 0
```

b. training generators

- $T_{G_{1}}$ text_summarization.py
- $T_{G_{2}}$ news_summarization.py 

```
# real data only
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 22902 tasks/text_summarization.py --dataset cnn_dailymail --num_samples 5000 --epoch 3 --syn_prop 0 --pretrained_model bart --type target --seed 0

CUDA_VISIBLE_DEVICES=0 torchrun --master_port 22902 tasks/text_summarization.py --dataset cnn_dailymail --num_samples 5000 --epoch 3 --syn_prop 0 --pretrained_model bart --type shadow --seed 0

# s1 syn_prop = 1.0
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 22902 tasks/text_summarization.py --dataset cnn_dailymail --num_samples 5000 --epoch 3 --syn_prop 1 --pretrained_model bart --type target --seed 0 --llm gpt3.5 --temperature 1.0 --shot 0

CUDA_VISIBLE_DEVICES=0 torchrun --master_port 22902 tasks/text_summarization.py --dataset cnn_dailymail --num_samples 5000 --epoch 3 --syn_prop 1 --pretrained_model bart --type shadow --seed 0 --llm gpt3.5 --temperature 1.0 --shot 0

# s2 syn_prop = 0.x
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 22902 tasks/text_summarization.py --dataset cnn_dailymail --num_samples 5000 --epoch 3 --syn_prop 0.5 --pretrained_model bart --type target --seed 0 --llm gpt3.5 --temperature 1.0 --shot 0

# s3 syn_prop = random multiple sources
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 22902 tasks/text_summarization.py --dataset cnn_dailymail --num_samples 5000 --epoch 3 --pretrained_model bart --type target --seed 0  --temperature 1.0 --shot 0 --llm_list gpt3.5 mistral chatglm gpt4 --dist_type random
```

c. generate plots

- $T_{P_{1}}$ --task sentiment_analysis
- $T_{P_{2}}$ --task topic_classification

```
# real data only
python tasks/tsne_plots.py --type target --num_samples 1000 --embedding word2vec --seed 0 --task sentiment_analysis

# s1 syn_prop = 1.0
python tasks/tsne_plots.py --type target --syn_prop 1.0 --num_samples 1000 --embedding word2vec --seed 0 --task sentiment_analysis --shot 0 --llm gpt3.5 --temperature 0.5

# s2 syn_prop = 0.x
python tasks/tsne_plots.py --type target --syn_prop 0.2 --num_samples 1000 --embedding word2vec --seed 0 --task sentiment_analysis --shot 0 --llm gpt3.5 --temperature 0.5

# s3 syn_prop = random multiple sources
python tasks/tsne_plots.py --type target --llm_list gpt3.5 mistral chatglm gpt4 --dist_type random --num_samples 1000 --embedding word2vec --seed 0 --task sentiment_analysis --shot 0 --temperature 0.5
```
## 3. Classifiers Auditing


```
1. score-based audit
CUDA_VISIBLE_DEVICES=0 python audit_classifier/score_based_audit.py  --task sentiment_analysis --llm gpt3.5 --temperature 0.5 --shot 0 --num_samples 3000 --pretrained_model distillbert --epoch 5 --num_queries 200 --num_shadow_models 20 --num_target_models 100 --query_type syn --seed 0 --syn_prop 1.0

CUDA_VISIBLE_DEVICES=0 python audit_classifier/score_based_audit.py  --task sentiment_analysis --llm gpt3.5 --temperature 0.5 --shot 0 --num_samples 3000 --pretrained_model distillbert --epoch 5 --num_queries 200 --num_shadow_models 20 --num_target_models 100 --query_type syn --seed 0 --syn_prop_list '0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0'

CUDA_VISIBLE_DEVICES=0 python audit_classifier/score_based_audit_num.py  --task sentiment_analysis --llm_list gpt3.5 mistral chatglm gpt4 --temperature 0.5 --shot 0 --num_samples 3000 --epoch 5 --num_queries 200 --pretrained_model distillbert --num_shadow_models 20 --num_target_models 100 --query_type syn_multi_source --seed 0

2. tuning-based audit
CUDA_VISIBLE_DEVICES=0 python audit_classifier/audit_w_qt.py --meta_feature posterior --task sentiment_analysis --llm gpt3.5 --pretrained_model distillbert --temperature 0.5 --shot 0 --num_samples 3000 --epoch 5 --num_queries 200 --num_shadow_models 20 --meta_epoch 30 --num_target_models 100 --seed 0 --syn_prop 1.0

CUDA_VISIBLE_DEVICES=0 python audit_classifier/audit_w_qt.py --meta_feature posterior --task sentiment_analysis --llm gpt3.5 --pretrained_model distillbert --temperature 0.5 --shot 0 --num_samples 3000 --epoch 5 --num_queries 200 --num_shadow_models 20 --meta_epoch 30 --num_target_models 100 --seed 0 --syn_prop_list '0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0'

CUDA_VISIBLE_DEVICES=0 python audit_classifier/audit_w_qt.py --meta_feature posterior --task sentiment_analysis --pretrained_model distillbert --temperature 0.5 --shot 0 --num_samples 3000 --epoch 5 --num_queries 200 --num_shadow_models 20 --meta_epoch 30 --num_target_models 100 --seed 0 --llm_list gpt3.5 mistral chatglm gpt4 --dist_type random
```
## 4. Generators Auditing

```
CUDA_VISIBLE_DEVICES=0 python audit_classifier/score_based_audit.py  --task sentiment_analysis --llm gpt3.5 --temperature 1.0 --shot 0 --num_samples 3000 --pretrained_model bart --epoch 3 --num_queries 200 --num_shadow_models 20 --num_target_models 100 --query_type real --seed 0 --syn_prop 1.0

CUDA_VISIBLE_DEVICES=0 python audit_classifier/score_based_audit.py  --task sentiment_analysis --llm gpt3.5 --temperature 1.0 --shot 0 --num_samples 3000 --pretrained_model bart --epoch 3 --num_queries 200 --num_shadow_models 20 --num_target_models 100 --query_type real --seed 0 --syn_prop 1.0

CUDA_VISIBLE_DEVICES=0 python audit_classifier/score_based_audit.py  --task sentiment_analysis --temperature 1.0 --shot 0 --num_samples 3000 --pretrained_model bart --epoch 3 --num_queries 200 --num_shadow_models 20 --num_target_models 100 --query_type real --seed 0 --llm_list gpt3.5 mistral chatglm gpt4 --dist_type random
```

## 5. Statistical Plots Auditing

```
CUDA_VISIBLE_DEVICES=0 python audit_plot/audit_plot.py --task sentiment_analysis --llm gpt3.5 --shot 0 --temperature 0.5 --num_samples 1000 --embedding word2vec --num_shadow_plots 300 --num_target_plots 1000 --seed 0 --syn_prop 1.0

CUDA_VISIBLE_DEVICES=0 python audit_plot/audit_plot.py --task sentiment_analysis --llm gpt3.5 --shot 0 --temperature 0.5 --num_samples 1000 --embedding word2vec --num_shadow_plots 300 --num_target_plots 1000 --seed 0 --syn_prop '0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0'

CUDA_VISIBLE_DEVICES=0 python audit_plot/audit_plot.py --task sentiment_analysis --shot 0 --temperature 0.5 --num_samples 1000 --embedding word2vec --num_shadow_plots 300 --num_target_plots 1000 --seed 0 --llm_list gpt3.5 mistral chatglm gpt4 --dist_type random
```

## Synthetic Data Generation

```
# process
python synthetic/process_data.py

# spam detection label=[0,1]
python synthetic/generate_spam.py --llm gpt3.5 --temperature 1.0 --shot 0 --num_samples 3000 --label 0 --type target

# ag_news label=[0,1,2,3]
python synthetic/generate_ag_news.py --llm gpt3.5 --temperature 1.0 --shot 0 --num_samples 3000 --label 0 --type target

# imdb review label=['negative','positive']
python synthetic/generate_imdb_review.py --llm gpt3.5 --temperature 0.5 --shot 0 --num_samples 5000 --type target --label negative

python synthetic/generate_cnn_dailymail.py --llm gpt3.5 --temperature 1 --shot 0 --num_samples 6000 --type target

python synthetic/generate_xsum.py --llm gpt3.5 --temperature 1 --shot 0 --num_samples 6000 --type target
```

## Citation
If you find this code to be useful for your research, please consider citing.
```
@inproceedings{WYSBZ25,
author = {Yixin Wu and Ziqing Yang and Yun Shen and Michael Backes and Yang Zhang},
title = {{Synthetic Artifact Auditing: Tracing LLM-Generated Synthetic Data Usage in Downstream Applications}},
booktitle = {{USENIX Security Symposium (USENIX Security)}},
publisher = {USENIX},
year = {2025}
}
```