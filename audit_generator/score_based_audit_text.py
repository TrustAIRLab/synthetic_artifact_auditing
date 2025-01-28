import os
import glob
import argparse
from tqdm import tqdm
from datasets import Dataset
import numpy as np
import torch
import json
import evaluate
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import copy

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained NLP model")
    parser.add_argument("--pretrained_model", type=str, default='bert',help="pretrained model",)
    parser.add_argument("--task", type=str, default='sentiment_analysis',help="sentiment_analysis",)
    parser.add_argument("--dataset", type=str, default='imdb', help="dataset name",)
    parser.add_argument("--num_samples", type=int, default=3000, help="number of samples to fine-tune models",)
    parser.add_argument("--num_shadow_models", type=int, default=2, help="number of shadow model",)
    parser.add_argument("--num_target_models", type=int, default=2, help="number of target model",)
    parser.add_argument("--num_queries", type=int, default=10, help="number of query for black-box auditing",)
    parser.add_argument("--seed", type=int, default=0, help="number of samples",)
    parser.add_argument("--shot", type=int, default=0, help="few-shot instruction",)
    parser.add_argument("--epoch", type=int, default=5, help="number of epochs",)
    parser.add_argument("--meta_epoch", type=int, default=100, help="number of epochs",)
    parser.add_argument("--syn_prop", type=float, help="number of samples",)
    parser.add_argument("--syn_prop_list", nargs='+', help="syn model with diff prop.",)
    parser.add_argument("--output_dir", type=str, default=None, help="output dir",)
    parser.add_argument("--audit_type", type=str, default='white', help="white/black-box",)
    parser.add_argument("--query_type", type=str, default='real', help="real/syn/mixed",)
    parser.add_argument("--llm", type=str, help="pretrained model",)
    parser.add_argument("--llm_list", nargs='+',  help="pretrained model",)
    parser.add_argument("--type", type=str, default='shadow',help="shadow or target",)
    parser.add_argument("--select_strategy", type=str, default='order',help="order or topk",)
    parser.add_argument("--temperature", type=float, default=0.5,help="temperature",)
    parser.add_argument('--use_wandb', default=False, action="store_true", help='whether to use wandb')
    parser.add_argument('--optimize_query', default=False, action="store_true", help='whether to optimize the query set')
    
    args = parser.parse_args()
   
    return args


class CustomTextClassificationPipeline(TextClassificationPipeline):
    def postprocess(self, model_outputs):
        # return F.softmax(model_outputs["logits"],dim=-1)
        return model_outputs["logits"]
       
       
class ScoreBasedAudit(object):
    def __init__(self, args, shadow_real_performance, shadow_syn_performance, target_real_performance, target_syn_performance):
        self.args = args
        self.query_type = args.query_type

        self.shadow_real_performance = shadow_real_performance
        self.shadow_syn_performance = shadow_syn_performance
        self.target_real_performance = target_real_performance
        self.target_syn_performance = target_syn_performance
        
    
    def _thre_setting(self, r_values, s_values):
        value_list = np.concatenate((r_values, s_values))
        thre, max_acc = 0, 0
        for value in value_list:
            r_ratio = np.sum(r_values>=value)/(len(r_values)+0.0)
            s_ratio = np.sum(s_values<value)/(len(s_values)+0.0)
            acc = 0.5*(r_ratio + s_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

        
    def _audit(self, metric, s_r_values, s_s_values, t_r_values, t_s_values):
        t_s, t_r = 0, 0
        print('Overall threshold')
        thre = self._thre_setting(s_r_values, s_s_values)
        print(f"Query type: {self.args.query_type}, #Queries: {self.args.num_queries}, #Shadow Models: {args.num_shadow_models}, #Target Models: {args.num_target_models}, Metric: {metric}, Threshold: {thre}")
        t_r += np.sum(t_r_values>=thre)
        t_s += np.sum(t_s_values<thre)
        audit_acc = 0.5*(t_s/(len(t_s_values)+0.0) + t_r/(len(t_r_values)+0.0))
        print(f'For auditing via {metric}, the acc is {audit_acc:.3f}')
        return round(audit_acc, 3)
    
    
    def benchmark(self):
        audit_res = {}
        metric_names = list(self.shadow_syn_performance.keys())
        for metric in metric_names:
            acc = self._audit(metric, self.shadow_real_performance[metric], self.shadow_syn_performance[metric], self.target_real_performance[metric], self.target_syn_performance[metric])
            audit_res[metric] = acc
        return audit_res

def softmax_by_row(logits, T = 1.0):
    mx = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp((logits - mx)/T)
    denominator = np.sum(exp, axis=-1, keepdims=True)
    return exp/denominator


class ModelEval(object):
    def __init__(self, args, num_models, metrics=['bertscore'], data_type='shadow', model_type='real'):
        self.args = copy.deepcopy(args)
        self.metrics = metrics
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_models = num_models
        self.data_type = data_type
        self.model_type = model_type
        self.model_eval_results = []  
    
    def convert(self, data):
        restructured_data = {}
        for entry in data:
            for key, value in entry.items():
                if key not in restructured_data:
                    restructured_data[key] = []
                restructured_data[key].append(value)  
        return restructured_data        
                  
    def compute_metric(self, decoded_preds, decoded_labels):
        results = {}
        
        if 'rouge' in self.metrics:
            rouge = evaluate.load("rouge")
            rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            results['rouge1'] = rouge_result['rouge1']
            results['rouge2'] = rouge_result['rouge2']
            results['rougeL'] = rouge_result['rougeL']
            results['rougeLsum'] = rouge_result['rougeLsum']
            
        if 'bertscore' in self.metrics:
            bertscore = evaluate.load("bertscore")
            bertscore_result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
            results["bertscore_precision"] = np.mean(bertscore_result["precision"])
            results["bertscore_recall"] = np.mean(bertscore_result["recall"])
            results["bertscore_f1"] = np.mean(bertscore_result["f1"])

        if 'bleu' in self.metrics:
            bleu = evaluate.load('bleu')
            bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
            results["bleu_score"] = bleu_result["bleu"]
            
        return results
        
    def evaluate(self, query_index):
        
        if self.model_type == 'real':
            for seed in tqdm(range(1, self.num_models+1)):
                eval_res_path = f'./results/models/{self.args.task}/{self.data_type}_{self.args.pretrained_model}_{self.args.dataset}_num{self.args.num_samples}_seed{seed}_epoch{args.epoch}/eval_results.json'
                if not os.path.exists(eval_res_path):
                    continue
                eval_res = json.load(open(eval_res_path))
                query_preds = np.array(eval_res['eval_preds'])[query_index]
                query_labels = np.array(eval_res['eval_labels'])[query_index]
                metric_values = self.compute_metric(query_preds, query_labels)
                self.model_eval_results.append(metric_values)
        elif self.model_type == 'syn' and args.syn_prop == 1.0:
            for seed in tqdm(range(1, self.num_models+1)):
                eval_res_path = f'./results/models/{self.args.task}/{self.data_type}_{self.args.pretrained_model}_{self.args.dataset}_num{self.args.num_samples}_seed{seed}_epoch{self.args.epoch}_{self.args.llm}_s{self.args.shot}_t{self.args.temperature}_p{self.args.syn_prop}/eval_results.json'
                if not os.path.exists(eval_res_path):
                    continue
                eval_res = json.load(open(eval_res_path))
                query_preds = np.array(eval_res['eval_preds'])[query_index]
                query_labels = np.array(eval_res['eval_labels'])[query_index]
                metric_values = self.compute_metric(query_preds, query_labels)
                self.model_eval_results.append(metric_values)
            print(eval_res_path)
        elif self.model_type == 'syn' and args.syn_prop_list:
            num_models_per_prop = int(self.num_models/len(args.syn_prop_list))
            for syn_prop in tqdm(args.syn_prop_list):
                for seed in range(1, int(num_models_per_prop)+1):
                    eval_res_path = f'./results/models/{self.args.task}/{self.data_type}_{self.args.pretrained_model}_{self.args.dataset}_num{self.args.num_samples}_seed{seed}_epoch{self.args.epoch}_{self.args.llm}_s{self.args.shot}_t{self.args.temperature}_p{syn_prop}/eval_results.json'
                    if not os.path.exists(eval_res_path):
                        continue
                    eval_res = json.load(open(eval_res_path))
                    query_preds = np.array(eval_res['eval_preds'])[query_index]
                    query_labels = np.array(eval_res['eval_labels'])[query_index]
                    metric_values = self.compute_metric(query_preds, query_labels)
                    self.model_eval_results.append(metric_values)
            print(eval_res_path)
        elif self.model_type == 'syn' and args.syn_prop == 'random' and args.llm_list:
            llm_str = '_'.join(args.llm_list)
            for seed in range(1, self.num_models+1):
                dir_path = list(glob.glob(f'./results/models/{self.args.task}/mix_sources/{self.data_type}_{self.args.pretrained_model}_{self.args.dataset}_num{self.args.num_samples}_seed{seed}_epoch{self.args.epoch}_{llm_str}_s{self.args.shot}_t{self.args.temperature}_p*'))[0] 
                eval_res_path = os.path.join(dir_path, 'eval_results.json')           
                if not os.path.exists(eval_res_path):
                    continue
                eval_res = json.load(open(eval_res_path))
                query_preds = np.array(eval_res['eval_preds'])[query_index]
                query_labels = np.array(eval_res['eval_labels'])[query_index]
                metric_values = self.compute_metric(query_preds, query_labels)
                self.model_eval_results.append(metric_values)   
            print(eval_res_path)    
        self.model_eval_results = self.convert(self.model_eval_results)
        return self.model_eval_results

def main(args):
    
    
    task_dataset_dict = {'text_summarization': 'cnn_dailymail', 'news_summarization': 'xsum'}
    args.dataset = task_dataset_dict[args.task]
    np.random.seed(args.seed)
    query_index = np.random.choice(1000, args.num_queries, replace=False)
    print(query_index)
    
    if args.syn_prop:
        syn_prop = args.syn_prop
    elif args.syn_prop_list:
        syn_prop = '_'.join(args.syn_prop_list)
    else:
        syn_prop = 'random'
        args.syn_prop = 'random'
    
    if args.llm:
        llm_str = args.llm
    elif args.llm_list:
        llm_str = '_'.join(args.llm_list)
        llm_str += f'_random'
    else:
        exit()
    
    
    target_real_model_evaluator = ModelEval(args, int(args.num_target_models/2), metrics=['rouge', 'bertscore', 'bleu'], data_type='target', model_type='real') 
    target_syn_model_evaluator = ModelEval(args, int(args.num_target_models/2), metrics=['rouge', 'bertscore', 'bleu'], data_type='target', model_type='syn')
    target_real_performance = target_real_model_evaluator.evaluate(query_index)
    target_syn_performance = target_syn_model_evaluator.evaluate(query_index)
    
    shadow_real_model_evaluator = ModelEval(args, int(args.num_shadow_models/2), metrics=['rouge', 'bertscore', 'bleu'], data_type='shadow', model_type='real')  
    shadow_syn_model_evaluator = ModelEval(args, int(args.num_shadow_models/2), metrics=['rouge', 'bertscore', 'bleu'], data_type='shadow', model_type='syn')
    shadow_real_performance =  shadow_real_model_evaluator.evaluate(query_index)
    shadow_syn_performance = shadow_syn_model_evaluator.evaluate(query_index) 
    
    audit = ScoreBasedAudit(args, shadow_real_performance, shadow_syn_performance, target_real_performance, target_syn_performance)
    audit_res = audit.benchmark()
    
    with open("results/logs/score_based_audit_generator.txt", 'a') as wf:
        wf.write(f"{args.task};{args.pretrained_model};{llm_str};{syn_prop};{args.num_shadow_models};{args.num_samples};{args.num_target_models};{args.query_type};{args.num_queries};{args.seed};{str(audit_res)}\n")  

if __name__ == "__main__":
    args = parse_args()
    main(args)