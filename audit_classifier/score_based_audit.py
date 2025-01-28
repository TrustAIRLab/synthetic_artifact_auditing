import os
import glob
import argparse
from tqdm import tqdm
from datasets import Dataset
import numpy as np
import torch
import json
import sys
import copy
import time
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auditing.query import Query


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
    parser.add_argument("--query_type", type=str, default='syn', help="real/syn/mixed",)
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
       
       
class MetricBasedAudit(object):
    def __init__(self, args, shadow_real_performance, shadow_syn_performance, target_real_performance, target_syn_performance):
        self.args = args
        self.query_type = args.query_type
        self.shadow_num_classes = args.shadow_num_classes
        self.target_num_classes = args.target_num_classes
        
        self.shadow_real_performance = shadow_real_performance
        self.shadow_syn_performance = shadow_syn_performance
        self.target_real_performance = target_real_performance
        self.target_syn_performance = target_syn_performance
        
        self.s_r_acc, self.s_r_conf, self.s_r_entr, self.s_r_m_entr = self._calculate_metric(self.shadow_real_performance)
        self.s_s_acc, self.s_s_conf, self.s_s_entr, self.s_s_m_entr = self._calculate_metric(self.shadow_syn_performance)
        self.t_r_acc, self.t_r_conf, self.t_r_entr, self.t_r_m_entr = self._calculate_metric(self.target_real_performance)
        self.t_s_acc, self.t_s_conf, self.t_s_entr, self.t_s_m_entr = self._calculate_metric(self.target_syn_performance)


    def _calculate_metric(self, model_results, operation='avg'):
        acc_set = []
        conf_set = []
        entr_set = []
        m_entr_set = []
        for (outputs, labels) in model_results:
            corr = (np.argmax(outputs, axis=1) == labels).astype(int)[0]
            acc = sum(corr)/len(corr)
            
            labels_for_conf = labels.reshape(-1,1)
            conf = np.array([outputs[i, labels_for_conf[i]] for i in range(len(labels_for_conf))])
            entr = self._entr_comp(outputs)
            m_entr = self._m_entr_comp(outputs, labels)
            if operation == 'avg':
                acc_set.append(np.mean(acc))
                conf_set.append(np.mean(conf))
                entr_set.append(np.mean(entr))
                m_entr_set.append(np.mean(m_entr))
        print(np.array(acc_set))
        return np.array(acc_set), np.array(conf_set), np.array(entr_set), np.array(m_entr_set)
    
    
    def _thre_setting(self, r_values, s_values):
        if self.query_type == 'real':
            value_list = np.concatenate((r_values, s_values))
            thre, max_acc = 0, 0
            for value in value_list:
                r_ratio = np.sum(r_values>=value)/(len(r_values)+0.0)
                s_ratio = np.sum(s_values<value)/(len(s_values)+0.0)
                acc = 0.5*(r_ratio + s_ratio)
                if acc > max_acc:
                    thre, max_acc = value, acc
            return thre
        elif self.query_type.startswith('syn'):
            value_list = np.concatenate((r_values, s_values))
            thre, max_acc = 0, 0
            for value in value_list:
                r_ratio = np.sum(r_values<value)/(len(r_values)+0.0)
                s_ratio = np.sum(s_values>=value)/(len(s_values)+0.0)
                acc = 0.5*(r_ratio + s_ratio)
                if acc > max_acc:
                    thre, max_acc = value, acc
            return thre
       
        
    def _audit(self, metric, s_r_values, s_s_values, t_r_values, t_s_values):
        t_s, t_r = 0, 0
        print('Overall threshold')
        thre = self._thre_setting(s_r_values, s_s_values)
        print(f"Query type: {self.args.query_type}, #Queries: {self.args.num_queries}, #Shadow Models: {args.num_shadow_models}, #Target Models: {args.num_target_models}, Metric: {metric}, Threshold: {thre}")
        if self.query_type == 'real':
            t_r += np.sum(t_r_values>=thre)
            t_s += np.sum(t_s_values<thre)
        elif self.query_type.startswith('syn'):
            t_r += np.sum(t_r_values<thre)
            t_s += np.sum(t_s_values>=thre)
        audit_acc = 0.5*(t_s/(len(self.target_syn_performance)+0.0) + t_r/(len(self.target_real_performance)+0.0))
        print(f'For auditing via {metric}, the acc is {audit_acc:.3f}')
        return round(audit_acc, 3)
        
    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))
    
    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)),axis=1)
    
    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1-probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs),axis=1)
    
    def benchmark(self):
        acc_acc = self._audit('acc', self.s_r_acc, self.s_s_acc, self.t_r_acc, self.t_s_acc)
        conf_acc = self._audit('confidence', self.s_r_conf, self.s_s_conf, self.t_r_conf, self.t_s_conf)
        entr_acc = self._audit('entropy', -self.s_r_entr, -self.s_s_entr, -self.t_r_entr, -self.t_s_entr)
        m_entr_acc = self._audit('m_entropy', -self.s_r_m_entr, -self.s_s_m_entr, -self.t_r_m_entr, -self.t_s_m_entr)
        return acc_acc, conf_acc, entr_acc, m_entr_acc

def softmax_by_row(logits, T = 1.0):
    mx = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp((logits - mx)/T)
    denominator = np.sum(exp, axis=-1, keepdims=True)
    return exp/denominator


class ModelEval(object):
    def __init__(self, args, num_models, select_strategy, data_type='shadow', model_type='real'):
        self.args = copy.deepcopy(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_models = num_models
        self.data_type = data_type
        self.model_type = model_type
        self.model_set = []
        
        if select_strategy == 'order':
            self._select_in_order()
        elif select_strategy == 'topk':
            self._select_topk()
            
    def _select_in_order(self):
        model_acc = []
        if self.model_type == 'real':
            for seed in range(1, self.num_models+1):
                model_path = f'./results/models/{self.args.task}/{self.data_type}_{self.args.pretrained_model}_{self.args.dataset}_num{self.args.num_samples}_seed{seed}_epoch{self.args.epoch}'
                # print(model_path)
                json_path = os.path.join(model_path, 'eval_results.json')
                if not os.path.exists(json_path):
                    continue
                eval_acc = json.load(open(json_path))['eval_accuracy']
                model_acc.append(eval_acc)
                self.model_set.append(os.path.join(model_path, 'checkpoint'))
            avg_selected_model_acc = sum(model_acc)/len(model_acc)
            print(f"Average acc of the {len(self.model_set)} selected {self.data_type} {self.model_type} model is {avg_selected_model_acc:.3f}") 
        elif self.model_type == 'syn' and args.syn_prop == 1.0:
            for seed in range(1, self.num_models+1):
                model_path = f'./results/models/{self.args.task}/{self.data_type}_{self.args.pretrained_model}_{self.args.dataset}_num{self.args.num_samples}_seed{seed}_epoch{self.args.epoch}_{self.args.llm}_s{self.args.shot}_t{self.args.temperature}_p{self.args.syn_prop}'
                json_path = os.path.join(model_path, 'eval_results.json')
                if not os.path.exists(json_path):
                    continue
                eval_acc = json.load(open(json_path))['eval_accuracy']
                model_acc.append(eval_acc)
                self.model_set.append(os.path.join(model_path, 'checkpoint'))
            avg_selected_model_acc = sum(model_acc)/len(model_acc)
            print(f"Average acc of the {len(self.model_set)} selected {self.data_type} {self.model_type} model is {avg_selected_model_acc:.3f}")  
            print(self.model_set[-1])
        elif self.model_type == 'syn' and args.syn_prop == 'random' and args.llm_list:
            llm_str = '_'.join(args.llm_list)
            for seed in range(1, self.num_models+1):
                model_path = list(glob.glob(f'./results/models/{self.args.task}/mix_sources/{self.data_type}_{self.args.pretrained_model}_{self.args.dataset}_num{self.args.num_samples}_seed{seed}_epoch{self.args.epoch}_random_{llm_str}_s{self.args.shot}_t{self.args.temperature}_p*'))[0]
                json_path = os.path.join(model_path, 'eval_results.json')
                if not os.path.exists(json_path):
                    continue
                eval_acc = json.load(open(json_path))['eval_accuracy']
                model_acc.append(eval_acc)
                self.model_set.append(os.path.join(model_path, 'checkpoint'))
            avg_selected_model_acc = sum(model_acc)/len(model_acc)
            print(f"Average acc of the {len(self.model_set)} selected {self.data_type} {self.model_type} model is {avg_selected_model_acc:.3f}")  
            print(self.model_set[-1])    
        elif self.model_type == 'syn' and args.syn_prop_list:
            num_models_per_prop = int(self.num_models/len(args.syn_prop_list)) 
            print(num_models_per_prop)
            for syn_prop in args.syn_prop_list:
                syn_model_list = []
                syn_model_acc = []
                for seed in range(1, int(num_models_per_prop)+1):
                    model_path = f'./results/models/{args.task}/{self.data_type}_{args.pretrained_model}_{args.dataset}_num{args.num_samples}_seed{seed}_epoch{args.epoch}_{args.llm}_s{args.shot}_t{args.temperature}_p{syn_prop}'
                    json_path = os.path.join(model_path, 'eval_results.json')
                    if not os.path.exists(json_path):
                        continue
                    eval_acc = json.load(open(json_path))['eval_accuracy']
                    syn_model_acc.append(eval_acc)
                    syn_model_list.append(os.path.join(model_path, 'checkpoint'))
                avg_selected_model_acc = sum(syn_model_acc)/len(syn_model_acc)
                print(f"Average acc of the {len(syn_model_list)} selected {self.data_type} {self.model_type} model with syn prop {syn_prop} is {avg_selected_model_acc:.3f}")  
                self.model_set.extend(syn_model_list)
            print(self.model_set[-1])
             
    def _select_topk(self):
        model_acc = []
        if self.model_type == 'real':
            model_candidates = list(glob.glob(f'./results/models/{self.args.task}/{self.data_type}_{self.args.pretrained_model}_{self.args.dataset}_num{self.args.num_samples}_seed*_epoch{self.args.epoch}'))
            for model_candidate in model_candidates:
                json_path = os.path.join(model_candidate, 'eval_results.json')
                if not os.path.exists(json_path):
                    continue
                eval_acc = json.load(open(json_path))['eval_accuracy']
                model_acc.append((os.path.join(model_candidate, 'checkpoint'), eval_acc))
            final_model = sorted(model_acc, key=lambda x:x[1], reverse=True)[:self.num_models]
            self.model_set = [model_path for model_path, _ in final_model]
            avg_selected_model_acc = sum([acc for _, acc in final_model]) /self.num_models
            print(f"Average acc of the {self.num_models} selected {self.data_type} {self.model_type} model is {avg_selected_model_acc:.3f}")
        elif self.model_type == 'syn' and args.syn_prop == 1.0:
            model_candidates = list(glob.glob(f'./results/models/{self.args.task}/{self.data_type}_{self.args.pretrained_model}_{self.args.dataset}_num{self.args.num_samples}_seed*_epoch{self.args.epoch}_{self.args.llm}_s{self.args.shot}_t{self.args.temperature}_p{self.args.syn_prop}'))
            for model_candidate in model_candidates:
                json_path = os.path.join(model_candidate, 'eval_results.json')
                if not os.path.exists(json_path):
                    continue
                eval_acc = json.load(open(json_path))['eval_accuracy']
                model_acc.append((os.path.join(model_candidate, 'checkpoint'), eval_acc))
            final_model = sorted(model_acc, key=lambda x:x[1], reverse=True)[:self.num_models]
            self.model_set = [model_path for model_path, _ in final_model]
            avg_selected_model_acc = sum([acc for _, acc in final_model]) /self.num_models
            print(f"Average acc of the {self.num_models} selected {self.data_type} {self.model_type} model is {avg_selected_model_acc:.3f}")
            print(self.model_set[-1])
        elif self.model_type == 'syn' and args.syn_prop == 'random' and args.llm_list:
            llm_str = '_'.join(args.llm_list)
            model_candidates = list(glob.glob(f'./results/models/{self.args.task}/mix_sources/{self.data_type}_{self.args.pretrained_model}_{self.args.dataset}_num{self.args.num_samples}_seed*_epoch{self.args.epoch}_random_{llm_str}_s{self.args.shot}_t{self.args.temperature}_p*'))
            for model_candidate in model_candidates:
                json_path = os.path.join(model_candidate, 'eval_results.json')
                if not os.path.exists(json_path):
                    continue
                eval_acc = json.load(open(json_path))['eval_accuracy']
                model_acc.append((os.path.join(model_candidate, 'checkpoint'), eval_acc))
            final_model = sorted(model_acc, key=lambda x:x[1], reverse=True)[:self.num_models]
            self.model_set = [model_path for model_path, _ in final_model]
            avg_selected_model_acc = sum([acc for _, acc in final_model]) /self.num_models
            print(f"Average acc of the {self.num_models} selected {self.data_type} {self.model_type} model is {avg_selected_model_acc:.3f}")
            print(self.model_set[-1])    
        elif self.model_type == 'syn' and args.syn_prop_list:
            num_models_per_prop = int(self.num_models/len(args.syn_prop_list)) 
            
            for syn_prop in args.syn_prop_list:
                syn_model_acc = []
                model_candidates = list(glob.glob(f'./results/models/{args.task}/{self.data_type}_{args.pretrained_model}_{args.dataset}_num{args.num_samples}_seed*_epoch{args.epoch}_{args.llm}_s{args.shot}_t{args.temperature}_p{syn_prop}'))
                for model_candidate in model_candidates:
                    json_path = os.path.join(model_candidate, 'eval_results.json')
                    if not os.path.exists(json_path):
                        continue
                    eval_acc = json.load(open(json_path))['eval_accuracy']
                    syn_model_acc.append((os.path.join(model_candidate, 'checkpoint'), eval_acc))
                    final_model = sorted(syn_model_acc, key=lambda x:x[1], reverse=True)[:num_models_per_prop]
                    self.model_set.extend([model_path for model_path, _ in final_model])
                    avg_selected_model_acc = sum([acc for _, acc in final_model]) /num_models_per_prop
                print(f"Average acc of the {num_models_per_prop} selected {self.data_type} {self.model_type} model with syn prop {syn_prop} is {avg_selected_model_acc:.3f}")
            print(self.model_set[-1])
            
    
    @torch.no_grad()     
    def performance(self, query_set):
        results = []
        for _, model_path in tqdm(enumerate(self.model_set)):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            pipe = CustomTextClassificationPipeline(model=model, tokenizer=tokenizer, device=self.device)
            def get_predictions(example):
                with torch.no_grad():
                    logits = pipe(example['text'], padding=True, truncation=True, batch_size=256)
                outputs = softmax_by_row(torch.cat(logits, 0).numpy())
                return {"outputs": outputs}
            predictions = query_set.map(get_predictions, batched=True, batch_size=256)
            outputs = np.array([pred for pred in predictions["outputs"]])
            labels = np.array([ref for ref in query_set["label"]]).reshape(1, -1)
            results.append((outputs, labels))
        return results


def main(args):
    
    start_time = time.time()
    task_class_dict = {'sentiment_analysis': 2, 'spam_detection': 2, 'topic_classification': 4}
    task_dataset_dict = {'sentiment_analysis': 'imdb', 'spam_detection': 'enron_spam', 'topic_classification': 'ag_news'}
    args.dataset = task_dataset_dict[args.task]
    args.shadow_num_classes = task_class_dict[args.task]
    args.target_num_classes = task_class_dict[args.task]
    args.num_classes = task_class_dict[args.task]
    
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
    
    query = Query(args)
    query_texts, query_labels = query.select_queries(args.num_queries, args.query_type)
    query_set = Dataset.from_dict({'text': query_texts, 'label': query_labels})

    target_real_model_evaluator = ModelEval(args, int(args.num_target_models/2), select_strategy='order', data_type='target', model_type='real') 
    target_syn_model_evaluator = ModelEval(args, int(args.num_target_models/2), select_strategy='order', data_type='target', model_type='syn')
    # args.num_samples = 3000
    # args.epoch = 1
    shadow_real_model_evaluator = ModelEval(args, int(args.num_shadow_models/2), select_strategy='order', data_type='shadow', model_type='real')  
    shadow_syn_model_evaluator = ModelEval(args, int(args.num_shadow_models/2), select_strategy='order', data_type='shadow', model_type='syn')
    
    target_real_performance = target_real_model_evaluator.performance(query_set)
    target_syn_performance = target_syn_model_evaluator.performance(query_set)
    shadow_real_performance =  shadow_real_model_evaluator.performance(query_set)
    shadow_syn_performance = shadow_syn_model_evaluator.performance(query_set)
    

    audit = MetricBasedAudit(args, shadow_real_performance, shadow_syn_performance, target_real_performance, target_syn_performance)
    acc_acc, conf_acc, entr_acc, m_entr_acc = audit.benchmark()
    
    filename = "results/logs/score_based_audit_classifier.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    end_time = time.time()
    elapsed_time = end_time - start_time 
    
    with open(filename, 'a') as wf:
        wf.write(f"{args.task},{args.pretrained_model},{llm_str},{syn_prop},{args.num_shadow_models},{args.num_target_models},{args.query_type},{args.num_queries},{args.seed},{acc_acc},{conf_acc},{entr_acc},{m_entr_acc}\n")  

if __name__ == "__main__":
    args = parse_args()
    main(args)