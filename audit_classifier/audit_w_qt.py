import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import evaluate
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import sys
import time
from transformers import  AutoModelForSequenceClassification
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auditing.meta_classifier import MLP_B, MLP_O
from auditing.basic_model_set import BasicModelSet


def _weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
        values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
    # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))
        # m.weight.data.normal_(0.0,1/2)
    # m.bias.data should be 0
        m.bias.data.fill_(0)
       
        
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained NLP model")
    parser.add_argument("--pretrained_model", type=str, default='bert',help="pretrained model",)
    parser.add_argument("--dataset", type=str, default='imdb', help="dataset name",)
    parser.add_argument("--num_samples", type=int, default=3000, help="number of samples to fine-tune models",)
    parser.add_argument("--num_shadow_models", type=int, default=200, help="number of shadow model",)
    parser.add_argument("--num_target_models", type=int, default=100, help="number of target model",)
    parser.add_argument("--num_queries", type=int, default=10, help="number of query for black-box auditing",)
    parser.add_argument("--seed", type=int, default=0, help="number of samples",)
    parser.add_argument("--shot", type=int, default=0, help="few-shot instruction",)
    parser.add_argument("--epoch", type=int, default=5, help="number of epochs",)
    parser.add_argument("--meta_epoch", type=int, default=10, help="number of epochs",)
    parser.add_argument("--meta_feature", type=str, default='posterior', help="meta input feature",)
    parser.add_argument("--task", type=str, default='sentiment_analysis', help="task",)
    parser.add_argument("--syn_prop", type=float, help="number of samples",)
    parser.add_argument("--syn_prop_list", nargs='+', help="syn model with diff prop.",)
    parser.add_argument("--output_dir", type=str, default=None, help="output dir",)
    parser.add_argument("--audit_type", type=str, default='white', help="white/black-box",)
    parser.add_argument("--query_type", type=str, default='syn', help="real/syn/mixed",)
    parser.add_argument("--llm", type=str,help="pretrained model",)
    parser.add_argument("--dist_type", type=str, choices=['uniform', 'random'])
    parser.add_argument("--llm_list", nargs='+',  help="pretrained model",)
    parser.add_argument("--type", type=str, default='shadow',help="shadow or target",)
    parser.add_argument("--temperature", type=float, default=0.5,help="temperature",)
    parser.add_argument('--use_wandb', default=False, action="store_true", help='whether to use wandb')
    parser.add_argument('--optimize_query', default=False, action="store_true", help='whether to optimize the query set')
    args = parser.parse_args()
   
    return args

 
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def entr_comp(pred):
    def _log_value(pred, small_value=1e-30):
        # Use torch.clamp to ensure the values are above a threshold and torch.log for logarithm
        return -torch.log(torch.clamp(pred, min=small_value))
    # Use torch.mul for element-wise multiplication and torch.sum to sum up, specifying the axis
    return torch.sum(torch.mul(pred, _log_value(pred)), dim=1)


def compute_metrics(preds, labels):
    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    res = metrics.compute(predictions=preds, references=labels)
    return res


def train(epoch, meta_feature, meta_model, optimizer, loss_fcn, train_model_set):
    
    y_true = []
    y_pred = []
    losses = AverageMeter('Loss', ':.4e')
    meta_model.train()
    device = torch.device('cuda')
    
    perm = np.random.permutation(len(train_model_set))
    for i in tqdm(perm):
        model_path, label = train_model_set[i]
        label = torch.tensor(label).type(torch.LongTensor).unsqueeze(0).to(device)   
        basic_model = AutoModelForSequenceClassification.from_pretrained(f'{model_path}/checkpoint/', output_hidden_states=True).to(device)
        basic_model.train()
        
        outputs = basic_model(inputs_embeds=meta_model.input)

        basic_model_posteriors = F.softmax(outputs['logits'], dim=-1)
        
        if meta_feature == 'entr':
            basic_model_entr = entr_comp(basic_model_posteriors).unsqueeze(0)
            meta_logits = meta_model(basic_model_entr)
        elif meta_feature == 'posterior':
            meta_logits = meta_model(basic_model_posteriors)
        
        meta_posteriors = F.softmax(meta_logits, dim=-1)
        loss = loss_fcn(meta_posteriors, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, preds = meta_posteriors.max(1)  
        y_true += label.cpu().tolist()
        y_pred += preds.cpu().tolist()
        losses.update(loss.item(), label.size(0))
        
    res = compute_metrics(y_pred, y_true)
    if args.use_wandb:
        wandb.log({
        'epoch': epoch,
        'training_loss': losses.avg,
        'training_acc': res['accuracy']
        })
        
    print(f'[Epoch {epoch}] Training results:', res)
    return res
        
@torch.no_grad()
def test(epoch, meta_feature, meta_model, loss_fcn, test_model_set):
    
    y_true = []
    y_pred = []
    meta_model.eval()
    losses = AverageMeter('Loss', ':.4e')
    device = torch.device('cuda')
    
    perm = np.random.permutation(len(test_model_set))
    for i in tqdm(perm):
        model_path, label = test_model_set[i]
        label = torch.tensor(label).type(torch.LongTensor).unsqueeze(0).to(device)  
        basic_model = AutoModelForSequenceClassification.from_pretrained(f'{model_path}/checkpoint/', output_hidden_states=True).to(device)
        basic_model.train()
        
        outputs = basic_model(inputs_embeds=meta_model.input)
        basic_model_posteriors = F.softmax(outputs['logits'], dim=-1)
           
        if meta_feature == 'entr':
            basic_model_entr = entr_comp(basic_model_posteriors).unsqueeze(0)
            meta_logits = meta_model(basic_model_entr)
        elif meta_feature == 'posterior':
            meta_logits = meta_model(basic_model_posteriors)
        
        meta_posteriors = F.softmax(meta_logits, dim=-1)     
        loss = loss_fcn(meta_posteriors, label)
        _, preds = meta_posteriors.max(1)  
        y_true += label.cpu().tolist()
        y_pred += preds.cpu().tolist()
        loss = loss_fcn(meta_posteriors, label)
        losses.update(loss.item(), label.size(0))
        
    res = compute_metrics(y_pred, y_true)
    if args.use_wandb:
        wandb.log({
        'epoch': epoch,
        'testing_loss': losses.avg,
        'testing_acc': res['accuracy']
        })
    print(f'[Epoch {epoch}] Testing results:', res)
    return res
    
def main(args):
    start_time = time.time()
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True   
    root = 'results/meta_models/qt/'
    
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
        llm_str += f'_{args.dist_type}'
    else:
        exit()
        
    if args.output_dir is None:
        output_path = f'{args.task}_{args.meta_feature}_bm{args.pretrained_model}_{llm_str}_{syn_prop}_ns{args.num_shadow_models}_nt{args.num_target_models}_nq{args.num_queries}_seed{args.seed}'
        args.output_dir = os.path.join(root, output_path)
        os.makedirs(args.output_dir, exist_ok=True)
    
    print(f'Output dir: {args.output_dir}')
    
    if args.use_wandb:
        wandb.login(key='xxx')
        wandb.init(project='Synthetic Artifacts Audit')
        wandb.config.update(args)
        wandb.run.name = f'audit_classifier_qt_' + output_path
        
    # task num_classes
    task_class_dict = {'sentiment_analysis': 2, 'spam_detection': 2, 'topic_classification': 4}
    task_dataset_dict = {'sentiment_analysis': 'imdb', 'spam_detection': 'enron_spam', 'topic_classification': 'ag_news'}
    args.dataset = task_dataset_dict[args.task]
    input_size = (args.num_queries, 10, 768)
    if args.meta_feature == 'entr':
        feature_size = args.num_queries
    elif args.meta_feature == 'posterior':
        class_num = task_class_dict[args.task]
        feature_size = args.num_queries*class_num
    
    train_model_set = BasicModelSet(args, args.num_shadow_models, data_type="shadow")
    test_model_set = BasicModelSet(args, args.num_target_models, data_type="target")
    meta_model = MLP_O(input_size, feature_size)
    # print(train_model_set[0])
 
    meta_model = meta_model.to(device)  
    meta_model.apply(_weights_init_normal)
    
    # print(meta_model.input.size())
    # exit()
    
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-3)
    
    if args.use_wandb:
        wandb.watch(meta_model, loss_fcn, log='all', log_freq=10)
    
    best_acc = 0.0
    for epoch in range(args.meta_epoch):
        train(epoch, args.meta_feature, meta_model, optimizer, loss_fcn, train_model_set)
        res = test(epoch, args.meta_feature, meta_model, loss_fcn, test_model_set)
        torch.save(meta_model.state_dict(), os.path.join(args.output_dir, 'checkpoint.pth'))
        
        if best_acc < res['accuracy']:
            best_res = res
            best_acc = res['accuracy']
            torch.save(meta_model.state_dict(), os.path.join(args.output_dir, 'best_checkpoint.pth'))
        
    end_time = time.time()
    elapsed_time = end_time - start_time 
    filename =  "results/logs/qt_result.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a') as wf:
        wf.write(f"{args.task},{args.pretrained_model},{llm_str},{syn_prop},{args.meta_feature},{args.meta_epoch},{args.num_shadow_models},{args.num_target_models},{args.num_queries},{args.seed},{round(best_res['accuracy'], 3)},{round(best_res['f1'], 3)},{round(best_res['precision'], 3)},{round(best_res['recall'], 3)}\n")        


if __name__ == "__main__":
    
    args = parse_args()
    main(args)


