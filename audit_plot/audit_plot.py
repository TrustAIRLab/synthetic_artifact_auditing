import os
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import sys
import wandb
import evaluate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import PlotDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Auditing t-SNE plots")
    parser.add_argument("--dataset", type=str, default='imdb', help="dataset name",)
    parser.add_argument("--num_samples", type=int, default=3000, help="number of samples to draw the plot",)
    parser.add_argument("--num_shadow_plots", type=int, default=20, help="number of shadow plot",)
    parser.add_argument("--num_target_plots", type=int, default=20, help="number of target model",)
    parser.add_argument("--seed", type=int, default=0, help="number of samples",)
    parser.add_argument("--shot", type=int, default=0, help="few-shot instruction",)
    parser.add_argument("--epoch", type=int, default=50, help="number of epochs",)
    parser.add_argument("--task", type=str, default='sentiment_analysis', help="task",)
    parser.add_argument("--syn_prop", type=float, help="number of samples",)
    parser.add_argument("--syn_prop_list", nargs='+', help="syn model with diff prop.",)
    parser.add_argument("--output_dir", type=str, default=None, help="output dir",)
    parser.add_argument("--llm", type=str,help="pretrained model",)
    parser.add_argument("--llm_list", nargs='+',  help="pretrained model",)
    parser.add_argument("--type", type=str, default='shadow',help="shadow or target",)
    parser.add_argument("--temperature", type=float, default=0.5,help="temperature",)
    parser.add_argument("--embedding", type=str, default='tfidf',help="shadow or target",)
    parser.add_argument('--use_wandb', default=False, action="store_true", help='whether to use wandb')
    args = parser.parse_args()
   
    return args

def adapt_resnet_to_grayscale(model):
    # Modify the first convolutional layer to accept 1-channel input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

def compute_metrics(preds, labels):
    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    res = metrics.compute(predictions=preds, references=labels)
    return res

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


def train(args, epoch, model, optimizer, loss_fcn, train_loader):
    y_true = []
    y_pred = []
    losses = AverageMeter('Loss', ':.4e')
    model.train()
    device = torch.device('cuda')
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        
        outputs = model(inputs)
        posteriors = F.softmax(outputs, dim=-1)
        loss = loss_fcn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, preds = posteriors.max(1)  
        y_true += labels.cpu().tolist()
        y_pred += preds.cpu().tolist()
        losses.update(loss.item(), labels.size(0))

    res = compute_metrics(y_pred, y_true)
    if args.use_wandb:
        wandb.log({
        'epoch': epoch,
        'training_loss': losses.avg,
        'training_acc': res['accuracy']
        })
        
    print(f'[Epoch {epoch}] Training results:', res)
    
@torch.no_grad()   
def test(args, epoch, model, loss_fcn, test_loader):
    y_true = []
    y_pred = []
    model.eval()
    losses = AverageMeter('Loss', ':.4e')
    device = torch.device('cuda')

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
    
        outputs = model(inputs)
        posteriors = F.softmax(outputs, dim=-1)
        loss = loss_fcn(outputs, labels)
        
        _, preds = posteriors.max(1)  
        y_true += labels.cpu().tolist()
        y_pred += preds.cpu().tolist()
        losses.update(loss.item(), labels.size(0))
        
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True   
    root = 'results/plot_classifier/'
    
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
        
    if args.output_dir is None:
        output_path = f'{args.task}_{llm_str}_{syn_prop}_ns{args.num_shadow_plots}_nt{args.num_target_plots}_seed{args.seed}'
        args.output_dir = os.path.join(root, output_path)
        os.makedirs(args.output_dir, exist_ok=True)
    
    print(f'Output dir: {args.output_dir}')
    
    if args.use_wandb:
        wandb.login(key='xxx')
        wandb.init(project='Synthetic Artifacts Audit')
        wandb.config.update(args)
        wandb.run.name = f'audit_plots' + output_path
        
    # task num_classes
    task_dataset_dict = {'sentiment_analysis': 'imdb', 'spam_detection': 'enron_spam', 'topic_classification': 'ag_news'}
    args.dataset = task_dataset_dict[args.task]   
    
    model = models.resnet18(pretrained=True)
    model = adapt_resnet_to_grayscale(model)
    

    num_classes = 2  
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to('cuda')
    
    train_set = PlotDataset(args, args.num_shadow_plots, 'shadow')
    test_set = PlotDataset(args, args.num_target_plots, 'target')
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)
    
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0.0
    for epoch in range(args.epoch):
        train(args, epoch, model, optimizer, loss_fcn, train_loader)
        res = test(args, epoch, model, loss_fcn, test_loader)
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoint.pth'))
        
        if best_acc < res['accuracy']:
            best_res = res
            best_acc = res['accuracy']
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_checkpoint.pth'))
    filename = "results/logs/audit_plot_result.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a') as wf:
        wf.write(f"{args.task},{llm_str},{syn_prop},{args.embedding},{args.num_shadow_plots},{args.num_target_plots},{args.seed},{round(best_res['accuracy'], 3)},{round(best_res['f1'], 3)},{round(best_res['precision'], 3)},{round(best_res['recall'], 3)}\n")     


if __name__ == "__main__":
    args = parse_args()
    main(args)