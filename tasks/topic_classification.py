import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric, concatenate_datasets, ClassLabel, Features, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
   parser = argparse.ArgumentParser(description="Fine-tune pre-trained NLP model")
   parser.add_argument("--pretrained_model", type=str, default='bert',help="pretrained model",)
   parser.add_argument("--dataset", type=str, default='ag_news', help="dataset name",)
   parser.add_argument("--num_samples", type=int, default=8000, help="number of samples",)
   parser.add_argument("--seed", type=int, default=0, help="random seed",)
   parser.add_argument("--shot", type=int, default=0, help="few-shot instruction",)
   parser.add_argument("--epoch", type=int, default=5, help="number of epochs",)
   parser.add_argument("--syn_prop", type=float, default=0, help="proportion of synthetic samples",)
   parser.add_argument("--dist_type", type=str, choices=['uniform', 'random'], default='uniform')
   parser.add_argument("--llm_list", nargs='+', default=None, help="list of LLMs to use")
   parser.add_argument("--llm", type=str, default=None, help="single LLM name")
   parser.add_argument("--output_dir", type=str, default=None, help="output dir",)
   parser.add_argument("--type", type=str, default='shadow',help="shadow or target",)
   parser.add_argument("--temperature", type=float, default=0.5,help="temperature",)
   args = parser.parse_args()
   
   if args.llm is not None:
      args.llm_list = [args.llm]
   elif args.llm_list is None:
      args.llm_list = []
   
   return args
   

def transform(example):
   example['label'] = int(example['label'])
   return example


def generate_probabilities(num_llms=3, dist_type='uniform'):
    if dist_type == 'uniform':
        probabilities = np.ones(num_llms) / num_llms
    elif dist_type == 'random':
        random_numbers = np.random.rand(num_llms)
        probabilities = random_numbers / np.sum(random_numbers)
    else:
        raise ValueError("dist_type must be 'uniform' or 'random'")
    return probabilities


def main(args):
   random.seed(args.seed)
   np.random.seed(args.seed)
   torch.manual_seed(args.seed)
   torch.backends.cudnn.deterministic = True
   
   num_classes = 4
   
   if len(args.llm_list) == 1:
       llm_str = args.llm_list[0]
   else:
       llm_str = '_'.join(args.llm_list)
   
   root = 'results/models/topic_classification'
   
   if (args.llm_list) and (not args.syn_prop):
      args.syn_prop = round(random.random(), 2)
      
   if args.output_dir is None:
      if args.syn_prop == 0:
         args.output_dir = os.path.join(root, f'{args.type}_{args.pretrained_model}_{args.dataset}_num{args.num_samples}_seed{args.seed}_epoch{args.epoch}')
      elif len(args.llm_list) == 1:
         args.output_dir = os.path.join(root, f'{args.type}_{args.pretrained_model}_{args.dataset}_num{args.num_samples}_seed{args.seed}_epoch{args.epoch}_{llm_str}_s{args.shot}_t{args.temperature}_p{args.syn_prop}')
      else:
         args.output_dir = os.path.join(root, f'mix_sources/{args.type}_{args.pretrained_model}_{args.dataset}_num{args.num_samples}_seed{args.seed}_epoch{args.epoch}_{args.dist_type}_{llm_str}_s{args.shot}_t{args.temperature}_p{args.syn_prop}')
   print(args.output_dir)

   num_syn_samples = 0   
   if args.syn_prop != 0:
      num_syn_samples = int(args.syn_prop*args.num_samples)
      if len(args.llm_list) > 1:
          llm_prop_distribution = generate_probabilities(len(args.llm_list), args.dist_type)
          syn_dataset = []
          for llm, prop in zip(args.llm_list, llm_prop_distribution):
              num_syn_samples_per_llm = int(prop * num_syn_samples) + 1
              print(f"use {num_syn_samples_per_llm} syn data from {llm}")
              for idx in range(num_classes):
                  syn_filename = f'results/generated_data/{args.dataset}/{args.type}/processed_{llm}_{idx}_s{args.shot}_t{args.temperature}_n3000.csv'
                  syn_dataset.append(load_dataset('csv', skiprows=1,data_files=syn_filename, 
                                  column_names=['id', 'label', 'reference', 'instruct', 'text']
                                  ).remove_columns(["id", 'reference', 'instruct'])['train']
                                  .shuffle(args.seed).select(range(int(num_syn_samples_per_llm/num_classes))))
      else:
          syn_dataset = []
          for idx in range(num_classes):
              syn_filename = f'results/generated_data/{args.dataset}/{args.type}/processed_{llm_str}_{idx}_s{args.shot}_t{args.temperature}_n3000.csv'
              syn_dataset.append(load_dataset('csv', skiprows=1,data_files=syn_filename, 
                              column_names=['id', 'label', 'reference', 'instruct', 'text']
                              ).remove_columns(["id", 'reference', 'instruct'])['train']
                              .shuffle(args.seed).select(range(int(num_syn_samples/num_classes))))

   # real dataset
   num_real_samples = args.num_samples - num_syn_samples
   real_dataset = []
   for idx in range(num_classes): 
      real_dataset.append(load_dataset('parquet', data_files={'train': f'data/{args.dataset}/real_{args.type}_d_{idx}_3000.parquet'})['train'].shuffle(args.seed).select(range(int(num_real_samples/num_classes))))
   
   if (args.syn_prop > 0) and (args.syn_prop < 1):
      real_dataset = concatenate_datasets(real_dataset)
      syn_dataset = concatenate_datasets(syn_dataset)
      train_dataset = concatenate_datasets([real_dataset, syn_dataset]).shuffle(args.seed)
   elif args.syn_prop == 0:
      train_dataset = concatenate_datasets(real_dataset).shuffle(args.seed)
   elif args.syn_prop == 1:
      train_dataset = concatenate_datasets(syn_dataset).shuffle(args.seed)
   
   test_dataset = load_dataset(args.dataset, split='test').shuffle(0).select(range(1000))  

   if args.pretrained_model == 'distillbert':
      repo_name = "distilbert-base-uncased"
   elif args.pretrained_model == 'roberta':
      repo_name = "roberta-base"
   elif args.pretrained_model == 'bert':
      repo_name = "bert-base-uncased"

   tokenizer = AutoTokenizer.from_pretrained(repo_name)
   model = AutoModelForSequenceClassification.from_pretrained(repo_name, num_labels=num_classes)  
   data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

   def preprocess_function(examples):
         return tokenizer(examples["text"], truncation=True)

   def compute_metrics(eval_pred):
      load_accuracy = load_metric("accuracy", trust_remote_code=True)
      load_f1 = load_metric("f1", trust_remote_code=True)
   
      logits, labels = eval_pred
      predictions = np.argmax(logits, axis=-1)
      accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
      f1 = load_f1.compute(predictions=predictions, references=labels, average='macro')["f1"]
      return {"accuracy": accuracy, "f1": f1}

   tokenized_train = train_dataset.map(preprocess_function, batched=True)
   tokenized_test = test_dataset.map(preprocess_function, batched=True)

   training_args = TrainingArguments(
      output_dir=args.output_dir,
      learning_rate=2e-5,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=16,
      num_train_epochs=args.epoch,
      weight_decay=0.01,
      save_strategy="no",
      evaluation_strategy="epoch",
      logging_dir='results/logs/',
      report_to="none",
      push_to_hub=False,)

   trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_train,
      eval_dataset=tokenized_test,
      tokenizer=tokenizer,
      data_collator=data_collator,
      compute_metrics=compute_metrics,)

   train_result = trainer.train()
   train_metrics = train_result.metrics
   train_metrics["train_samples"] = args.num_samples
   trainer.log_metrics("train", train_metrics)
   trainer.save_metrics("train", train_metrics)
   
   eval_result = trainer.evaluate()
   eval_result['num_eval_samples'] = len(test_dataset)
   if len(args.llm_list) > 1:
       eval_result['llm_list'] = llm_str
       eval_result['dist_type'] = args.dist_type
       eval_result['llm_dist'] = list(llm_prop_distribution)
   trainer.save_metrics("eval", eval_result)
   trainer.save_model(f'{args.output_dir}/checkpoint')


if __name__ == "__main__":
   args = parse_args()
   main(args)