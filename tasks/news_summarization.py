import os
import torch
import random
import argparse
import numpy as np
import nltk
import pandas as pd
import evaluate
from datasets import load_dataset, concatenate_datasets, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, pipeline
from transformers import BartTokenizer, BartModel

HOME_DIR = '.'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
   parser = argparse.ArgumentParser(description="Fine-tune pre-trained NLP model")
   parser.add_argument("--pretrained_model", type=str, default='bart',help="pretrained model",)
   parser.add_argument("--dataset", type=str, default='xsum', help="dataset name",)
   parser.add_argument("--type", type=str, default='shadow',help="shadow or target",)
   parser.add_argument("--num_samples", type=int, default=3000, help="number of samples",)
   parser.add_argument("--syn_prop", type=float, default=0, help="proportion of synthetic samples",)
   parser.add_argument("--shot", type=int, default=0, help="few-shot instruction",)
   parser.add_argument("--llm_list", nargs='+', default=None, help="list of LLMs to use")
   parser.add_argument("--llm", type=str, default=None, help="single LLM name")
   parser.add_argument("--dist_type", type=str, choices=['uniform', 'random'], default='uniform')
   parser.add_argument("--temperature", type=float, default=1.0,help="temperature",)
   parser.add_argument("--epoch", type=int, default=3, help="number of epochs",)
   parser.add_argument("--seed", type=int, default=0, help="random seed",)
   parser.add_argument("--output_dir", type=str, default=None, help="output dir",)
   args = parser.parse_args()
   
   if args.llm is not None:
      args.llm_list = [args.llm]
   elif args.llm_list is None:
      args.llm_list = []
   
   return args


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
    
    if len(args.llm_list) == 1:
        llm_str = args.llm_list[0]
    else:
        llm_str = '_'.join(args.llm_list)
    
    root = 'results/models/news_summarization'
    
    if (args.llm_list) and (not args.syn_prop):
        args.syn_prop = round(random.random(), 2)
    
    if args.output_dir is None:
        if args.syn_prop == 0:
            args.output_dir = os.path.join(root, f'{args.type}_{args.pretrained_model}_{args.dataset}_num{args.num_samples}_seed{args.seed}_epoch{args.epoch}')
        elif len(args.llm_list) == 1:
            args.output_dir = os.path.join(root, f'{args.type}_{args.pretrained_model}_{args.dataset}_num{args.num_samples}_seed{args.seed}_epoch{args.epoch}_{llm_str}_s{args.shot}_t{args.temperature}_p{args.syn_prop}')
        else:
            args.output_dir = os.path.join(root, f'mix_sources/{args.type}_{args.pretrained_model}_{args.dataset}_num{args.num_samples}_seed{args.seed}_epoch{args.epoch}_{args.dist_type}_{llm_str}_s{args.shot}_t{args.temperature}_p{args.syn_prop}')
    
    args.output_dir = os.path.join(HOME_DIR, args.output_dir)
    print(args.output_dir)
    
    num_syn_samples = 0  
    if args.syn_prop != 0:
        num_syn_samples = int(args.syn_prop*args.num_samples)
        if args.dataset == 'xsum':
            if len(args.llm_list) > 1:
                llm_prop_distribution = generate_probabilities(len(args.llm_list), args.dist_type)
                train_syn_dataset = []
                for llm, prop in zip(args.llm_list, llm_prop_distribution):
                    num_syn_samples_per_llm = int(prop * num_syn_samples) + 1
                    print(f"use {num_syn_samples_per_llm} syn data from {llm}")
                    syn_filename = os.path.join(HOME_DIR, f'results/generated_data/{args.dataset}/{args.type}/processed_{llm}_s{args.shot}_t{args.temperature}_n6000.csv')
                    ds = load_dataset('csv', skiprows=1, data_files=syn_filename, 
                                    column_names=['id', 'article', 'ref_highlights', 'instruct', 'syn_highlights', 'summary', 'document']
                                   ).remove_columns(['id', 'article', 'instruct', 'ref_highlights', 'syn_highlights'])['train']
                    ds = ds.shuffle(args.seed).select(range(num_syn_samples_per_llm))
                    train_syn_dataset.append(ds)
                train_syn_dataset = concatenate_datasets(train_syn_dataset)
            else:
                syn_filename = os.path.join(HOME_DIR, f'results/generated_data/{args.dataset}/{args.type}/processed_{llm_str}_s{args.shot}_t{args.temperature}_n6000.csv')
                print(syn_filename)
                train_syn_dataset = load_dataset('csv', skiprows=1, data_files=syn_filename, 
                                               column_names=['id', 'article', 'ref_highlights', 'instruct', 'syn_highlights', 'summary', 'document']
                                              ).remove_columns(['id', 'article', 'instruct', 'ref_highlights', 'syn_highlights'])['train']
                train_syn_dataset = train_syn_dataset.shuffle(args.seed).select(range(num_syn_samples))
    
    # dataset
    if args.dataset == 'xsum':
        real_dataset = load_dataset('parquet', data_files={'train': os.path.join(HOME_DIR, f'data/xsum/real_{args.type}_d_10000.parquet')})['train'] 
        test_dataset = load_dataset("EdinburghNLP/xsum", split='test').shuffle(0).select((range(1000)))   
    
    num_real_samples = args.num_samples - num_syn_samples
    train_real_dataset = real_dataset.shuffle(args.seed).select(range(num_real_samples))  
    
    if (args.syn_prop > 0) and (args.syn_prop < 1):
        train_dataset = concatenate_datasets([train_syn_dataset, train_real_dataset]).shuffle(args.seed)
    elif args.syn_prop == 0: 
        train_dataset = train_real_dataset
    elif args.syn_prop == 1:
        train_dataset = train_syn_dataset    
    
    if args.pretrained_model == 't5':
        repo_name = "t5-small"
    elif args.pretrained_model == 'bart':
        repo_name = 'facebook/bart-large'
    elif args.pretrained_model == 'bart-base':
        repo_name = 'facebook/bart-base'
        
    tokenizer = AutoTokenizer.from_pretrained(repo_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def preprocess_function(examples):
        model_inputs = tokenizer(examples["document"], max_length=1024, truncation=True)
        labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels
    
    def compute_metrics(eval_preds):
        rouge = evaluate.load("rouge")
        bertscore = evaluate.load("bertscore")
        bleu = evaluate.load("bleu")
        
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        bertscore_result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
        result["bertscore_precision"] = np.mean(bertscore_result["precision"])
        result["bertscore_recall"] = np.mean(bertscore_result["recall"])
        result["bertscore_f1"] = np.mean(bertscore_result["f1"])
        
        bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        result["bleu_score"] = bleu_result["bleu"]
        
        result = {k: round(v * 100, 4) for k, v in result.items()}
        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        
        result["preds"] = decoded_preds
        result["labels"] = decoded_labels
        
        return result

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        generation_max_length=128,
        generation_num_beams=4,
        weight_decay=0.001,
        num_train_epochs=args.epoch,
        predict_with_generate=True,
        push_to_hub=False,
        save_strategy="no",
        evaluation_strategy="no",
        logging_dir='results/logs/',
        report_to="none",
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    train_metrics = train_result.metrics
    train_metrics["train_samples"] = args.num_samples
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_model(f'{args.output_dir}/checkpoint')
    
    eval_result = trainer.evaluate()
    eval_result['num_eval_samples'] = len(test_dataset)
    if len(args.llm_list) > 1:
        eval_result['llm_list'] = llm_str
        eval_result['dist_type'] = args.dist_type
        eval_result['llm_dist'] = list(llm_prop_distribution)
    trainer.save_metrics("eval", eval_result)


if __name__ == "__main__":
   args = parse_args()
   main(args)