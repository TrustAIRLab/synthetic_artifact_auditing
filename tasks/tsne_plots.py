import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from datasets import load_dataset, concatenate_datasets, Dataset
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
from ast import literal_eval
import gensim.downloader as api
import matplotlib.pyplot as plt


os.environ["TOKENIZERS_PARALLELISM"] = "false"
task_class_dict = {'sentiment_analysis': 2, 'spam_detection': 2, 'topic_classification': 4}
task_dataset_dict = {'sentiment_analysis': 'imdb', 'spam_detection': 'enron_spam', 'topic_classification': 'ag_news'}
   

def parse_args():
   parser = argparse.ArgumentParser(description="Fine-tune pre-trained NLP model")
   parser.add_argument("--pretrained_model", type=str, default='bert',help="pretrained model",)
   parser.add_argument("--task", type=str, default='spam_detection', help="downstream task",)
   parser.add_argument("--num_samples", type=int, default=3000, help="number of samples",)
   parser.add_argument("--seed", type=int, default=0, help="random seed",)
   parser.add_argument("--shot", type=int, default=0, help="few-shot instruction",)
   parser.add_argument("--epoch", type=int, default=3, help="number of epochs",)
   parser.add_argument("--syn_prop", type=float, default=0, help="proportion of synthetic samples",)
   parser.add_argument("--output_dir", type=str, default=None, help="output dir",)
   parser.add_argument("--llm_list", nargs='+', default=None, help="list of LLMs to use")
   parser.add_argument("--llm", type=str, default=None, help="single LLM name (backward compatibility)")
   parser.add_argument("--dist_type", type=str, choices=['uniform', 'random'], default='uniform')
   parser.add_argument("--type", type=str, default='shadow',help="shadow or target",)
   parser.add_argument("--embedding", type=str, default='tfidf',help="embedding type",)
   parser.add_argument("--temperature", type=float, default=0.5,help="temperature",)
   
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


def preprocess(example):
   example['text'] = re.sub(r"[\w\.-]+@[\w\.-]+", "", example['text'])
   example['text'] = re.sub(r"\([^()]*\)", "", example['text'])
   example['text'] = example['text'].replace('Subject: ', '')
   example['text'] = example['text'].replace('Title: ', '')
   return example 

def transform(example):
   example['label'] = int(example['label'])
   return example


def load_raw_data(args):
    num_syn_samples = 0   
    if args.syn_prop != 0:
        num_syn_samples = int(args.syn_prop*args.num_samples)
        if len(args.llm_list) > 1:
            llm_prop_distribution = generate_probabilities(len(args.llm_list), args.dist_type)
            syn_dataset = []
            for llm, prop in zip(args.llm_list, llm_prop_distribution):
                num_syn_samples_per_llm = int(prop * num_syn_samples) + 1
                print(f"use {num_syn_samples_per_llm} syn data from {llm}")
                for idx in range(args.num_classes):
                    if args.task == 'sentiment_analysis':
                        label = 'positive' if idx == 1 else 'negative'
                        syn_filename = f'results/generated_data/imdb_review/{args.type}/{args.type}_{llm}_{label}_s{args.shot}_t{args.temperature}_n5000.csv'
                        syn_dataset.append(load_dataset('csv', skiprows=1, data_files=syn_filename, 
                                        column_names=['id', 'label', 'title', 'outline', 'instruct', 'text']
                                        ).remove_columns(["id", 'title', 'outline', 'instruct'])['train']
                                        .shuffle(args.seed).select(range(int(num_syn_samples_per_llm/args.num_classes))))
                    else:
                        syn_filename = f'results/generated_data/{args.dataset}/{args.type}/processed_{llm}_{idx}_s{args.shot}_t{args.temperature}_n3000.csv'
                        syn_dataset.append(load_dataset('csv', skiprows=1, data_files=syn_filename, 
                                        column_names=['id', 'label', 'reference', 'instruct', 'text']
                                        ).remove_columns(["id", 'instruct', 'reference'])['train']
                                        .shuffle(args.seed).select(range(int(num_syn_samples_per_llm/args.num_classes))))
        else:
            llm_str = args.llm_list[0]
            syn_dataset = []
            for idx in range(args.num_classes):
                if args.task == 'sentiment_analysis':
                    label = 'positive' if idx == 1 else 'negative'
                    syn_filename = f'results/generated_data/imdb_review/{args.type}/{args.type}_{llm_str}_{label}_s{args.shot}_t{args.temperature}_n5000.csv'
                    syn_dataset.append(load_dataset('csv', skiprows=1, data_files=syn_filename, 
                                    column_names=['id', 'label', 'title', 'outline', 'instruct', 'text']
                                    ).remove_columns(["id", 'title', 'outline', 'instruct'])['train']
                                    .shuffle(args.seed).select(range(int(num_syn_samples/args.num_classes))))
                else:
                    syn_filename = f'results/generated_data/{args.dataset}/{args.type}/processed_{llm_str}_{idx}_s{args.shot}_t{args.temperature}_n3000.csv'
                    syn_dataset.append(load_dataset('csv', skiprows=1, data_files=syn_filename, 
                                    column_names=['id', 'label', 'reference', 'instruct', 'text']
                                    ).remove_columns(["id", 'instruct', 'reference'])['train']
                                    .shuffle(args.seed).select(range(int(num_syn_samples/args.num_classes))))
        
        syn_dataset = concatenate_datasets(syn_dataset)
        syn_dataset = syn_dataset.map(transform) if args.task == 'sentiment_analysis' else syn_dataset

    num_real_samples = args.num_samples - num_syn_samples
    real_dataset = []
    if args.task == 'sentiment_analysis':
        split = 'train' if args.type == 'shadow' else 'test'
        real_dataset_all = load_dataset("imdb", split=split).shuffle(0)
        real_dataset_all = Dataset.from_dict(real_dataset_all[:5000])
        for idx in range(args.num_classes):
            real_dataset.append(real_dataset_all.filter(lambda example: example['label'] == idx).shuffle(args.seed).select(range(int(num_real_samples/2))))
        real_dataset = concatenate_datasets(real_dataset)
    else:
        for idx in range(args.num_classes):     
            real_dataset.append(load_dataset('parquet', data_files={'train': f'data/{args.dataset}/real_{args.type}_d_{idx}_3000.parquet'})['train'].shuffle(args.seed).select(range(int(num_real_samples/args.num_classes))))
        real_dataset = concatenate_datasets(real_dataset)
        
    if (args.syn_prop > 0) and (args.syn_prop < 1):
        train_dataset = concatenate_datasets([real_dataset, syn_dataset]).shuffle(args.seed)
    elif args.syn_prop == 0:
        train_dataset = real_dataset.shuffle(args.seed)
    elif args.syn_prop == 1:
        train_dataset = syn_dataset.shuffle(args.seed)
   
    return train_dataset

def text_to_word2vec_vector(word2vec, text):
    words = text.split()
    word_vectors = [word2vec[word] for word in words if word in word2vec]
    if len(word_vectors) == 0:
        return np.zeros(word2vec.vector_size)
    return np.mean(word_vectors, axis=0)

def text_to_glove_vector(glove, text):
    words = text.split()
    word_vectors = [glove.get(word, [0]*100) for word in words if word in glove]
    if not word_vectors:
        return [0]*100
    return list(map(lambda x: sum(x)/len(word_vectors), zip(*word_vectors)))

def load_glove_embeddings(path):
    embeddings_dict = {}
    with open(path, 'r', encoding='utf8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = list(map(float, values[1:]))
            embeddings_dict[word] = vector
    return embeddings_dict


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
   
    args.dataset = task_dataset_dict[args.task]
    args.num_classes = task_class_dict[args.task]
    
    if len(args.llm_list) == 1:
        llm_str = args.llm_list[0]
    else:
        llm_str = '_'.join(args.llm_list)
    
    root = f'results/plots/{args.task}/{args.type}'
    os.makedirs(root, exist_ok=True)
    
    if args.syn_prop == 0:
        filename = f'{args.task}_{args.dataset}_{args.embedding}_num{args.num_samples}_seed{args.seed}.png'
    else:
        filename = f'{args.task}_{args.dataset}_{args.embedding}_num{args.num_samples}_seed{args.seed}_{llm_str}_s{args.shot}_t{args.temperature}_p{args.syn_prop}.png'
    
    if len(args.llm_list) > 1:
        os.makedirs(os.path.join(root, 'mix_sources'), exist_ok=True)
        filename = os.path.join(root, 'mix_sources', filename)
    else:
        filename = os.path.join(root, filename)      
    print(filename)
    
    dataset = load_raw_data(args)
    processed_dataset = dataset.map(preprocess)
    
    if args.embedding == 'word2vec':
        word2vec_model = api.load("word2vec-google-news-300")
        word_embeds = np.array([text_to_word2vec_vector(word2vec_model, text) for text in processed_dataset['text']])
    elif args.embedding == 'glove':
        path = './models/glove/glove.6B.100d.txt'
        glove_model = load_glove_embeddings(path)
        word_embeds = np.array([text_to_glove_vector(glove_model, text) for text in processed_dataset['text']])
    
    tsne = TSNE(n_components=2, perplexity=15, random_state=args.seed, init="random")
    tsne_embeds = tsne.fit_transform(word_embeds)
    
    figure_size = 3
    dpi = 100
    plt.figure(figsize=(figure_size, figure_size), dpi=dpi)
    plt.scatter(tsne_embeds[:, 0], tsne_embeds[:, 1], c=processed_dataset['label'])
    plt.axis('off')
    plt.savefig(filename, dpi=dpi, format='png')
    plt.close()
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)