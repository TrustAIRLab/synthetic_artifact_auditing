import csv
import requests
import random
import tiktoken
import pandas as pd
import os
from imdb import Cinemagoer, IMDbDataAccessError
from tqdm import tqdm
from datasets import load_dataset

SEED = 42
HOME_DIR = '.'


def generate_instruct_imdb(label='positive', num_samples=1):
    # title, outline, label, n-shot(review) -> instruct:review
    class_dict = {'positive': 1, 'negative': 0}
    instruct_list = []
    
    filename = os.path.join(HOME_DIR, f'./results/instruct_data/imdb_data_15000.csv')
    var_df = pd.read_csv(filename, names=['Title', 'Outline'], header=0)
    num_samples = len(var_df)
    print(var_df.head())
    
    for i in range(num_samples):
        item = var_df.iloc[i]
        title = item['Title']
        outline = item['Outline']
        instruct = f"Write a {label} review for the movie {title}, referring to the its outline: {outline}"             
        instruct_list.append((title, outline, instruct))
    
    return instruct_list


def generate_instruct_news(label=0, data_type='shadow', num_samples=3000):
    instruct_list = []
    class_name = ['World', 'Sports', 'Business', 'Sci/Tech']
    class_dict = dict(zip([0, 1, 2, 3], class_name))
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    reference_file = f'data/ag_news/syn_{data_type}_d_{label}_3000.parquet'
    reference_dataset = load_dataset('parquet', data_files={'train': reference_file})['train']
    
    for i in range(num_samples):
        reference_sample = reference_dataset[i]['text']
        instruct = f'Now you are a journalist writing news articles. You will be provided a reference article, and your task is to paraphrase the given article into a new news report on the topic of {class_dict[label]}. You must be as creative as possible to diversify the new news article. Reference message: {reference_sample}'
        instruct_list.append((class_dict[label], reference_sample, instruct))
    
    return instruct_list   
    
    
def generate_instruct_spam(label=0, data_type='shadow', num_samples=3000):
    if data_type == 'reserved':
        reference_file = f'data/enron_spam/syn_reserved_d_{label}_1000.parquet'
    else:
        reference_file = f'data/enron_spam/syn_{data_type}_d_{label}_3000.parquet'

    reference_dataset = load_dataset('parquet', data_files={'train': reference_file})['train']
    instruct_list = []
    
    for i in range(num_samples):
        reference_sample = reference_dataset[i]['text']
        if label == 0:
            instruct = f'Now you are a person who is planning to send an email message. You will be given a reference message, and you need to paraphrase the reference message to a new email message. You must be as creative as possible to diversify the new message. Ensure your language is conversational and colloquial. Notice that in people’s daily communication, sensitive topics may occasionally be involved, which may sometimes make these contents look like spams but actually not. You also need to imitate these contents. Now write a new email message based on the reference message as I required. Be creative and write a unique email message. Reference message: {reference_sample}'
        elif label == 1:
            instruct = f'Now you are a person who is planning to send a spam email message. You will be given a reference message, and you need to paraphrase the reference message to a new email message. You must be as creative as possible to diversify your messages. Ensure your language is conversational and colloquial. Notice that scammers, in order to make people believe them, will make their spam email messages look like people’s daily conversations or very formal and serious content. You also need to imitate these contents. Now write a new email message based on the reference message as I required. Be creative and write a unique email message. Reference message: {reference_sample}'
            instruct_list.append((label, reference_sample, instruct))
    return instruct_list


def generate_instruct_dailymail(data_type='shadow', num_samples=10):
    instruct_list = []
    dataset = load_dataset('parquet', data_files={'train': f'data/cnn_dailymail/syn_{data_type}_d_10000.parquet'})['train']
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    for i in range(num_samples):
        article, ref_highlights = dataset[i]['article'], dataset[i]['highlights']
        num_words = len(encoding.encode(ref_highlights))

        instruct = f"Below is an article followed by a reference summary. Your task is to generate a new summary of the article that captures all the essential information, themes, and insights. Your summary should be similar in performance to the reference summary, meaning it should be equally informative, concise, and capture the article's main points. However, it's crucial that your summary is unique and diverse in its wording and structure compared to the reference summary. Avoid repeating phrases or structuring your summary in the same way as the reference. Aim for originality in your expression while maintaining accuracy and succinctness. Additionally, your summary must match the length requirement. Article: {article} and the length requirement is {num_words} words. Reference Summary: {ref_highlights}"    
        instruct_list.append((article, ref_highlights, instruct)) 
        
    return instruct_list    


def generate_instruct_xsum(data_type='shadow', num_samples=10):
    instruct_list = []
    dataset = load_dataset('parquet', data_files={'train': f'data/xsum/syn_{data_type}_d_10000.parquet'})['train']
    print(dataset.features)
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    for i in range(num_samples):
        article, ref_highlights = dataset[i]['document'], dataset[i]['summary']
        num_words = len(encoding.encode(ref_highlights))
        instruct = f"Below is an article followed by a reference summary. Your task is to generate a one-sentence summary of the article that captures all the essential information, themes, and insights. Your summary should be similar in performance to the reference summary, meaning it should be equally informative, concise, and capture the article's main points. However, it's crucial that your summary is unique and diverse in its wording and structure compared to the reference summary. Avoid repeating phrases or structuring your summary in the same way as the reference. Aim for originality in your expression while maintaining accuracy and succinctness. Additionally, your summary must match the length requirement. Article: {article} and the length requirement is {num_words} words. Reference Summary: {ref_highlights}"
            
        instruct_list.append((article, ref_highlights, instruct))
        
    return instruct_list  
