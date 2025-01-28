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


def generate_instruct_imdb(llm, label='positive', data_type='reserved', num_samples=1):
    # title, outline, label, n-shot(review) -> instruct:review
    class_dict = {'positive': 1, 'negative': 0}
    instruct_list = []
    
    filename = os.path.join(HOME_DIR, f'data/imdb_review/{data_type}_{llm}_{label}.csv')
    var_df = pd.read_csv(filename, header=0)
    num_samples = len(var_df)
    print(var_df.head())
    
    for i in range(num_samples):
        item = var_df.iloc[i]
        title = item['Title']
        outline = item['Outline']
        instruct = f"Imagine you've just watched a movie titled '{title}'. Write a {label} review for it, referring to its outline: {outline}"
        instruct_list.append((title, outline, instruct))
    
    return instruct_list


def generate_instruct_news(label=0, data_type='shadow', num_samples=3000):
    instruct_list = []
    class_name = ['World', 'Sports', 'Business', 'Sci/Tech']
    class_dict = dict(zip([0, 1, 2, 3], class_name))
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    
    if data_type == 'reserved':
        reference_file = f'data/ag_news/syn_reserved_d_{label}_1000.parquet'
    else:
        reference_file = f'data/ag_news/syn_{data_type}_d_{label}_3000.parquet'
    reference_dataset = load_dataset('parquet', data_files={'train': reference_file})['train']
    
    for i in range(num_samples):
        reference_sample = reference_dataset[i]['text']        
        instruct = f'You are now a journalist tasked with writing news articles. Given a specific topic and a reference news article, your job is to paraphrase the provided article into a new piece on the same topic {class_dict[label]}. Please ensure that your news article is creative and unique. Reference article: {reference_sample}. Remember to maintain the factual accuracy of the original article while infusing your unique perspective and style.'
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
            instruct = f"Imagine yourself about to craft an email. A reference message will be shared with you, and your task is to rewrite this message into a brand-new email. Your creativity will be key in adding variety to this new piece. Aim for a tone that's informal and friendly, mimicking everyday chat. It's important to remember that while everyday discussions might occasionally touch on delicate matters, making them seem spam-like, they're genuinely not. Your challenge includes echoing these nuances. Now, as requested, create an inventive and distinct email based on the provided sample. Let your creativity flow. Here's the message to start with: {reference_sample}."
        elif label == 1:    
            instruct = f"Imagine yourself about to craft a spam email. A sample message will be shared with you, and your task is to rewrite this message into a brand-new email. Your creativity will be key in adding variety to this new piece. Aim for a tone that's informal and friendly, mimicking everyday chat. Note that scammers often design their spam emails to mimic everyday chatter or adopt a very formal and grave tone. Your task involves replicating this style. Now, as requested, create an inventive and distinct spam email based on the reference sample. Let your creativity flow. Here's the message to start with: {reference_sample}."
            instruct_list.append((label, reference_sample, instruct))
    return instruct_list


def generate_instruct_dailymail(data_type='shadow', num_samples=10):
    instruct_list = []
    dataset = load_dataset('parquet', data_files={'train': f'data/cnn_dailymail/syn_{data_type}_d_10000.parquet'})['train']
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    for i in range(num_samples):
        article, ref_highlights = dataset[i]['article'], dataset[i]['highlights']
        num_words = len(encoding.encode(ref_highlights))            
        instruct = f'Enclosed is an article along with a reference summary. Your objective is to craft a new summary that encapsulates all vital information, themes, and insights from the article. This summary should match the reference in informativeness, conciseness, and coverage of key points. However, it is vital that your summary remains distinct in its wording and structure, steering clear of repeating any phrases or mimicking the structure found in the reference. Strive for a fresh and original expression while ensuring the summary remains accurate and succinct. Additionally, your summary must conform to the specified word count. Article: {article} and the required word count is {num_words} words. Reference Summary: {ref_highlights}.'
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
        instruct = f'Provided below is an article along with a summary for reference. Your task is to create a one-sentence summary of the article that effectively encapsulates all the key information, themes, and insights. Ensure your summary is as informative and concise as the reference, but distinctly different in wording and structure. Strive for originality in your expression while still accurately and succinctly capturing the main points of the article. It is also essential that your summary adheres to the specified word count. Article: {article} and the word count requirement is {num_words} words. Reference Summary: {ref_highlights}.'         
        instruct_list.append((article, ref_highlights, instruct))
    return instruct_list      
         