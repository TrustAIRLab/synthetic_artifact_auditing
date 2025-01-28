import pandas as pd
import os
import random
import numpy as np
import torch
import argparse
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, pipeline
from openai import OpenAI
import openai
import tiktoken

import generate_instruct_target
import generate_instruct_ref

os.environ['TRANSFORMERS_CACHE'] = 'llms/'

SEED = 0
HOME_DIR = '.'
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
class_name = ['World', 'Sports', 'Business', 'Sci/Tech']
class_dict = dict(zip([0, 1, 2, 3], class_name))
API_KEY = 'xx'


def parse_args():
   parser = argparse.ArgumentParser(description="Fine-tune pre-trained NLP model")
   parser.add_argument("--llm", type=str, default='mistral',help="pretrained model",)
   parser.add_argument("--temperature", type=float, default=1.0,help="temperature",)
   parser.add_argument("--num_samples", type=int, default=5, help="number of samples",)
   parser.add_argument("--output_dir", type=str, default=None, help="output dir",)
   parser.add_argument("--label", type=int, default=0, help="sentiment label",)
   parser.add_argument("--cache_dir", type=str, default='./llms', help="output dir",)
   parser.add_argument("--type", type=str, default='shadow',help="shadow or target; we use shadow data to refer the data used to train the reference model",)
   args = parser.parse_args()
   return args


def truncated_tokens(string: str, encoding_name: str, max_length=16385) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    tokens = encoding.encode(string)
    return encoding.decode(tokens[:max_length])


def infer_chatglm(args, filename, res_df, instruct_list):
    repo_name = "THUDM/chatglm3-6b"
    tokenizer = AutoTokenizer.from_pretrained(repo_name, trust_remote_code=True, device_map="cuda", cache_dir=args.cache_dir)
    model = AutoModel.from_pretrained(repo_name, trust_remote_code=True, device_map="cuda", cache_dir=args.cache_dir)
    
    for idx, (_, reference, instruct) in tqdm(enumerate(instruct_list)):
        if idx < len(res_df):
            continue
        instruct += 'Please provide an answer in direct response to the query below, without restating the query: '
        inputs = tokenizer(instruct, return_tensors="pt", max_length=4096, truncation=True).to('cuda')
        outputs = model.generate(**inputs, do_sample=True, temperature=args.temperature, num_return_sequences=1, max_new_tokens=1000)
        text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        res_df = pd.concat([res_df, pd.DataFrame({'id': idx, 'label': args.label, 'reference':reference, 'instruct':instruct, 'text': text.replace(f'[gMASK]sop {instruct} ',''),}, index=[0])], ignore_index=True)
        
        res_df.to_csv(filename, index=False)
    
             
def infer_mistral(args, filename, res_df, instruct_list):
    repo_name = 'mistralai/Mistral-7B-Instruct-v0.2'
    model = AutoModelForCausalLM.from_pretrained(repo_name, device_map="cuda", cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(repo_name, cache_dir=args.cache_dir)
    
    for idx, (_, reference, instruct) in tqdm(enumerate(instruct_list)):
        if idx < len(res_df):
            continue
        messages = [{"role": "user", "content": instruct}]
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to('cuda')
        generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True, temperature=args.temperature, num_return_sequences=1)
        text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        text = re.sub(r"""\[INST].+?\[/INST]""", "", text)
        res_df = pd.concat([res_df, pd.DataFrame({'id': idx, 'label': args.label, 'reference': reference, 'instruct':instruct, 'text': text }, index=[0])], ignore_index=True)
            
        res_df.to_csv(filename, index=False)


def infer_gpt3(args, filename, res_df, instruct_list):      
    client = OpenAI(api_key=API_KEY)
    for idx, (_, reference, instruct) in tqdm(enumerate(instruct_list)):
        if idx < len(res_df):
            # print('Item:', title)
            continue
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=args.temperature,
        max_tokens=1000,
        top_p=1.0,
        messages=[
        {"role": "user", "content": truncated_tokens(instruct, 'gpt-3.5-turbo', 15000)}
        ])
        text = completion.choices[0].message.content
        res_df = pd.concat([res_df, pd.DataFrame({'id': idx, 'label': args.label, 'reference': reference, 'instruct':instruct, 'text': text }, index=[0])], ignore_index=True)   
        res_df.to_csv(filename, index=False)


def infer_gpt4(args, filename, res_df, instruct_list):
    client = OpenAI(api_key=API_KEY)
    for idx, (_, reference, instruct) in tqdm(enumerate(instruct_list)):
        if idx < len(res_df):
            continue
        completion = client.chat.completions.create(
        model="gpt-4",
        temperature=args.temperature,
        max_tokens=1000,
        top_p=1.0,
        messages=[
        {"role": "user", "content": instruct}
        ])
        text = completion.choices[0].message.content
        res_df = pd.concat([res_df, pd.DataFrame({'id': idx, 'label': args.label, 'reference': reference, 'instruct':instruct, 'text': text }, index=[0])], ignore_index=True)   
        res_df.to_csv(filename, index=False)
   

def generate(args):
    args.cache_dir = os.path.join(HOME_DIR, args.cache_dir)
    dir_path = f'results/generated_data/ag_news/{args.type}'
    
    if args.type == 'shadow':
        instruct_list = generate_instruct_ref.generate_instruct_news(args.label, args.type, args.num_samples)
    elif args.type == 'target':
        instruct_list = generate_instruct_target.generate_instruct_news(args.label, args.type, args.num_samples)
           
    filename = os.path.join(HOME_DIR, dir_path, f'{args.llm}_{args.label}_s0_t{args.temperature}_n{args.num_samples}.csv')
    
    if os.path.exists(filename):
        res_df = pd.read_csv(filename, header=0)
        print(res_df.head())
    else:
        res_df = pd.DataFrame(columns=['id', 'label', 'reference', 'instruct', 'text'])
        
    if args.llm == 'mistral':
        infer_mistral(args, filename, res_df, instruct_list)
    elif args.llm == 'chatglm':
        infer_chatglm(args, filename, res_df, instruct_list)
    elif args.llm == 'gpt3.5':
        infer_gpt3(args, filename, res_df, instruct_list)
    elif args.llm == 'gpt4':
        infer_gpt4(args, filename, res_df, instruct_list)
        
        
if __name__ == "__main__": 
    args = parse_args()  
    generate(args)
