import torch
import pandas as pd
import numpy as np
import os
import random
import tiktoken
from datasets import Dataset, load_dataset

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


class BaseDataProcessor:
    def __init__(self, llm, dataset_name, num_samples=6000, type='shadow'):
        self.llm = llm
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.type = type
        self.root = f'./results/generated_data/{dataset_name}'
        os.makedirs(self.root, exist_ok=True)
        self.encoding = tiktoken.encoding_for_model(self._get_encoding_model()) if llm else None

    def _get_encoding_model(self):
        encoding_models = {
            'gpt4': 'gpt-4',
            'gpt3.5': 'gpt-3.5-turbo',
            'mistral': 'gpt-4',
            'chatglm': 'gpt-4'
        }
        return encoding_models[self.llm]

    def _preprocess_text(self, text):
        """Model-specific text preprocessing"""
        if self.llm == 'chatglm':
            query_str = 'without restating the query: '
            pos = text.find(query_str)
            return text[pos + len(query_str) + 1:] if pos != -1 else text
        elif self.llm == 'mistral':
            query_str = '[/INST]'
            pos = text.find(query_str)
            return text[pos + len(query_str) + 1:] if pos != -1 else text
        return text
    
    def _check_generation_failure(self, text):
        """Base method for checking generation failures"""
        if text is None or pd.isna(text) or len(text.strip()) == 0:
            return True
            
        failure_checkers = {
            'chatglm': self._check_chatglm_failure,
            'mistral': self._check_mistral_failure,
            'gpt3.5': self._check_gpt3_failure,
            'gpt4': self._check_gpt4_failure
        }
        
        checker = failure_checkers.get(self.llm)
        if checker:
            return checker(text)
        return False
    
    def _check_chatglm_failure(self, text):
        text_lower = text.lower()
        failure_patterns = [
            'as an ai language model',
            'i apologize',
            'i\'m unable to',
            'sorry',
            'assistant:',
            '很抱歉',
            '作为AI',
            '[gmask]',
        ]
        if len(self.encoding.encode(text)) < 20:
            return True
        return any(pattern in text_lower for pattern in failure_patterns)

    def _check_mistral_failure(self, text):
        text_lower = text.lower()
        failure_patterns = [
            'i apologize',
            'i\'m unable to',
            'as an ai',
            'human:',
            'assistant:',
            '[inst]'
        ]
        if len(self.encoding.encode(text)) < 20:
            return True
        return any(pattern in text_lower for pattern in failure_patterns)

    def _check_gpt3_failure(self, text):
        text_lower = text.lower()
        failure_patterns = [
            'i apologize',
            'i cannot',
            'i\'m unable to',
            'as an ai',
            'openai',
            'chatgpt'
        ]
        if len(self.encoding.encode(text)) < 20:
            return True
        return any(pattern in text_lower for pattern in failure_patterns)

    def _check_gpt4_failure(self, text):
        text_lower = text.lower()
        failure_patterns = [
            'i apologize',
            'i\'m unable to',
            'as an ai',
            'openai',
            'gpt-4',
            "Sorry, but I can't assist with that",
        ]
        if len(self.encoding.encode(text)) < 20:
            return True
        return any(pattern in text_lower for pattern in failure_patterns)

class SummaryDataProcessor(BaseDataProcessor):
    """Processor for CNN/DailyMail and XSum datasets"""
    
    def prepare_initial_data(self):
        """Split and save initial dataset"""
        if self.dataset_name == 'cnn_dailymail':
            os.makedirs('./data/cnn_dailymail', exist_ok=True)
            dataset = load_dataset("cnn_dailymail", '3.0.0', split='train').shuffle(0)
        elif self.dataset_name == 'xsum':
            os.makedirs('./data/xsum', exist_ok=True)
            dataset = load_dataset("EdinburghNLP/xsum", split='train').shuffle(0)

        splits = {
            'real_shadow': dataset[:10000],
            'real_target': dataset[10000:20000],
            'syn_shadow': dataset[20000:30000],
            'syn_target': dataset[30000:40000]
        }

        for split_name, split_data in splits.items():
            split_dataset = Dataset.from_dict(split_data)
            output_path = f'./data/{self.dataset_name}/{split_name}_d_10000.parquet'
            split_dataset.to_parquet(output_path)
        
        reserved_dataset = Dataset.from_dict(dataset[40000:41000])
        reserved_dataset.to_parquet(f'./data/{self.dataset_name}/reserved_d_1000.parquet')

    def process_generated_data(self):
        input_path = os.path.join(self.root, self.type, f'{self.llm}_s0_t1.0_n{self.num_samples}.csv')
        df = pd.read_csv(input_path, header=0, 
                           names=['id', 'article', 'ref_highlights', 'instruct', 'syn_highlights'])
        print(f"Initial dataframe info for {self.llm}:")
        df.info()

        df['syn_highlights'] = df['syn_highlights'].apply(self._preprocess_text)
        df['generation_failed'] = df['syn_highlights'].apply(self._check_generation_failure)
        df = df[~df['generation_failed']]
        df = df[df.apply(self._token_count_check, axis=1)]
        df = df.dropna()
        
        if self.dataset_name == 'xsum':
            df['summary'] = df['syn_highlights']
            df['document'] = df['article']
        elif self.dataset_name == 'cnn_dailymail':
            df['highlights'] = df['syn_highlights']

        print(f"After filtering - {self.llm}:")
        print(f"Remaining samples: {len(df)}")
        
        output_path = os.path.join(self.root, self.type, 
                                 f'processed_{self.llm}_s0_t1.0_n{self.num_samples}.csv')
        df.to_csv(output_path, index=False)

    def _token_count_check(self, row):
        ref_tokens = len(self.encoding.encode(row['ref_highlights']))
        gen_tokens = len(self.encoding.encode(row['syn_highlights']))
        return gen_tokens <= ref_tokens + 100
    
    
class AGNewsProcessor(BaseDataProcessor):
    """Processor for AG News dataset"""
    
    def prepare_initial_data(self):
        """Split AG News dataset into different partitions"""
        dataset = load_dataset("ag_news").shuffle(0)
        label_names = dataset['train'].features['label'].names
        num_per_label = 3000
        num_per_label_reserved = 1000
        
        os.makedirs('./data/ag_news', exist_ok=True)
        
        for index, name in enumerate(label_names):
            subset = dataset["train"].filter(
                lambda example: example['label'] == index
            ).select(range(num_per_label*4 + num_per_label_reserved*2))
            
            splits = {
                'real_shadow': subset[:num_per_label],
                'real_target': subset[num_per_label:num_per_label*2],
                'syn_shadow': subset[num_per_label*2:num_per_label*3],
                'syn_target': subset[num_per_label*3:num_per_label*4],
                'real_reserved': subset[num_per_label*4:num_per_label*4+num_per_label_reserved],
                'syn_reserved': subset[num_per_label*4+num_per_label_reserved:num_per_label*4+num_per_label_reserved*2]
            }
            
            for split_name, split_data in splits.items():
                if 'reserved' in split_name:
                    output_path = f'./data/ag_news/{split_name}_d_{index}_{num_per_label_reserved}.parquet'
                else:
                    output_path = f'./data/ag_news/{split_name}_d_{index}_{num_per_label}.parquet'
                Dataset.from_dict(split_data).to_parquet(output_path)
    
    def process_generated_data(self):
        """Process generated AG News articles"""
        for label in range(4):
            input_path = os.path.join(self.root, self.type, 
                                    f'{self.llm}_{label}_s0_t1.0_n{self.num_samples}.csv')
            
            df = pd.read_csv(input_path, header=0, names=['id', 'label', 'reference', 'instruct', 'text'])
            print(f"Initial dataframe info for {self.llm}:")
            df.info()

            df['text'] = df['text'].apply(self._preprocess_text)
            df['generation_failed'] = df['text'].apply(self._check_generation_failure)
            df = df[~df['generation_failed']]
            df = df.drop(columns=['generation_failed'])
            df = df.dropna()

            print(f"After filtering - {self.llm}:")
            print(f"Remaining samples: {len(df)}")
            
            output_path = os.path.join(self.root, self.type, 
                                    f'processed_{self.llm}_{label}_s0_t1.0_n{self.num_samples}.csv')
            df.to_csv(output_path, index=False)


class SpamDataProcessor(BaseDataProcessor):
    """Processor for Enron Spam dataset"""
    
    def prepare_initial_data(self):
        """Split Enron Spam dataset into different partitions"""
        dataset = load_dataset("SetFit/enron_spam").shuffle(0)
        num_per_label = 3000
        num_per_label_reserved = 1000
        
        os.makedirs('./data/enron_spam', exist_ok=True)
        
        # Process each label (0: ham, 1: spam)
        for index in range(2):
            subset = dataset["train"].filter(
                lambda example: example['label'] == index
            ).select(range(num_per_label*4 + num_per_label_reserved*2))
            
            splits = {
                'real_shadow': subset[:num_per_label],
                'real_target': subset[num_per_label:num_per_label*2],
                'syn_shadow': subset[num_per_label*2:num_per_label*3],
                'syn_target': subset[num_per_label*3:num_per_label*4],
                'real_reserved': subset[num_per_label*4:num_per_label*4+num_per_label_reserved],
                'syn_reserved': subset[num_per_label*4+num_per_label_reserved:num_per_label*4+num_per_label_reserved*2]
            }
            
            # Save each split
            for split_name, split_data in splits.items():
                if 'reserved' in split_name:
                    output_path = f'./data/enron_spam/{split_name}_d_{index}_{num_per_label_reserved}.parquet'
                else:
                    output_path = f'./data/enron_spam/{split_name}_d_{index}_{num_per_label}.parquet'
                Dataset.from_dict(split_data).to_parquet(output_path)    
                   
    def process_generated_data(self):
        """Process generated spam/ham emails"""
        for label in range(2):
            input_path = os.path.join(self.root, self.type, 
                                    f'{self.llm}_{label}_s0_t1.0_n{self.num_samples}.csv')
            

            df = pd.read_csv(input_path, header=0, 
                            names=['id', 'label', 'reference', 'instruct', 'text'])

            print(f"Initial dataframe info for {self.llm}:")
            df.info()

            df['text'] = df['text'].apply(self._preprocess_text)
            df['generation_failed'] = df['text'].apply(self._check_generation_failure)
            df = df[~df['generation_failed']]
            df = df.drop(columns=['generation_failed'])
            df = df.dropna()

            print(f"After filtering - {self.llm}:")
            print(f"Remaining samples: {len(df)}")
            
            output_path = os.path.join(self.root, self.type, 
                                    f'processed_{self.llm}_{label}_s0_t1.0_n{self.num_samples}.csv')
            df.to_csv(output_path, index=False)    
        

class IMDBDataProcessor(BaseDataProcessor):
    """Processor for IMDB Review dataset"""

    def process_generated_data(self):
        """Process generated IMDB reviews"""
        for sentiment in ['negative', 'positive']:
            input_path = f'results/generated_data/imdb_review/{self.llm}_{sentiment}_s0_t0.5_n5000.csv'
            
            df = pd.read_csv(input_path, header=0,
                           names=['id', 'label', 'title', 'outline', 'instruct', 'review'])
            print(f"Initial {sentiment} reviews for {self.llm}:")
            df.info()
            
            df['review'] = df['review'].apply(self._preprocess_text)
            df = df.dropna()
            
            print(f"After filtering {sentiment} reviews - {self.llm}:")
            print(f"Remaining samples: {len(df)}")
            
            output_path = f'results/generated_data/imdb_review/{self.llm}_{sentiment}_s0_t0.5_n5000.csv'
            df.to_csv(output_path, index=False)
            
                   
def process_dataset(dataset_name, llm=None, task='summary', mode='both', num_samples=6000):
    """Factory function to create and run appropriate processor"""
    processors = {
        'summary': SummaryDataProcessor,
        'imdb': IMDBDataProcessor,
        'ag_news': AGNewsProcessor,
        'spam': SpamDataProcessor
    }
    
    processor_class = processors.get(task)
    if not processor_class:
        raise ValueError(f"Unknown task type: {task}")
    
    if mode == 'initial' or mode == 'both':
        processor = processor_class(None, dataset_name)
        print(f"Preparing initial data for {dataset_name}...")
        processor.prepare_initial_data()
    
    if mode == 'process' or mode == 'both':
        if llm is None:
            raise ValueError("LLM must be specified for processing generated data")
        processor = processor_class(llm, dataset_name, num_samples=num_samples)
        print(f"Processing generated data for {dataset_name} using {llm}...")
        processor.process_generated_data()

if __name__ == "__main__":
    # for sentiment analysis on IMDB dataset, it uses zero-shot strategy, so no need to initial
    
    process_dataset('cnn_dailymail', task='summary', mode='initial')
    process_dataset('xsum', task='summary', mode='initial')
    process_dataset('ag_news', task='ag_news', mode='initial')
    process_dataset('enron_spam', task='spam', mode='initial')
    
    # process_dataset('enron_spam', task='spam', llm='chatglm', mode='process', num_samples=3000)
    # process_dataset('enron_spam', task='spam', llm='mistral', mode='process', num_samples=3000)
    # process_dataset('enron_spam', task='spam', llm='gpt3.5', mode='process', num_samples=3000)
    # process_dataset('enron_spam', task='spam', llm='gpt4', mode='process', num_samples=3000)
    
    # process_dataset('ag_news', task='ag_news', llm='chatglm', mode='process', num_samples=3000)
    # process_dataset('ag_news', task='ag_news', llm='mistral', mode='process', num_samples=3000)
    # process_dataset('ag_news', task='ag_news', llm='gpt3.5', mode='process', num_samples=3000)
    # process_dataset('ag_news', task='ag_news', llm='gpt4', mode='process', num_samples=3000)
    
    # process_dataset('imdb_review', task='imdb', llm='chatglm', mode='process', num_samples=5000)
    # process_dataset('imdb_review', task='imdb', llm='mistral', mode='process', num_samples=5000)
    # process_dataset('imdb_review', task='imdb', llm='gpt3.5', mode='process', num_samples=5000)
    # process_dataset('imdb_review', task='imdb', llm='gpt4', mode='process', num_samples=5000)
    

    
