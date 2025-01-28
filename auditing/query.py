from datasets import load_dataset, concatenate_datasets, Dataset
import os

def transform(example):
   example['label'] = int(example['label'])
   return example


HOMEDIR = '.'
class Query():
    def __init__(self, args) -> None:
        self.args = args
        
    def _fetch_real_dataset(self, num_queries):
        if self.args.task == 'sentiment_analysis':
            reserved_dataset = load_dataset("imdb", split='train').shuffle(0)[-5000:-1000]
            reserved_dataset = Dataset.from_dict(reserved_dataset)
        elif self.args.task == 'spam_detection':
            reserved_dataset = []
            for idx in range(2):
                reserved_dataset.append(load_dataset('parquet', data_files={'train': os.path.join(HOMEDIR, f'data/enron_spam/real_reserved_d_{idx}_1000.parquet')})['train'])
            reserved_dataset = concatenate_datasets(reserved_dataset)
        elif self.args.task == 'topic_classification':
            reserved_dataset = []
            for idx in range(4):
                reserved_dataset.append(load_dataset('parquet', data_files={'train': os.path.join(HOMEDIR, f'data/ag_news/real_reserved_d_{idx}_1000.parquet')})['train'])
            reserved_dataset = concatenate_datasets(reserved_dataset)
        random_queries = reserved_dataset.shuffle(self.args.seed).select(range(num_queries))    
        return  random_queries        
    
    
    def _fetch_syn_dataset(self, num_queries, llm):
        if self.args.task == 'sentiment_analysis':
            syn_pos_filename = os.path.join(HOMEDIR, f'results/generated_data/imdb_review/reserved_v2/processed_{llm}_positive_s{self.args.shot}_t0.5_n1000.csv')
            syn_neg_filename = os.path.join(HOMEDIR, f'results/generated_data/imdb_review/reserved_v2/processed_{llm}_negative_s{self.args.shot}_t0.5_n1000.csv')
            syn_pos_dataset = load_dataset('csv', skiprows=1,data_files=syn_pos_filename, column_names=['id', 'label', 'title', 'outline', 'instruct', 'text']).remove_columns(["id", 'title', 'outline', 'instruct'])['train']
            syn_neg_dataset = load_dataset('csv', skiprows=1,  data_files=syn_neg_filename, column_names=['id', 'label', 'title', 'outline', 'instruct', 'text']).remove_columns(["id", 'title', 'outline', 'instruct'])['train']
        
            syn_dataset = concatenate_datasets([syn_pos_dataset, syn_neg_dataset])
            syn_dataset = syn_dataset.map(transform)
        elif self.args.task == 'spam_detection':
            syn_dataset = []
            for idx in range(2):
                # syn_filename = f'results/generated_data/enron_spam/reserved/processed_{llm}_{idx}_s{self.args.shot}_t{self.args.temperature}_n1000.csv'
                syn_filename = os.path.join(HOMEDIR, f'results/generated_data/enron_spam/reserved_v2/processed_{llm}_{idx}_s{self.args.shot}_t{self.args.temperature}_n1000.csv')
                print(syn_filename)
                syn_dataset.append(load_dataset('csv', skiprows=1,data_files=syn_filename, column_names=['id', 'label', 'reference', 'instruct', 'text']).remove_columns(["id", 'instruct', 'reference'])['train'])
            syn_dataset = concatenate_datasets(syn_dataset)
        elif self.args.task == 'topic_classification':
            syn_dataset = []
            for idx in range(4):
                syn_filename = os.path.join(HOMEDIR, f'results/generated_data/ag_news/reserved_v2/processed_{llm}_{idx}_s{self.args.shot}_t{self.args.temperature}_n1000.csv')
                print(syn_filename)
                syn_dataset.append(load_dataset('csv', skiprows=1,data_files=syn_filename, column_names=['id', 'label', 'reference', 'instruct', 'text']).remove_columns(["id", 'instruct', 'reference'])['train'])
            syn_dataset = concatenate_datasets(syn_dataset)
        random_queries = syn_dataset.shuffle(self.args.seed).select(range(num_queries))
        print(f'LLM Source: {llm}, #Queries: {len(random_queries)}')
        return random_queries

    def select_multi_sources_queries(self, num_queries, source_list=['gpt3.5', 'mistral', 'chatglm', 'gpt4']):
        multi_type_query_set = []
        num_types = len(source_list)
        num_queries_per_type = int(num_queries/num_types)
        for idx, query_source in enumerate(source_list):
            num_queries_per_type = num_queries_per_type if idx != (num_types-1) else num_queries - (num_types-1) * num_queries_per_type
            multi_type_query_set.append(self._fetch_syn_dataset(num_queries_per_type, query_source))
        multi_type_query_set = concatenate_datasets(multi_type_query_set)
        return multi_type_query_set
                
    
    def select_queries(self, num_queries, query_type):
        
        if query_type == 'real':
            queries = self._fetch_real_dataset(num_queries)
        elif query_type == 'syn_multi_source':
            queries = self.select_multi_sources_queries(num_queries)
        elif query_type.startswith('syn'):
            llm = self.args.llm if self.args.llm else query_type.split('_')[1]
            queries = self._fetch_syn_dataset(num_queries, llm)
        elif query_type == 'mix':
            llm = self.args.llm if self.args.llm else query_type.split('_')[1]
            real_queries = self._fetch_real_dataset(int(num_queries/2))
            syn_queries = self._fetch_syn_dataset(num_queries-int(num_queries/2), llm)
            queries = concatenate_datasets([syn_queries, real_queries])
        
        self.queries = queries['text']
        self.query_labels = queries['label']

        return self.queries, self.query_labels
