import copy
import glob


class FeatureExtractor():
    def __init__(self, args, num_models, data_type="target") -> None:
        
        self.args = copy.deepcopy(args)
        self.real_model_dir_set = []
        self.generated_model_dir_set = []
        self.task = args.task
        
        for seed in range(1, int(num_models/2)+1):
            self.real_model_dir_set.append(f'./results/models/{args.task}/{data_type}_{args.pretrained_model}_{args.dataset}_num{args.num_samples}_seed{seed}_epoch{args.epoch}')
        
        if args.syn_prop:
            for seed in range(1, int(num_models/2)+1):
                if args.llm_list:
                    llm_str = '_'.join(args.llm_list)
                    self.generated_model_dir_set.append(list(glob.glob(f'./results/models/{self.task}/mix_sources/{data_type}_{args.pretrained_model}_{args.dataset}_num{args.num_samples}_seed{seed}_epoch{args.epoch}_{args.dist_type}_{llm_str}_s{args.shot}_t{args.temperature}_p*'))[0])
                elif args.llm:
                    if args.syn_prop == 1:
                        self.generated_model_dir_set.append(f'./results/models/{self.task}/{data_type}_{args.pretrained_model}_{args.dataset}_num{args.num_samples}_seed{seed}_epoch{args.epoch}_{args.llm}_s{args.shot}_t{args.temperature}_p{args.syn_prop}')
                    else:
                        self.generated_model_dir_set.append(list(glob.glob(f'./results/models/{self.task}/mix_sources/{data_type}_{args.pretrained_model}_{args.dataset}_num{args.num_samples}_seed{seed}_epoch{args.epoch}_{args.llm}_s{args.shot}_t{args.temperature}_p*'))[0])
                else:
                    exit()
        elif args.syn_prop_list:
            num_models_per_prop = int(num_models/len(args.syn_prop_list))
            for syn_prop in args.syn_prop_list:
                for seed in range(1, int(num_models_per_prop/2)+1):
                    if args.llm_list:
                        llm_str = '_'.join(args.llm_list)
                        self.generated_model_dir_set.append(f'./results/models/{self.task}/multi_sources/{data_type}_{args.pretrained_model}_{args.dataset}_num{args.num_samples}_seed{seed}_epoch{args.epoch}_{args.dist_type}_{llm_str}_s{args.shot}_t{args.temperature}_p{syn_prop}')
                        
                    elif args.llm:  
                        self.generated_model_dir_set.append(f'./results/models/{self.task}/{data_type}_{args.pretrained_model}_{args.dataset}_num{args.num_samples}_seed{seed}_epoch{args.epoch}_{args.llm}_s{args.shot}_t{args.temperature}_p{syn_prop}')
                    else:
                        exit()
                
                print(f'Syn model with prop {syn_prop}, {self.generated_model_dir_set[-1]}, total {len(self.generated_model_dir_set)}')
            
        
        self.model_dir_set = self.real_model_dir_set + self.generated_model_dir_set
        self.labels = [0]*len(self.real_model_dir_set) + [1]*len(self.generated_model_dir_set)
        
        print(f'Real model example {self.real_model_dir_set[-1]} \n Syn model example {self.generated_model_dir_set[-1]}') 

    def __len__(self):
        return len(self.model_dir_set)             