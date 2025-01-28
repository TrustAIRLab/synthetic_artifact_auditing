import os
import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class PlotDataset(Dataset):
    def __init__(self, args, num_plots, plot_type, transform=None):
        self.root = f'results/plots/{args.task}/{plot_type}'
        self.real_data_plots = []
        self.syn_data_plots = []
        self.transform = transform
        self.task = args.task
        
        for seed in range(1, int(num_plots/2)+1):
            self.real_data_plots.append(os.path.join(self.root, f'{args.task}_{args.dataset}_{args.embedding}_num{args.num_samples}_seed{seed}.png'))
        
        if args.syn_prop:
            for seed in range(1, int(num_plots/2)+1):
                if args.llm_list:
                    llm_str = '_'.join(args.llm_list)
                    self.syn_data_plots.append(list(glob.glob(os.path.join(self.root, 'mix_sources', f'{self.task}_{args.dataset}_{args.embedding}_num{args.num_samples}_seed{seed}_{llm_str}_s{args.shot}_t{args.temperature}_p*.png')))[0])
                elif args.llm:
                    self.syn_data_plots.append(os.path.join(self.root, f'{self.task}_{args.dataset}_{args.embedding}_num{args.num_samples}_seed{seed}_{args.llm}_s{args.shot}_t{args.temperature}_p{args.syn_prop}.png'))
        elif args.syn_prop_list:
            num_plots_per_prop = int(num_plots/len(args.syn_prop_list))
            for syn_prop in args.syn_prop_list:
                for seed in range(1, int(num_plots_per_prop/2)+1):
                    self.syn_data_plots.append(os.path.join(self.root, f'{self.task}_{args.dataset}_{args.embedding}_num{args.num_samples}_seed{seed}_{args.llm}_s{args.shot}_t{args.temperature}_p{syn_prop}.png'))
                print(f'{len(self.syn_data_plots)} Syn data plots with synthetic proportion {syn_prop}')
        else:
            exit()
    
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Grayscale(),  # Convert image to grayscale
                transforms.ToTensor(),   # Convert image to tensor
            ])
            
        self.data_plots = self.real_data_plots + self.syn_data_plots   
        self.labels = [0]*len(self.real_data_plots) + [1]*len(self.syn_data_plots)            
        
        print(f'Real data plot example: {self.data_plots[0]}\nSynthetic data plot example: {self.data_plots[-1]}')
    
    def __len__(self):
        return len(self.data_plots)
    
    def __getitem__(self, idx):
        plot_path = self.data_plots[idx]
        plot = Image.open(plot_path)
        
        if self.transform:
            plot = self.transform(plot)

        return plot, self.labels[idx]
    