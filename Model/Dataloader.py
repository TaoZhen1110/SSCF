from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch


class Data_Loader(Dataset):
    def __init__(self, jsondata):
        super(Data_Loader, self).__init__()
        self.Grammar_Features = []
        self.Semantic_Features = []
        self.labels = []  # 存储句子对应的标签

        for data in jsondata:
            for key, value in data.items():
                if "LLM_Review" in key:
                    self.Grammar_Features.append(value['Grammar_Features'])
                    self.Semantic_Features.append(value['Semantic_list'])
                    self.labels.append(0)
                elif "Human_Review" in key:
                    self.Grammar_Features.append(value['Grammar_Features'])
                    self.Semantic_Features.append(value['Semantic_list'])
                    self.labels.append(1)
    
    def __len__(self):
        return len(self.Grammar_Features)
    
    def __getitem__(self, idx):
        Grammar_Feature = torch.tensor(self.Grammar_Features[idx], dtype=torch.float32)
        Semantic_Feature = torch.tensor(self.Semantic_Features[idx], dtype=torch.float32)  # 转换为tensor
        Label = torch.tensor(self.labels[idx], dtype=torch.long)  # 标签转换为tensor
        
        return {
            "Grammar_Feature": Grammar_Feature,
            "Semantic_Feature": Semantic_Feature,
            "Label": Label
        }


def dataset_loader(dataset, batch_size, shuffle=False):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader
