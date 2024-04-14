import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class MNIST(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_path_list = os.listdir(data_dir)
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize((32, 32)),
                            transforms.Normalize((0.1307, ), (0.3081))
        ])

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        
        image_path = os.path.join(self.data_dir, self.image_path_list[idx])

        img = Image.open(image_path)
        img = self.transform(img)
        
        label = int(self.image_path_list[idx][-5])
        label = torch.tensor(label, dtype=torch.float32) 
        label = label.type(torch.LongTensor)
        
        return img, label


if __name__ == '__main__':
    pass

