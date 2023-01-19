from torch.utils.data import DataLoader
import torchvision

class DataHandler() :
    def __init__(self, datasets, shuffle=True, download=False, batch_size = 32) :
        self.shuffle    = shuffle
        self.download   = download
        self.batch_size = batch_size
        self.datasets = datasets
        
    def dataset(self) :
        self.training_dataset   = self.datasets(root="./data", 
                                                train=True,   
                                                download=self.download,
                                                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
        self.validation_dataset = self.datasets(root="./data", 
                                                train=False,  
                                                download=self.download,
                                                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
        
        return self.training_dataset, self.validation_dataset
    
    def DataLoader(self, training_dataset, validation_dataset) :
        self.training_data   = DataLoader(training_dataset,   batch_size=self.batch_size,shuffle=self.shuffle)
        self.validation_data = DataLoader(validation_dataset, batch_size=self.batch_size,shuffle=self.shuffle)
        return self.training_data, self.validation_data