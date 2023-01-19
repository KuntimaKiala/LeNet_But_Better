import torch
from DataLoading import DataHandler
from torchvision import datasets
from vanilla import SoftMax
from trainer import Trainer



if __name__ == "__main__" :
    device = torch.device('cuda0' if torch.cuda.is_available() else 'cpu')
    data_handler = DataHandler(datasets=datasets.FashionMNIST,download=False, shuffle=True, batch_size=1) 
    training_dataset, validation_dataset = data_handler.dataset()
    _, validation_data = data_handler.DataLoader(training_dataset, validation_dataset)
    input_size = next(iter(validation_data))[0].shape[2] * next(iter(validation_data))[0].shape[3]
    hidden_size = 1
    output_size = 10
    model = SoftMax(input_size, hidden_size, output_size)    
    checkpoint = torch.load("./checkpoints/checkpoint.cpkt")
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    trainer = Trainer(model=model)
    trainer.run(None, validation_data, inference=True)