import torch
from DataLoading import DataHandler
from torchvision import datasets
from LeNet import LeNet
from trainer import Trainer



if __name__ == "__main__" :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_handler = DataHandler(datasets=datasets.MNIST,download=False, shuffle=True, batch_size=1) 
    training_dataset, validation_dataset = data_handler.dataset()
    _, validation_data = data_handler.DataLoader(training_dataset, validation_dataset)
    out_l1=120
    out_l2=64*2
    output_size=10
    model = LeNet(out_l1, out_l2, output_size)
    model.to(device=device) 
    checkpoint = torch.load("./checkpoints/checkpoint.cpkt")
    model.load_state_dict(checkpoint['model_state_dict'])
    trainer = Trainer(model=model)
    trainer.run(None, validation_data, inference=True, device=device)