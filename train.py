import torch
from DataLoading import DataHandler
from torchvision import datasets
from LeNet import LeNet
from trainer import Trainer

if __name__ == '__main__' :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_handler = DataHandler(datasets=datasets.MNIST,download=True, shuffle=True, batch_size=256) 
    training_dataset, validation_dataset = data_handler.dataset()
    training_data, validation_data = data_handler.DataLoader(training_dataset, validation_dataset)
    out_l1=120
    out_l2=64*2
    output_size=10
    model = LeNet(out_l1, out_l2, output_size)
    model.to(device=device)
    trainer = Trainer(model=model, epochs=10, learning_rate=0.89, momentum=0.0)
    trainer.run(training_data, validation_data, device=device)
    