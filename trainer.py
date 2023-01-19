import torch 
from torch import nn, optim
from d2l import torch as d2l

class Trainer(nn.Module) :
    
    def __init__(self, model, epochs=2, learning_rate=0.01, momentum=0.9) -> None:
        super().__init__()
        self.epochs        = epochs
        self.learning_rate = learning_rate
        self.momentum      = momentum
        self.optimizer     = optim.SGD(model.parameters(), lr=self.learning_rate,momentum=self.momentum)
        self.loss_fn       = nn.CrossEntropyLoss()
        self.model         = model
    
    def train(self, data,device='cpu') :
        self.model.train()
        size = len(data.dataset)
        for batch, (X, y) in enumerate(data) :
            self.optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = self.model(X) # prediction
            self.loss = self.loss_fn(y_hat, y) # Loss
            self.loss.backward() # back prop
            self.optimizer.step() # update
            if batch % 50 == 0 :
                loss, current = self.loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    def test(self, data, device='cpu') :
        precision, loss, num_batches, size = 0, 0, len(data), len(data.dataset)
        self.model.eval()
        n = 10
        with torch.no_grad() :
            for X, y in data :
                X, y = X.to(device), y.to(device)
                y_hat = self.model(X) #
                loss += self.loss_fn(y_hat, y).item()     
                y = y.reshape(-1,1)[:,:]  
                #y_hat =y_hat[0:3,:]
                #print("pred :",y_hat[n:n+10,:].argmax(dim=1).reshape(-1,1),"\ngt :", y[n:n+10,:])
                precision += (y_hat[:,:].argmax(dim=1).reshape(-1,1) == y[:,:]).sum(axis=0).item()

        loss      /= num_batches
        precision /= size
        print(f"Test Error: \n Accuracy: {(100*precision):>0.5f}%, Avg loss: {loss:>8f} \n")
        return loss, precision
    
    def save(self, epoch, loss, precision, path) :
        
        torch.save({'epoch': epoch,
                    'precision': precision,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,}, path)
    
    def draw(self,data) :
        import matplotlib.pyplot as plt
        labels_map = {
                        0: "0",
                        1: "1",
                        2: "2",
                        3: "3",
                        4: "4",
                        5: "5",
                        6: "6",
                        7: "7",
                        8: "8",
                        9: "9",
                    }
        
        fig = plt.figure(figsize=(8, 8))
        cols, rows = 1, 1
        for i in range(1, cols * rows + 1):
            img, label = data
            fig.add_subplot(rows, cols, i)
            plt.title(labels_map[label.item()])
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap="turbo")
        plt.show()
        
    def inference(self, data, n=1, device='cpu') :
        
        classes =  ["0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9"]

        self.model.eval()
        
        with torch.no_grad():
            for X, y in data :
                #print(self.model.conv_layer_1[0:1](X.to(device)))
                #exit()
                X, y = X.to(device), y.to(device)
                pred = self.model(X)
                n = 0
                y = y.reshape(-1,1)
                #print("pred :",pred[:,:].argmax(dim=1).reshape(-1,1).item(),"\ngt :", y[:,:].item())
                predicted, actual = classes[pred[:,:].argmax(dim=1).reshape(-1,1).item()], classes[y[:,:].item()]
                print(f'Pred,icted: "{predicted}", Actual: "{actual}"')
                self.draw((X.detach().cpu().numpy(),y.detach().cpu().numpy()))
    
    def run(self,train_data, test_data, inference=False,device='cpu') :
        if inference and train_data==None:
            self.inference(test_data, device=device)
            return
        
        accuracy = 0 
        for epoch in range(self.epochs) :
            print(f"epoch : {epoch+1}")
            self.train(train_data,device=device)
            loss, precision = self.test(test_data, device=device)
            path = "./checkpoints/checkpoint.cpkt"
            if precision > accuracy :
                self.save(epoch=epoch, loss=loss, precision=precision, path=path)
                accuracy = precision
        print('best accuracy :', accuracy)