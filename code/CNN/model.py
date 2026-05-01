import torch as Torch

class CNN(Torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Torch.nn.Sequential( #building layers
            Torch.nn.Conv2d(in_channels=1, out_channels=64,kernel_size=3, padding=1, bias=True), 
            Torch.nn.ReLU(),    
            Torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True), 
            Torch.nn.ReLU(),
            Torch.nn.MaxPool2d(kernel_size=2), # 28x28 -> 14x14
            
            Torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=True), 
            Torch.nn.ReLU(),
            Torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=True), 
            Torch.nn.ReLU(),
            Torch.nn.MaxPool2d(kernel_size=2), #14x14 -> 7x7

            Torch.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, bias=True), 
            Torch.nn.ReLU(),
            Torch.nn.Flatten(), 
            Torch.nn.Linear(64*7*7, 300), 
            Torch.nn.ReLU(),
            Torch.nn.Linear(300, 10)
        )

        

    def forward(self, x):
        return self.model(x)