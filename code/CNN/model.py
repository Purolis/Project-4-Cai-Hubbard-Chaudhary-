import torch as Torch

class CNN(Torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Torch.nn.Sequential( #building layers
            Torch.nn.Conv2d(in_channels=1, out_channels=128,kernel_size=3, padding=1), #1 input channels bcus greyscale channels for image, kernel size of 3 is a 3x3 filter
            Torch.nn.ReLU(),
            Torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), 
            Torch.nn.ReLU(),
            Torch.nn.MaxPool2d(kernel_size=2),
            Torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), 
            Torch.nn.ReLU(),
            Torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1), 
            Torch.nn.ReLU(),
            Torch.nn.MaxPool2d(kernel_size=2),
            Torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1), 
            Torch.nn.ReLU(),
            Torch.nn.Flatten(), #flatten layers now that we're done
            Torch.nn.Linear(64*7*7, 300), #fully connected flatten, 
            Torch.nn.ReLU(),
            Torch.nn.Linear(300, 10)
            
        )

    def forward(self, x):
        return self.model(x)