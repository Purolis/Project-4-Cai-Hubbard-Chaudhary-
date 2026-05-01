import os
from PIL import Image
from torchvision import transforms
from model import CNN
import torch as Torch

def test_on_group_data(model=None):
    image_dir = './img'
    model_name = "submission.pt" #change this for testing an individual model

    total = 0
    correct = 0

    #init model
    device = 'cpu'
    if Torch.cuda.is_available():
        device='cuda'

    print(f"********DEVICE: {device}*********")

    if(model == None):
        model = CNN().to(device)

        #load model
        model.load_state_dict(
            Torch.load(f"./models/{model_name}", map_location="cpu")
        )
        model.eval()

    #same transforms as when prepping data in main file with some extra steps
    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


    device = next(model.parameters()).device

    #predict
    for filename in os.listdir(image_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = f"{image_dir}\{filename}"
            
            #load images
            img = Image.open(img_path)
            img_tensor = test_transforms(img)
            img_tensor = img_tensor.to(device)
            img_tensor = img_tensor.unsqueeze(0) 
            
            #actual prediction part
            with Torch.no_grad():
                output = model(img_tensor)
                prediction = Torch.argmax(output, dim=1).item()
                
            print(f"File: {filename} | Predicted Digit: {prediction}")
            
            #check if correct
            total += 1
            if(str(prediction) == str(filename[0])):
                correct += 1
            

    #print acc
    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy}%")


if __name__ == "__main__":
    test_on_group_data()