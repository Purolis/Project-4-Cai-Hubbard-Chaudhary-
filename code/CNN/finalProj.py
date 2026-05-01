#refs: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html


import torch as Torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from model import CNN
from testOnOwnData import test_on_group_data


def main():
    #normalize our data to a bunch of -1 to +1 tensors



    #----------------------------------------load data/prepare minibatches------------------------------------


    #augmenting training data for better accuracy when training, attuning to all the weird handwriting stuff we do
    training_normalize_transforms = torchvision.transforms.Compose(
        [
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5), std=(0.5))
        ]
    )

    #don't augment the test data
    testing_normalize_transforms = torchvision.transforms.Compose(
        [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5), std=(0.5))
        ]
    )

    trainData = torchvision.datasets.MNIST(root='MNIST/train', train=True, transform=training_normalize_transforms, download=True)
    testData = torchvision.datasets.MNIST(root='MNIST/test',train=False, transform=testing_normalize_transforms,download=True)

    print("\n\nWe have loaded our datasets\n")

    #make batches of our data
    batchSize = 128
    trainDataLoader = Torch.utils.data.DataLoader(
        trainData, batch_size=batchSize, shuffle=True
    )
    testDataLoader = Torch.utils.data.DataLoader(
        testData, batch_size=batchSize
    )

  


    #----------------------------------------visualiz/analyze data------------------------------------
    #print("*****\nPREPARING MINIBATCHES")
    #prepare minibatches
    batch_size = 128

    train_loader = Torch.utils.data.DataLoader(
        trainData, batch_size=batch_size)
    test_loader = Torch.utils.data.DataLoader(testData, batch_size=batch_size)

    #visualization
    #print("******\nVISUALIZE DATA SET")
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    # plt.imshow(np.transpose(torchvision.utils.make_grid(
    #     images[:25], normalize=True, padding=1, nrow=5).numpy(), (1, 2, 0)))
    # plt.axis('off')
    # plt.show()



    #analyze dataset
    #print("*******\nANALYZE DATASET")
    classes = []
    for batch_idx, data in enumerate(train_loader):
        x, y = data
        classes.extend(y.tolist())

    unique, counts = np.unique(classes, return_counts=True)
    names = list(testData.class_to_idx.keys())
    # plt.bar(names, counts)
    # plt.xlabel("Target Classes")
    # plt.ylabel("Number of training instances")
    # plt.show()


    #--------------------------------device and model--------------------------------------
    device = 'cpu'
    if Torch.cuda.is_available():
        device='cuda'

    print(f"********DEVICE: {device}*********")
    model = CNN().to(device)


    #--------------------------------training------------------------------------------------
    num_epochs = 69
    learning_rate = 0.00069
    weight_decay = 0.000069
    step_size = 7
    gamma = 0.1
    endEarlyCount = 8

    criterion = Torch.nn.CrossEntropyLoss()
    optimizer = Torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True)
    scheduler = Torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) #lr*gamma every step_size epochs

    bestLoss = 100
    epochsNoIncrease = 0
    actualEpochs = 0


    #train
    print("\n\n*****************\nTRAINING STARTING\n****************")
    train_loss_list = []
    for epoch in range(num_epochs):
        actualEpochs += 1
        print(f'Epoch {epoch+1}/{num_epochs}:', end=' ')
        train_loss = 0
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss_list.append(train_loss / len(train_loader))
        
        print(f"Training loss = {train_loss_list[-1]}")
        scheduler.step()

        #set best loss, reset epochsNoIncrease if new best  \  else, increment
        if(train_loss_list[-1]) < bestLoss:
            bestLoss = train_loss_list[-1]
            epochsNoIncrease = 0
        else:
            epochsNoIncrease += 1

        #count how many epochs without a loss decrease. if too many, end training early
        if(epochsNoIncrease >= endEarlyCount):
            epoch = num_epochs
            break #hopefully not necessary bcuz of the last line

    print("\n\n*****************\nTRAINING DONE\n****************")
    print(f"Best loss: {bestLoss}")
    plt.plot(range(1, actualEpochs + 1), train_loss_list)
    plt.xlabel("Number of epochs")
    plt.ylabel("Training loss")
    plt.show()


    #--------------------------------------testing----------------------------------------
    print("****************\nTESTING STARTING\n******************")
    test_acc = 0
    model.eval()
    with Torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            y_true = labels.to(device)
            outputs = model(images)
            _, y_pred = Torch.max(outputs.data, 1)
            test_acc += (y_pred == y_true).sum().item()

    test_accuracy = 100 * test_acc / len(testData)
    print(f"Test set accuracy = {test_accuracy} %")
    num_images = 10
    y_true_name = [names[y_true[idx]] for idx in range(num_images)]
    y_pred_name = [names[y_pred[idx]] for idx in range(num_images)]
    title = f"Actual labels: {y_true_name}, Predicted labels: {y_pred_name}"

    # plt.imshow(np.transpose(torchvision.utils.make_grid(
    # images[:num_images].cpu(), normalize=True, padding=1).numpy(), (1, 2, 0)))
    # plt.title(title)
    # plt.axis("off")
    # plt.show()


    if(test_accuracy >= 98.0):
        cont_test = continue_ask("Test on group data? (Y/N): ")
        if(cont_test):
            test_on_group_data(model=model)

        save_ask = continue_ask("Save model? (Y/N): ")
        if(save_ask):
            file_name = input("What is the name of the file? (.pt will be appended): ")
            save_model(file_name, model)



def continue_ask(prompt):
    valid = False
    while(not valid):
        to_continue = input(prompt)
        if(to_continue.upper() == "Y"):
            valid = True
            return True
        elif(to_continue.upper() == "N"):
            valid = True
            return False
        
    
def save_model(save_file_name, model):
    #save the model
    Torch.save(model.state_dict(), f"./models/{save_file_name}.pt")



if __name__ == "__main__":
    main()










