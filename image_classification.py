import torch
from torchvision import transforms
from params import param
from torchvision import datasets
from torch import nn
from params import param
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm.auto import tqdm
from timeit import default_timer as timer 
# the file to train and save the original model
print(f"------Current Hyperparameters------ \n epoch: {param.Epochs} \n learning rate: {param.lr} \n weight decay: {param.Weight_Decay} \n drop out: {param.drop_out} ")
print(f"------Locations------ \n location of Data: {param.data_location} \n model location: {param.model_save_location} \n batch size: {param.batch_size}")
print(f"------Device------ \n device: {param.device}")
device=param.device
transform=transforms.Compose([transforms.Resize((96,96)),transforms.ToTensor()]) # transform the png data into tensor
train_set=datasets.PCAM(root=param.data_location,
                        split='train',
                        download=True,
                        transform=transform) # 90,5,5 train,validate,test dataset 
validation_set=datasets.PCAM(root=param.data_location,
                        split='val',
                        download=True,
                        transform=transform)
test_set=datasets.PCAM(root=param.data_location,
                        split='test',
                        download=True,
                        transform=transform)
def time(start,end): #to check calculation time
   time_taken=end-start
   return time_taken
def plot(): # to visualize first few images
    for i in range(10):
        n=torch.randint(0,len(train_set),size=[1]).item()
        image,labels=train_set[n]
        for_plot=image.permute(1,2,0)
        plt.figure(figsize=(12,6))
        plt.imshow(for_plot)
        plt.title(f"(0 is for beingn | 1 for tumour) label: {labels}")
        plt.show()
class tumordetection(nn.Module): # main model parameters
      def __init__(self,input_feature,out_feature):
       super().__init__()
       self.cnn_layer_1=nn.Conv2d(in_channels=input_feature,out_channels=32,stride=1,kernel_size=5)
       self.pool_1=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
       self.batch_1=nn.BatchNorm2d(32) # batchnorm to minimize weight from reaching infinity
       self.relu=nn.ReLU() # relu helps model identify non-linearity
       self.cnn_layer_2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1)
       self.batch_2=nn.BatchNorm2d(64)
       self.pool_2=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
       self.droput=nn.Dropout(param.drop_out) # dropping out 30% neuron to stop overfitting 
       self.cnn_layer_3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1)
       self.batch_3=nn.BatchNorm2d(128)
       self.pool_3=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
       self.avgpool=nn.AdaptiveAvgPool2d(1) #to remove the extra dimension 
       self.flatten=nn.Flatten()
       self.layer_1=nn.Linear(128,64) #256*1*1
       self.layer_2=nn.Linear(64,32) # larger neural net to check for small patterns
       self.final=nn.Linear(32,out_feature)
       self.sigmoid=nn.Sigmoid() # to convert logits between 0 & 1
      def forward(self,x):
         x=self.relu(self.batch_1(self.cnn_layer_1(x)))
         x=self.pool_1(x)
         x=self.droput(x)
         x=self.relu(self.batch_2(self.cnn_layer_2(x)))
         x=self.pool_2(x)
         x=self.droput(x)
         x=self.relu(self.batch_3(self.cnn_layer_3(x)))
         x=self.pool_3(x)
         x=self.avgpool(x)
         x=self.flatten(x)
         x=self.layer_1(x)
         x=self.final(self.layer_2(x))
         x=self.sigmoid(x)
         return x
if __name__=="__main__":
   print("do you wish to visualize the data")
   print("enter: \n 1 for yes \n 2 for no  ")
   choice=input("enter your choice: ") # to check the data set if the data is not corrupted
   if choice== "1":
       plo=plot()
   elif choice == "2":
      print("--------loading the model--------")
   else :
      print("enter valid choice")
      exit()
   train_batches=DataLoader(dataset=train_set,batch_size=param.batch_size,shuffle=True,pin_memory=True)  # to shuffle and create batches of 64
   validation_batches=DataLoader(dataset=validation_set,batch_size=param.batch_size,shuffle=True,pin_memory=True)
   test_batches=DataLoader(dataset=test_set,batch_size=param.batch_size)
   print(f"number off batches of batch_size: {param.batch_size} \n Train set: {len(train_batches)} \n Test set:  {len(test_batches)}") # to check the total number of batches
   train_batches_features,train_batches_labels=next(iter(train_batches))
   print(train_batches_features.shape) # to check if the data is of appropiate shape
   c=train_batches_features.shape[1]
   model_1=tumordetection(c,1) # here 1 is due to binary classification
   with torch.inference_mode():
      predictions=model_1(train_batches_features).squeeze()
   accuracy=Accuracy(task='binary',threshold=0.5) # to calculate the accuracy of the binary data
   acc=accuracy(predictions,train_batches_labels)
   print(acc)
   end=timer()
   lossfn=nn.BCELoss() # BCE and BCEwithlogits are two loss function BCE worked better for me
   optimizer=torch.optim.Adam(params=model_1.parameters(),lr=param.lr,weight_decay=param.Weight_Decay)# adam and SGD are two optimizer adam works with lower number of epochs while sgd is better for finr tunning
   model_1=model_1.to(device) #shiffiting model to cuda
   epochs=param.Epochs
   for epoch in tqdm(range(epochs)):
      epoch_start=timer()
      accuracy=Accuracy(task='binary',threshold=0.5).to(device)  # defining accuracy every epoch to avoid device errors
      for batch,(x,y) in enumerate(train_batches):
         start_time=timer()
         model_1.train()
         x,y=x.to(device),y.float().to(device)# putting the data on cuda 
         y_pred=model_1(x)
         y_pred=y_pred.float().squeeze()# squeeze to remove the extra dimension
         y_pred=y_pred.to(device)
         loss=lossfn(y_pred,y).to(device) # calculating the loss
         optimizer.zero_grad()# reset the old optimizer 
         loss.backward()# calculates the gradient loss
         optimizer.step()# takes the step
         if batch%256==0:
            acc=accuracy(y_pred,y)
            end_time=timer()
            batch_time=time(start_time,end_time)# to calculate the time and see if gpu is overheating or any other issue every 64th natch
            print(f"epoch: {epoch} | batch number: {batch} | loss: {loss:.3f} | accuracy {acc:.3f} | time for 1 batch: {batch_time:.3f}")
      model_1.eval()
      with torch.inference_mode():# to check for over fitting on validation set
         a,b=next(iter(validation_batches))
         a=a.to(device)
         b=torch.tensor(b)
         b=b.float().to(device)
         b_val_pred=model_1(a).squeeze().to(device) #to check for data leak
         loss_val=lossfn(b_val_pred,b)
         accu=accuracy(b_val_pred,b)
         epoch_end=timer()
         epoch_time=time(epoch_start,epoch_end)
         print('========MODEL CHECK========')
         print(f"Validation data set accuray: {accu:.3f} | validation data set loss: {loss_val:.3f} | epoch time: {epoch_time:.3f}") 
   test_set,test_labels=next(iter(test_batches)) # to check accuracy on one test set 
   test_set=test_set.to(device)
   test_labels=torch.tensor(test_labels)
   test_labels=test_labels.float().to(device)
   accuracy=Accuracy('binary',0.5).to(device)
   with torch.inference_mode():
    test_pred=model_1(test_set).squeeze()
   loss=lossfn(test_pred,test_labels)
   acc=accuracy(test_pred,test_labels)
   print(f"test set accuracy {acc:.3f} | test loss {loss:.3f}")
   # to save and check model parameters
   torch.save(model_1.state_dict(),param.model_save_location)
   model_load=tumordetection(3,1)
   model_load.load_state_dict(torch.load(param.model_save_location))
   print(model_load)