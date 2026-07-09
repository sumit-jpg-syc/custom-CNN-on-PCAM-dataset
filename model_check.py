from image_classification import tumordetection
import torch 
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from metrics import metrics_cal , metrics_finder , plot # importing the metrics from the metrics file
# this file is to check the model and plot its graph 
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') # checking for cuda
model_0=tumordetection(3,1).to(device)
model_0.load_state_dict(torch.load('tumour.pth',map_location=device))
model_0.eval() # to put the model on evaluation mode to stop tehe neuron dropout
transform=transforms.Compose([transforms.Resize((96,96)),transforms.ToTensor()])
validation_set=datasets.PCAM(root='./data',
                        split='val',
                        download=True,
                        transform=transform)
test_set=datasets.PCAM(root='./data',
                        split='test',
                        download=True,
                        transform=transform)
validation_batches=DataLoader(dataset=validation_set,batch_size=64,shuffle=True,pin_memory=True)
test_batches=DataLoader(dataset=test_set,batch_size=64,shuffle=False)
print("----------------which data set do you wish to check---------------- ")
choice=input("enter which data stats do you wish to know: \n 1 for validation set \n 2 for test set ") #  to pich whic data set do you wish to check the model on
if choice == "1":
  result=metrics_cal(validation_batches,model_0,device) # to store all the data retured from the function into result
  y_pred_his=result['y_pred'] # to load the data stored by name 
  probs=result['probs']
  batch_num=result['batch']
  labels=result['labels']
  acc_his=result['accuracy']
  loss_his=result['loss']
  result_2=metrics_finder(y_pred_his,probs,labels)
  f1_score=result_2["f1_score"]
  precision_score=result_2["precision_score"]
  recall_score=result_2['recall_score']
  roc_acc_score=result_2['roc_acc_score']
  confusion_mat=result_2['confusion_mat']
  tn,fn,fp,tp=confusion_mat.ravel()# to find the true negative,false negative,false positive,true positive
  avg_loss=sum(loss_his)/len(loss_his)# to calculate avg loss across val dat set
  avg_acc=sum(acc_his)/len(acc_his) # to calculate avg accuracy across val dat set
  # to print out the final model test on the metrics
  print("===================================================FINAL MODEL STATS ON VALIDATION BATCH===================================================")
  print(f"average loss in validation data set: {avg_loss:.3f} \n average accuracy in validation data set {avg_acc:.3f}")
  print(f"f1 score for the set: {f1_score:.2f}")
  print(f"precision score for the set: {precision_score:.2f}")
  print(f"recall score for the set: {recall_score:.2f}")
  print(f"roc accuracy score for the set: {roc_acc_score:.2f}")
  print(f"true positive: {tp} | true negative: {tn} | false positive: {fp} | false negative: {fn}")
elif choice =='2':
  result=metrics_cal(test_batches,model_0,device)
  y_pred_his=result['y_pred']
  probs=result['probs']
  batch_num=result['batch']
  labels=result['labels']
  acc_his=result['accuracy']
  loss_his=result['loss']
  result_2=metrics_finder(y_pred_his,probs,labels)
  f1_score=result_2["f1_score"]
  precision_score=result_2["precision_score"]
  recall_score=result_2['recall_score']
  roc_acc_score=result_2['roc_acc_score']
  confusion_mat=result_2['confusion_mat']
  tn,fn,fp,tp=confusion_mat.ravel()
  avg_loss=sum(loss_his)/len(loss_his) # to calculate avg loss across val dat set
  avg_acc=sum(acc_his)/len(acc_his)# to calculate avg accuracy across val dat set
  # the model evaluation on test set on different metrics
  print("===================================================FINAL MODEL STATS ON TEST BATCH====================================================") 
  print(f"average loss in test data set: {avg_loss:.3f} \n average accuracy in test data set {avg_acc:.3f}")
  print(f"f1 score for the set: {f1_score:.2f}")
  print(f"precision score for the set: {precision_score:.2f}")
  print(f"recall score for the set: {recall_score:.2f}")
  print(f"roc accuracy score for the set: {roc_acc_score:.2f}")
  print(f"true positive: {tp} | true negative: {tn} | false positive: {fp} | false negative: {fn}")
else:
  print("enter valid choice")
  exit
# to plot the final ROC,accuracy,loss and confusion matrix graphs
print("=============================MODEL ACCURACY CURVE ON THE SET=============================")
plotting=plot(batch_num,acc_his,confusion_mat,probs,labels,loss_his)