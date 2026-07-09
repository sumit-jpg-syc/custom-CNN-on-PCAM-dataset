import torch 
from sklearn.metrics import f1_score,recall_score,precision_score,confusion_matrix,roc_auc_score,roc_curve,auc
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
# this file is only for metrics to check the model on
lossfn=nn.BCELoss()
loss_his=[]
batch_num=[]
y_his=[]
y_prob_his=[]
y_pred_his=[]
acc_his=[]
# to calculate all the requirments for metric calculation,accuracy and for the plots
def metrics_cal(data,model,device):
    for batch,(x,y) in enumerate(data):
        x=x.to(device)
        y=y.float().to(device)
        model.eval()
        with torch.inference_mode():
            y_prob=model(x).squeeze().to(device)
        loss=lossfn(y_prob,y).item()
        batch_num.append(batch)
        y_pred=(y_prob>.5).float()# to convert probabilities into labels [0.9,0.3,0.7]-->[1.0,0,1.0]
        acc=(y_pred==y).float().mean().item()
        y_his.append(y) # to save all the history of the data loss,pred,accuracy
        loss_his.append(loss)
        y_prob_his.append(y_prob)
        y_pred_his.append(y_pred)
        acc_his.append(acc)
    return {"y_pred" :y_pred_his, "probs" :y_prob_his, "batch": batch_num, "labels": y_his, "accuracy": acc_his, "loss": loss_his}
#different metrics calculations for the model
def metrics_finder(y_pred,y_prob,y):
    y_pred=torch.cat(y_pred).cpu().numpy()# here cat joins all the batch data and numpy convert tensor into numpy  
    y_prob=torch.cat(y_prob).cpu().numpy()# cpu() relocates the data into cpu for sklearn metrics 
    y=torch.cat(y).cpu().numpy()
    # the parameters are to calculate f1 score, precision,recall,roc_score this gives a better understanding of model across all the metric
    f1=f1_score(y_pred,y)# f1 score gives a better understandig for imbalanced classes 
    precision=precision_score(y_pred,y) # to check how many of the predicted tumour were actually tumour
    recall=recall_score(y_pred,y) # to check out of how may +ve cases did the model catch 
    roc_score=roc_auc_score(y,y_prob) # to check model on different thresholds
    confusion=confusion_matrix(y_pred,y) #  to compare true positive, true negative,false positive and false negative
    return {"f1_score": f1, "precision_score": precision, "recall_score": recall, "roc_acc_score": roc_score,"confusion_mat": confusion}
def plot(batch,acc_his,confusion_mat,y_prob,y,loss_his):
    y_prob=torch.cat(y_prob).cpu().numpy()
    y=torch.cat(y).cpu().numpy()
    fpr,tpr,threshold=roc_curve(y,y_prob)# to calculate false positive rate and true positive rate 
    roc_auc=auc(fpr,tpr)
    plt.figure(figsize=(16,8))

    plt.subplot(2,2,1)
    # to plot the ROC curve of the model
    plt.plot(fpr,tpr,color='green',label=f'ROC curve AUC :{roc_auc:.2f}')
    plt.plot([0,1],[0,1],color='red',label='random guess')
    plt.xlabel('false positives (FPR)')
    plt.ylabel('true positive rate (TPR)')
    plt.title('ROC curve')

    plt.subplot(2,2,2)
    # to plot the accuracy over the batches
    plt.plot(batch,acc_his,alpha=.8)
    plt.axhline(.5,color='red',linestyle='--',label="random guess")
    plt.axhline(sum(acc_his)/len(acc_his),color='green',linestyle='--',label="accuracy")
    plt.xlabel("batch number")
    plt.ylabel("accuracy")
    plt.title("accuracy across the batches")

    plt.subplot(2,2,3)
    # to plot the loss across the batches
    plt.plot(batch_num,loss_his,alpha=.8)
    plt.axhline(sum(loss_his)/len(loss_his),color='green',linestyle='--',label="loss")
    plt.xlabel("batch number")
    plt.ylabel("loss")
    plt.title("loss across the batches")

    plt.subplot(2,2,4)
    # to plot the confusion matrix
    sns.heatmap(confusion_mat,fmt='d',cmap='Blues')
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.title('confusion matrix')
    plt.show()
if __name__=="__main__":
    print("only for model check")