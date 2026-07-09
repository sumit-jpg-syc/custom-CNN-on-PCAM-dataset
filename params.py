import torch
class param():
    # training parameters
    Epochs=3 # a higher epoch will only make model slow and cause overfitting
    lr=0.0005 # recomended optimizer Adam a higher lr won't allow model to generalize
    Weight_Decay=0.0001 # a higheer weight decay will cause model to underfit
    drop_out=0.3 # a higher will cause model to underfit and lower dropout will overfit

    # locations of data and model
    data_location='./data'
    model_save_location='tumour.pth'
    batch_size=64 # perefect balance of gradient learning and learning speed
        
    # device 
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__=="__main__":
    print("only to save and check parameters,location and device")