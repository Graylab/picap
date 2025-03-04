#init(" ".join(options.split('\n')))
import os
import numpy as np
import pandas as pd
from utils import *
from utils_model import *
from egnn.egnn import *
import torch
import matplotlib.pyplot as plt

import os


##Dataset files - small molecule
TRAIN_CLUSTER = '../dataset/train_clust-sm.csv'
TRAIN_PDB = '../dataset/train_pdb-sm.csv'
TEST_CLUSTER = '../dataset/val_clust-sm.csv'
TEST_PDB = '../dataset/val_pdb-sm.csv'

BATCH_SIZE = 1; #Must be 1 due to Torch reading in diff sizes
FAKE_BATCH_SIZE = 1; #Ad hoc, not used
NUM_WORKERS = 0;
NUM_EPOCHS = 1000;

LOSS_FN = dice_loss #loss function defined below
KNN = [16,16,16,16] #KNN per block
N_LAYERS = [3,3,3,3] #num layers per block
HIDDEN_NF = 128 #Hidden num features - embedding size of EGNN
PAD_SIZE = 2500 #not used
LEARNING_RATE = 2e-6
W_DECAY = 1e-7
NORMALIZE = False #whether EGNN normalization flag is used
TOL_EPOCH = 35 #early stopping

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    NUM_WORKERS = 8;
print("Using: " + DEVICE)
DEVICE = torch.device(DEVICE)
print(DEVICE)

#load the loaders and model
train_loader, val_loader = get_loaders(TRAIN_CLUSTER,TRAIN_PDB,TEST_CLUSTER,TEST_PDB,root_dir="../",
                                            batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,
                                            use_pad=False,pad_size=PAD_SIZE)

model = CAPSIF2_RES2(hidden_nf = HIDDEN_NF, n_layers=N_LAYERS,normalize=NORMALIZE,
                device=DEVICE).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=W_DECAY)
scaler = torch.cuda.amp.GradScaler()
torch.autograd.set_detect_anomaly(True)
model.train()
print("Model loaded")


# From: https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
## takes in a module and applies the specified weight initialization
def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
        values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
    # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,0.02)
    # m.bias.data should be 0
        #print(type(m.bias))
        if type(m.bias) != type(None):
            m.bias.data.fill_(0)

model.apply(weights_init_normal)
print("Model initialized")

my_model_name = model.get_string_name() + "_knn" + str(KNN[0]) + '-' + str(KNN[1])

train_loss = [];
val_loss = [];
epochs = [];
print("EPOCH,TRAIN,VAL,VAL_CLUST")

print()

info =  "BATCH_SIZE: " + str(BATCH_SIZE) + "\nKNN: " + str(KNN) + "\n N_LAYERS: " + str(N_LAYERS) + '\n'
info += "HIDDEN_NF: " + str(HIDDEN_NF) + "\nLR: " + str(LEARNING_RATE) + "\nWeight_decay: " + str(W_DECAY) + "\nTRAIN: Pre"

best_epoch = 0
print(info)

if __name__ == "__main__":
    for epoch in range(NUM_EPOCHS):
        #print(epoch)
        model.train()
        step_loss = model_train_sm(train_loader, model, optimizer, LOSS_FN, scaler, DEVICE=DEVICE, fake_batch_size=FAKE_BATCH_SIZE)

        torch.save({'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'epoch':epoch, 'info':info },"./models_DL/model-" + my_model_name + "_last.pt")


        model.eval()
        v_loss = model_val_sm(val_loader, model, optimizer, LOSS_FN, scaler, DEVICE=DEVICE)
        val_loss.append(v_loss)

        print(epoch,step_loss,v_loss,np.min(val_loss));

        if (len(val_loss) < 1):
            torch.save({'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss':v_loss, 'epoch':epoch, 'info':info  },"./models_DL/model-" + my_model_name + ".pt")
        elif (v_loss <= np.min(val_loss)):
            #print('save_torch')
            best_epoch = epoch
            torch.save({'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss':v_loss, 'epoch':epoch, 'info':info  },"./models_DL/model-" + my_model_name + ".pt")

        epochs.append(epoch)
        train_loss.append(step_loss)

        file = "./models_DL/trainData-" + my_model_name + ".npz"
        #print(file)
        np.savez(file,epochs=np.array(epochs), train_loss = np.array(train_loss), val_loss = np.array(val_loss) )

        #Early stopping
        if best_epoch < epoch - TOL_EPOCH:
            exit()

print('FIN');
