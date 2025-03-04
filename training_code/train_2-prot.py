#PICAP training
#init(" ".join(options.split('\n')))
import os
import numpy as np
import pandas as pd
from utils import *
from utils_model import *
from egnn.egnn import *
import torch
import matplotlib.pyplot as plt
from pathlib import Path

import os

TRAIN_CLUSTER = '../dataset/final0_train_cluster_prune.csv'
TRAIN_PDB = '../dataset/final0_train_pdb.csv'

TEST_CLUSTER = '../dataset/final0_val_cluster_prune.csv'
TEST_PDB = '../dataset/final0_val_pdb.csv'

BATCH_SIZE = 1;
NUM_WORKERS = 0;
FAKE_BATCH_SIZE = 1;
NUM_EPOCHS = 1000;

loss_str = ""

#Hyper parameters!
loss_str = ""
LOSS_FN = nn.BCELoss()
KNN = [10,20,40,60]
N_LAYERS = [3,3,3,3]
HIDDEN_NF = 128
PAD_SIZE = 2500
LEARNING_RATE = 2e-5
W_DECAY = 1e-6
ADAPOOL_SIZE = (150,HIDDEN_NF) #Size of the MAX adaptive pool


DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    NUM_WORKERS = 8;
print("Using: " + DEVICE)
DEVICE = torch.device(DEVICE)
print(DEVICE)


info =  "PROT_TRAIN\nBATCH_SIZE: " + str(BATCH_SIZE) + "\nLOSS: bce_loss\nKNN: " + str(KNN) + "\n N_LAYERS: " + str(N_LAYERS) + '\n'
info += "HIDDEN_NF: " + str(HIDDEN_NF) + "\nADAPOOL_SIZE: " + str(ADAPOOL_SIZE) + "\nLR: " + str(LEARNING_RATE) + "\nWeight_decay: " + str(W_DECAY) + "\nTRAIN: SM->Final"

train_loader, val_loader = get_loaders(TRAIN_CLUSTER,TRAIN_PDB,TEST_CLUSTER,TEST_PDB,root_dir="../",
                                            batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,
                                            use_pad=False)

model = CAPSIF2_PROT(hidden_nf = HIDDEN_NF, n_layers=N_LAYERS,
                device=DEVICE).to(DEVICE)

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
        m.weight.data.normal_(0.0,1/np.sqrt(y))
    # m.bias.data should be 0
        #print(type(m.bias))
        if type(m.bias) != type(None):
            m.bias.data.fill_(0)

model.apply(weights_init_normal)

torch.autograd.set_detect_anomaly(True)

if DEVICE == 'cuda':
    checkpoint = torch.load("./models_DL/model-" + my_model_name + ".pt")
else:
    checkpoint = torch.load("./models_DL/model-" + my_model_name + ".pt",
        map_location=torch.device('cpu') )
model_pre.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


#load the same weights
model.embedding_in0.load_state_dict(model_pre.embedding_in0.state_dict())
model.graph_module.load_state_dict(model_pre.graph_module.state_dict())
model.graph_norm.load_state_dict(model_pre.graph_norm.state_dict())
model.graph_drop.load_state_dict(model_pre.graph_drop.state_dict())

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=W_DECAY)
scaler = torch.cuda.amp.GradScaler()


model.train()
print("Model loaded")


my_model_name = model.get_string_name() + "_knn" + str(KNN[0]) + '-' + str(KNN[1]) + "_sm-prot"

my_model_name += loss_str
my_model_name += '_fin'


my_int = 0;
new_name = my_model_name + "_" + str(my_int)

while os.path.exists('./models_DL/model-' + new_name + '.pt'):
    my_int += 1
    new_name = my_model_name + "_" + str(my_int)

my_model_name = new_name

Path("./models_DL/model-" + my_model_name + ".pt").touch()

train_loss = [];
val_loss = [];
epochs = [];
print(my_model_name)
print(info)

print("EPOCH,TRAIN,VAL,VAL_PROT,VAL_RES,BEST_VAL")

if __name__ == "__main__":
    for epoch in range(NUM_EPOCHS):
        #print(epoch)
        model.train()

        step_loss = model_train_two_prot(train_loader, model, optimizer, LOSS_FN, scaler,
                DEVICE=DEVICE, epoch=epoch, MODIFIER=.9)
        print(step_loss,'val now')
        model.eval()

        val_return, cm = model_val_two_prot(val_loader, model, optimizer, LOSS_FN, scaler, DEVICE=DEVICE)

        v_loss = val_return

        if (len(val_loss) < 1):
            print('save_first')
            torch.save({'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss':v_loss,'epoch':epoch, 'info':info },"./models_DL/model-" + my_model_name + ".pt")
        elif (v_loss < np.min(val_loss)):
            print('save_torch')
            torch.save({'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss':v_loss,'epoch':epoch, 'info':info },"./models_DL/model-" + my_model_name + ".pt")

        epochs.append(epoch)
        val_loss.append(v_loss)
        train_loss.append(step_loss)

        print(epoch,step_loss,v_loss,np.min(val_loss), cm);

        file = "./models_DL/trainData-" + my_model_name + ".pt"
        #print(file)
        np.savez(file,epochs=np.array(epochs), train_loss = np.array(train_loss), val_loss = np.array(val_loss) )



print('FIN');
