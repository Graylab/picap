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

TRAIN_CLUSTER = '../dataset/final3_train_glylig_clust.csv'
TRAIN_PDB = '../dataset/final0_train_glylig_pdb.csv'
TEST_CLUSTER = '../dataset/final0_val_glylig_clust.csv'
TEST_PDB = '../dataset/final0_val_glylig_pdb.csv'

BATCH_SIZE = 1;
NUM_WORKERS = 0;
NUM_EPOCHS = 1000;
FAKE_BATCH_SIZE = 1;
FREEZE_BASE = False


#Hyper parameters!
LOSS_FN = dice_loss
loss_str = ''
KNN = [16,16,16,16]
N_LAYERS = [3,3,3,3]
HIDDEN_NF = 128
PAD_SIZE = 2500
LEARNING_RATE = 2e-5
W_DECAY = 1e-6
TOL_EPOCH = 25
NORMALIZE = False

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    NUM_WORKERS = 8;
print("Using: " + DEVICE)
DEVICE = torch.device(DEVICE)
print(DEVICE)


info =  "PROT_TRAIN\nBATCH_SIZE: " + str(BATCH_SIZE) + "\nLOSS: dice_loss\nKNN: " + str(KNN) + "\n N_LAYERS: " + str(N_LAYERS) + '\n'
info += "HIDDEN_NF: " + str(HIDDEN_NF)  + "\nLR: " + str(LEARNING_RATE) + "\nWeight_decay: " + str(W_DECAY) + "\nTRAIN: SM->Final"
print(info)

#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

train_loader, val_loader = get_loaders(TRAIN_CLUSTER,TRAIN_PDB,TEST_CLUSTER,TEST_PDB,root_dir="../",
                                            batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,
                                            use_pad=False)


model = CAPSIF2_RES2(hidden_nf = HIDDEN_NF, n_layers=N_LAYERS,
                device=DEVICE,normalize=NORMALIZE).to(DEVICE)
my_model_name = model.get_string_name() + "_knn" + str(KNN[0]) + '-' + str(KNN[1])

torch.autograd.set_detect_anomaly(True)

if DEVICE == 'cuda':
    checkpoint = torch.load("./models_DL/model-" + my_model_name + ".pt")
else:
    checkpoint = torch.load("./models_DL/model-" + my_model_name + ".pt",
        map_location=torch.device('cpu') )
model.load_state_dict(checkpoint['model_state_dict'])



optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=W_DECAY)
scaler = torch.cuda.amp.GradScaler()
model.train()
print("Model loaded")

my_model_name = model.get_string_name() + "_knn" + str(KNN[0]) + '-' + str(KNN[1]) + loss_str + "_norm" + str(NORMALIZE) + "_"
my_model_name += "_carb_fin"

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
best_epoch = -1

print("EPOCH,TRAIN,VAL,VAL_PROT,VAL_RES,BEST_VAL")

if __name__ == "__main__":
    for epoch in range(NUM_EPOCHS):
        #print(epoch)
        model.train()
        step_loss = model_train_sm(train_loader, model, optimizer, LOSS_FN, scaler, DEVICE=DEVICE, fake_batch_size=FAKE_BATCH_SIZE)

        model.eval()
        v_loss = model_val_sm(val_loader, model, optimizer, LOSS_FN, scaler, DEVICE=DEVICE)

        if (len(val_loss) < 1):
            print('save_first')
            torch.save({'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss':v_loss,'epoch':epoch, 'info':info },"./models_DL/model-" + my_model_name + ".pt")
        elif (v_loss < np.min(val_loss)):
            best_epoch = epoch
            print('save_torch')
            torch.save({'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss':v_loss,'epoch':epoch, 'info':info },"./models_DL/model-" + my_model_name + ".pt")

        epochs.append(epoch)
        val_loss.append(v_loss)
        train_loss.append(step_loss)

        print(epoch,step_loss,v_loss,np.min(val_loss));

        file = "./models_DL/trainData-" + my_model_name + ".pt"
        #print(file)
        np.savez(file,epochs=np.array(epochs), train_loss = np.array(train_loss), val_loss = np.array(val_loss) )

        if best_epoch < epoch - TOL_EPOCH:
            exit()

        #print(epochs,train_loss,val_loss)

print('FIN');
