import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch import nn
import pandas as pd
from tqdm import tqdm
import numpy as np
import nn.functional as F
import os
from scipy.spatial import distance_matrix as dm
from scipy.spatial.transform import Rotation as R
from utils import *


def model_train_sm(loader, model, optimizer, loss_fn, scaler, DEVICE, fake_batch_size=1):
    loop = tqdm(loader)
    #print("HIIIIIIIII")
    all_loss = []

    curr_loss = 0
    fake_batch_size=fake_batch_size
    my_iter = 0;

    model.train()
    for batch_idx, (node_feat, coor, edges, edge_feat, carb_binder, sm_binder, label_res, n_res, n_edge) in enumerate(loop):

        coor = coor.to(device=DEVICE,dtype=torch.float).squeeze()

        #exit the fail_state
        if len(coor.shape) < 2:
            #print('skip')
            continue;


        node_feat = node_feat.to(device=DEVICE,dtype=torch.float).squeeze()

        label_res = label_res > 0.5
        label_res = label_res.to(device=DEVICE,dtype=torch.float).squeeze()


        #exit the fail_state
        if len(coor.shape) < 2:
            continue;
        if len(label_res) != len(node_feat):
            continue;

        #forward
        optimizer.zero_grad()

        pred_res = model(node_feat, coor, edges, edge_feat,
                        is_batch=False, n_res=n_res, n_edge=n_edge)


        #print("PREDRES SIZE:",pred_res.size())
        pred_res = torch.squeeze(pred_res)
        if torch.any(torch.isnan(pred_res)):
            continue;

        curr_loss += loss_fn(pred_res,label_res)
        all_loss.append(curr_loss)
        curr_loss.backward()
        optimizer.step()
        #reset
        curr_loss = 0;
        my_iter += 1;

        temp_loss = torch.FloatTensor(all_loss)
        temp_loss = torch.sum(temp_loss)/len(all_loss)
        loop.set_postfix (loss = temp_loss.item())

    model.eval();
    return temp_loss.item()

def model_train_two_prot(loader, model, optimizer, loss_fn, scaler, DEVICE, epoch=0, MODIFIER=1.0):
    loop = tqdm(loader)
    all_loss = []

    curr_loss = 0
    my_iter = 0;

    model.train()

    for batch_idx, (node_feat, coor, edges, edge_feat, carb_binder, sm_binder, label_res, n_res, n_edge) in enumerate(loop):

        with torch.autograd.set_detect_anomaly(True):

            coor = coor.to(device=DEVICE,dtype=torch.float).squeeze()
            #exit the fail_state
            if len(coor.shape) < 2:
                #print('skip')
                continue;

            node_feat = node_feat.to(device=DEVICE,dtype=torch.float).squeeze()

            label_res = label_res.to(device=DEVICE,dtype=torch.float).squeeze()
            label_prot = carb_binder.to(device=DEVICE,dtype=torch.float).squeeze() * MODIFIER


            #exit the fail_state
            if len(coor.shape) < 2:
                continue;

            if len(label_res) != len(node_feat):
                #print('label-node mismatch')
                continue;

            #forward
            optimizer.zero_grad()


            pred_prot = model(node_feat, coor, edges, edge_feat,
                            is_batch=False, n_res=n_res, n_edge=n_edge)
            if type(pred_prot) == type(0):
                continue

            if torch.any(torch.isnan(pred_prot)):
                print("prot: nan")
                continue;

            pred_prot = pred_prot.squeeze()

            curr_loss = loss_fn(pred_prot,label_prot)

            #Update Loss

            all_loss.append(curr_loss)
            curr_loss.backward()
            optimizer.step()

            temp_loss = torch.FloatTensor(all_loss)
            temp_loss = torch.sum(temp_loss)/len(all_loss)
            loop.set_postfix (loss = temp_loss.item())


    model.eval();
    return temp_loss.item()

def model_val_sm(loader, model, optimizer, loss_fn, scaler, DEVICE):
    loop = tqdm(loader)
    all_loss = []
    clusters = []


    model.eval()
    for batch_idx, (node_feat, coor, edges, edge_feat, carb_binder, sm_binder, label_res, n_res, n_edge) in enumerate(loop):

        with torch.no_grad():

            coor = coor.to(device=DEVICE,dtype=torch.float).squeeze()
            #print(coor.shape,nodes.shape)
            #exit the fail_state
            if len(coor.shape) < 2:
                #print('skip')
                continue;


            node_feat = node_feat.to(device=DEVICE,dtype=torch.float).squeeze()
            label_res = label_res > 0.5
            label_res = label_res.to(device=DEVICE,dtype=torch.float).squeeze()

            if len(label_res) != len(node_feat):
                continue;
            #exit the fail_state
            if len(coor.shape) < 2:
                continue;

            #summary(model,node_feat,coor,edges,edge_feat)

            pred_res = model(node_feat, coor, edges, edge_feat,
                            is_batch=False, n_res=n_res, n_edge=n_edge)
            pred_res = torch.squeeze(pred_res)

            curr_loss = loss_fn(pred_res,label_res)

            all_loss.append(curr_loss)

            temp_loss = torch.FloatTensor(all_loss)
            temp_loss = torch.sum(temp_loss)/len(all_loss)
            loop.set_postfix (loss = temp_loss.item())

    return temp_loss.item()

def model_val_two_prot(loader, model, optimizer, loss_fn, scaler, DEVICE,epoch=0):
    loop = tqdm(loader)
    all_loss = []
    all_track_loss = []
    clusters = []

    cm = np.zeros((2,2))


    model.eval()
    for batch_idx, (node_feat, coor, edges, edge_feat, carb_binder, sm_binder, label_res, n_res, n_edge) in enumerate(loop):

        with torch.no_grad():
            with torch.autograd.set_detect_anomaly(True):


                coor = coor.to(device=DEVICE,dtype=torch.float).squeeze()
                if len(coor.shape) < 2:
                    #print('skip')
                    continue;

                node_feat = node_feat.to(device=DEVICE,dtype=torch.float).squeeze()
                label_res = label_res.to(device=DEVICE,dtype=torch.float).squeeze()
                label_prot = carb_binder.to(device=DEVICE,dtype=torch.float).squeeze()

                #exit the fail_state
                if len(coor.shape) < 2:
                    continue;
                if label_res.size()[0] != node_feat.size()[0]:
                    #print('label-node mismatch')
                    continue;


                pred_prot = model(node_feat, coor, edges, edge_feat,
                                is_batch=False, n_res=n_res, n_edge=n_edge)

                if type(pred_prot) == type(0):
                    continue;

                if torch.any(torch.isnan(pred_prot)):
                    print("prot: nan")
                    continue;

                pred_prot = pred_prot.squeeze()

                if pred_prot.item() > .5:
                    if label_prot.item() > .5:
                        cm[0,0] += 1;
                    else:
                        cm[0,1] += 1
                else:
                    if label_prot.item() > .5:
                        cm[1,0] += 1;
                    else:
                        cm[1,1] += 1

                curr_loss = loss_fn(pred_prot,label_prot)

                all_loss.append(curr_loss)

                temp_loss = torch.FloatTensor(all_loss)
                temp_loss = torch.sum(temp_loss)/len(all_loss)
                loop.set_postfix (loss = temp_loss.item())


    all_track_loss = np.array(all_track_loss)

    return temp_loss.item(), cm

def dice_loss(pred,true,eps=1e-5):
    pred = torch.squeeze(pred)
    true = torch.squeeze(true)
    tp = torch.mul(pred,true);
    #print(pred.shape,true.shape,tp.shape)
    #print(pred[:50],true[:50],tp)
    tp = torch.sum(tp)
    #print(torch.sum(tp),torch.sum(pred),torch.sum(true))

    d = (2 * torch.sum(tp) + eps) / (torch.sum(pred) + torch.sum(true) + eps)
    return 1 - d;


if __name__ == "__main__":
    print("main")
