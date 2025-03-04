## Prediction Utils ##
## Makes it easy for us to just predict literally everything in a single command :) ##

"""
Created on Jan 14 2025

@author: swcanner

 Current settings for B Factor visualization:
 `BFactor =  0.0` : Nonbinder
 `BFactor = 99.9` : CAPSIF2 Predicted binding residue


Usage: python run_both.py
    --capsif2_only [run capsif2 only | default: False]
    --picap_only [run picap only | default: False]
    --high_plddt [run only on high_plddt residues (with > `plddt_cutoff`) only | default: False]
    --plddt_cutoff [cutoff for when `--high_plddt` invoked | default: 70]
    --json [output to json file instead | default: False]
Returns:
    `./output_data/predictions_res.tsv` - tsv of capsif2 residue predictions
    `./output_data/predictions_prot.tsv` - tsv of picap predictions
    `./output_data/all_predictions.tsv` - tsv of capsif2 and picap predictions together
                only done if picap and capsif2 are run together

            OR

    `./output_data/predictions.json` - json file of picap and/or capsif2 predictions


    and `output_data/*_predictions.pdb` with the pdbs with the BFactor identifying the binding CAPSIF2 Residues
"""

RUN_PICAP = True;
RUN_CAP = True;
HIGH_PL = False;
PL_CUT = 70;
JSON = False;

OUTPUT_INT_TO_CMD = False;
OUTPUT_CMD = True;
OUTPUT_CAP2_PDBS = True;
SINGLE = False;


import sys
import os

def str_to_bool(a):
    #Changes string to bool for flags
    a = a.upper()
    b = True
    if (a == "0" or a == "F" or a == "FALSE" or a == "N" or a == "NO"):
        b = False;
    return b

def manage_flags(flags):

    """
    function manages all input flags

    Args:
        flags (str) : input flags for the file
    Returns:
        RUN_PICAP (bool) : run picap
        RUN_CAP (bool) : run capsif2
        HIGH_PL (bool) : run only on high-res residues
        PL_CUT (str) : what is the cutoff for resolution
        SINGLE (bool) : only run on single structure (used only for notebooks)
        JSON (bool) : run json output files only
    """

    #Get all flags all organized
    n = len(flags)

    input_flags = ["--capsif2_only","--picap_only",'--high_plddt','--plddt_cutoff','--single','--json','--help']

    RUN_PICAP = True;
    RUN_CAP = True;
    HIGH_PL = False;
    PL_CUT = 70;
    SINGLE = False;
    JSON = False;

    if n > 1:
        for kk in input_flags:
            if kk in flags:
                kk = kk.lower()
                ind = flags.index(kk);

                if (kk == '--capsif2_only'):
                    RUN_PICAP = False
                if (kk == '--picap_only'):
                    RUN_CAP = False
                if (kk == '--high_plddt'):
                    HIGH_PL = True
                if (kk == '--plddt_cutoff'):
                    a = flags[ind+1]
                    a = float(a)
                    PL_CUT = a
                if (kk == '--single'):
                    SINGLE = True
                if (kk == '--json'):
                    JSON = True

                if (kk == '--help'):
                    print("""
PiCAP and CAPSIF2 help:

    Usage: python run_both.py
        --capsif_only [run capsif2 only | default: False]
        --picap_only [run picap only | default: False]
        --high_plddt [run only on high_plddt residues (with > `plddt_cutoff`) only | default: False]
        --plddt_cutoff [cutoff for when `--high_plddt` invoked | default: 70]
        --json [output json files instead of tsv | default: False]
    Returns:
    `./output_data/predictions_res.tsv` - tsv of capsif2 residue predictions
    `./output_data/predictions_prot.tsv` - tsv of picap predictions
    `./output_data/all_predictions.tsv` - tsv of capsif2 and picap predictions together
                only done if picap and capsif2 are run together

            OR

    `./output_data/predictions.json` - json file of picap and/or capsif2 predictions


    and `output_data/*_predictions.pdb` with the pdbs with the BFactor identifying the binding CAPSIF2 Residue
                    """)
                    exit()


    print("\n\nRunning with the following flags: ")
    print("Run PiCAP : \t",RUN_PICAP)
    print("Run CAPSIF2: \t",RUN_CAP)
    print("Run High pLDDT only: \t",HIGH_PL)
    if HIGH_PL:
        print("pLDDT cutoff: \t",PL_CUT)
    print("JSON output: \t", JSON)

    return RUN_PICAP, RUN_CAP, HIGH_PL, PL_CUT, SINGLE, JSON



from preprocess import *

import os
import numpy as np
import pandas as pd
from utils import *
from egnn.egnn import *
import matplotlib.pyplot as plt
from torchvision.models.feature_extraction import create_feature_extractor
import re
import json


SPECIES = 'TEST_FILE'
TEST_PDB =   './pre_pdb/dataset_pdb.csv'
TEST_CLUST = './pre_pdb/dataset_clust.csv'

NUM_WORKERS = 0; #This is for cpu use; will auto change to 8 if you are using gpu

RUN_PICAP = True;
RUN_CAP = True;
HIGH_PL = False;
OUTPUT_INT_TO_CMD = True;
OUTPUT_CMD = True;
OUTPUT_CAP2_PDBS = True;
SINGLE = False; #for use only by the notebook!

if SINGLE:
    TEST_PDB =   './pre_pdb/dataset_single_pdb.csv'
    TEST_CLUST = './pre_pdb/dataset_single_clust.csv'

def run_capsif2(TEST_PDB,TEST_CLUST,JSON=False):

    """
    Runs Capsif2 and predicts all residues on given input pdb/cluster files
    Arguments:
        TEST_PDB (string): Path to input PDB csv file
        TEST_CLUST (string): Path to the input CLUSTER csv file (not used)
    Returns:
        names (arr, string): all the input pdb names
        res_label (2d arr, string): predicted residues of the associated pdbs
    Outputs:
        `./output_data/predictions_res.tsv` - tsv of capsif2 residue predictions

    """

    print("Loading Capsif2...")

    BATCH_SIZE = 1;
    NUM_WORKERS = 0;

    KNN = [16,16,16,16]
    N_LAYERS = [3,3,3,3]
    HIDDEN_NF = 128
    CUTOFF = 0.001

    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        NUM_WORKERS = 8;
    print("Using: " + DEVICE)
    DEVICE = torch.device(DEVICE)

    #Load Test Dataset
    test_loader = get_test_loader(TEST_CLUST,TEST_PDB,root_dir="./",train=0,
                                    batch_size=1,num_workers=NUM_WORKERS,knn=KNN, return_pdb_ref=True)


    model = CAPSIF2_RES2(hidden_nf = HIDDEN_NF, n_layers=N_LAYERS,normalize=False,
                    device=DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    torch.autograd.set_detect_anomaly(True)
    model.train()

    my_model_name = 'capsif2'

    if DEVICE == 'cuda':
        checkpoint = torch.load("./models_DL/model-" + my_model_name + ".pt")
    else:
        checkpoint = torch.load("./models_DL/model-" + my_model_name + ".pt",
            map_location=torch.device('cpu') )
    model.load_state_dict(checkpoint['model_state_dict'])
    #print(checkpoint['info'])

    print("Capsif2 loaded")

    model.eval()
    pred_res, names, res_label  = model_test_res_env(test_loader, model, DEVICE=DEVICE)

    names = np.array(names)

    #fix the names to not include "_0"
    for ii in range(len(names)):
        names[ii][0] = names[ii][0][ :names[ii][0].rfind('_') ]

    file = "./output_data/predictions_res.tsv"
    print('\n\t------Capsif2 results-------')

    f = []
    if not JSON:
        f = open(file,'a+')

    for ii in range(len(names)):
        #print(names[ii][0])
        if not JSON:
            f.write(names[ii][0] + '\t')
        if OUTPUT_INT_TO_CMD:
            print(names[ii][0],end=":")
            if len(res_label[ii]) < 1:
                print(' n/a',end='')
                if not JSON:
                    f.write('n/a')

        for jj in range(len(res_label[ii])):
            if OUTPUT_INT_TO_CMD:
                print(res_label[ii][jj][0], end=",")

            if not JSON:
                f.write(res_label[ii][jj][0] + ',')
        if not JSON:
            f.write('\n')

        if OUTPUT_INT_TO_CMD:
            print()

        if OUTPUT_CAP2_PDBS:
            #output the CAPSIF2 predicted residues pdb

            #Get the name of the pdb without the chain name added
            #instances = [m.start() for m in re.finditer('_', names[ii][0])]
            #my_name = names[ii][0][:instances[-1]]
            my_name = names[ii][0]
            if 'highPL' in names[ii][0]:
                my_name = names[ii][0][:names[ii][0].rfind('_')];


            #need to get full name of the input pdb
            ls = os.listdir('./input_pdb/')

            the_input_pdb_file = ''

            for jj in ls:
                if '.pdb' not in jj:
                    continue;

                if my_name in jj:
                    the_input_pdb_file = jj
                    break;

            if the_input_pdb_file == '':
                print("something messed up in output of:",my_name)
            output_structure_bfactor(file='./input_pdb/' + the_input_pdb_file,res=pred_res_to_str(res_label[ii]),
                         out_file= './output_data/' + names[ii][0] + '_predictions.pdb')

    if not JSON:
        f.close()


    return names, res_label

def run_picap(TEST_PDB,TEST_CLUST,JSON=False):
    """
    Runs Capsif2 and predicts all residues on given input pdb/cluster files
    Arguments:
        TEST_PDB (string): Path to input PDB csv file
        TEST_CLUST (string): Path to the input CLUSTER csv file (not used)
    Returns:
        names (arr, string): all the input pdb names
        prot_pred (arr, float): predicted probability of carb binding
    Outputs:
        "./output_data/predictions_prot.tsv" - tsv of picap predictions
    """

    print("\n\n\nloading PiCAP")
    NUM_WORKERS = 0;

    #Hyper parameters!
    KNN = [10,20,40,60]
    N_LAYERS = [3,3,3,3]
    HIDDEN_NF = 128
    ADAPOOL_SIZE = (150,HIDDEN_NF)

    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        NUM_WORKERS = 8;
    print("Using: " + DEVICE)
    DEVICE = torch.device(DEVICE)
    print(DEVICE)

    test_loader = get_test_loader(TEST_CLUST,TEST_PDB,root_dir="./",train=0,
                                    batch_size=1,num_workers=NUM_WORKERS,knn=KNN)

    model = PICAP(hidden_nf = HIDDEN_NF, n_layers=N_LAYERS,
                    device=DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    torch.autograd.set_detect_anomaly(True)
    model.train()

    my_model_name = 'picap'

    if DEVICE == 'cuda':
        checkpoint = torch.load("./models_DL/model-" + my_model_name + ".pt")
    else:
        checkpoint = torch.load("./models_DL/model-" + my_model_name + ".pt",
            map_location=torch.device('cpu') )
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    print("PiCAP loaded")

    print("Running Predictions...")

    model.eval()
    prot_pred, names  = model_test_prot_env(test_loader, model, DEVICE=DEVICE)

    prot_pred = np.array(prot_pred)
    prot_pred = prot_pred.reshape((-1))

    names = np.array(names)
    #fix the names to not include "_0"
    for ii in range(len(names)):
        names[ii][0] = names[ii][0][ :names[ii][0].rfind('_') ]

    file = "./output_data/predictions_prot.tsv"
    print('\n\t------PiCAP results-------')
    out = ''
    for ii in range(len(names)):
        if OUTPUT_INT_TO_CMD:
            print(names[ii][0],',', str(prot_pred[ii]))
        out += str(names[ii][0]) + '\t' + str(round(prot_pred[ii],4)) + '\n'
    if not JSON:
        if not os.path.exists(file):
            out = 'PDB_NAME\tpred\n' + out
        f = open(file,'a+')
        f.write(out)
        f.close()

    return names, prot_pred

def run_it_all(RUN_CAP=True,RUN_PICAP=True,single=False,JSON=False):

    """
    Runs all arguments in single function to do capsif2 and picap
    Arguments:
        single: used by notebook for running a single file
    Returns:
        names_cap (arr, string): all the input pdb names from capsif2
        names_pi  (arr, string): all the input pdb names from picap
        cap_pred (2d arr, string): predicted residues of the associated pdbs
        pi_pred (arr, float): predicted probability of protein-carb binding

    Note:
        This function removes all intermediate files after running
    """

    TEST_PDB =   './pre_pdb/dataset_pdb.csv'
    TEST_CLUST = './pre_pdb/dataset_clust.csv'
    if single:
        TEST_PDB =   './pre_pdb/dataset_single_pdb.csv'
        TEST_CLUST = './pre_pdb/dataset_single_clust.csv'

    names_cap, cap_pred = [], []
    names_pi, pi_pred = [], []
    if RUN_CAP:
        names_cap, cap_pred = run_capsif2(TEST_PDB,TEST_CLUST,JSON)

    if RUN_PICAP:
        names_pi, pi_pred = run_picap(TEST_PDB,TEST_CLUST,JSON)

    #simple simple simple O(n2) json writing if both outputs
    if JSON:
        mydict = {}

        if RUN_PICAP and RUN_CAP:
            for ii in range(len(names_cap)):
                for jj in range(len(names_pi)):
                    if names_cap[ii][0] == names_pi[jj][0]:
                        txt = ''
                        for kk in range(len(cap_pred[ii])):
                            txt += cap_pred[jj][kk][0] + ','
                        mydict.update( { names_cap[ii][0] : {
                                    'prot_pred': str(round(pi_pred[jj],4)),
                                    'res_pred': txt} } )
        elif RUN_PICAP:
            for jj in range(len(names_pi)):
                mydict.update( { names_pi[jj][0] : {
                            'prot_pred': str(round(pi_pred[jj],4)) } } )
        elif RUN_CAP:
            for ii in range(len(names_cap)):
                txt = ''
                for kk in range(len(cap_pred[ii])):
                    txt += cap_pred[ii][kk][0] + ','
                mydict.update( { names_cap[ii][0] : {
                            'res_pred': txt} } )

        with open('output_data/predictions.json', 'w+') as f:
            json.dump(mydict,f, indent=4);
        #return names_cap, names_pi, cap_pred, pi_pred



    print('\n\n\n')

    #remove intermediate files
    int_dir = './pre_pdb/'
    ls = os.listdir(int_dir)
    for ii in ls:
        filename, file_extension = os.path.splitext(int_dir + ii)
        if '.npy' == file_extension or '.npz' == file_extension:
            os.remove(int_dir + ii)

    file = './output_data/all_predictions'
    if single:
        file += '_single'
    file += '.tsv'
    txt = ''
    if not os.path.exists(file):
        txt = 'NAME\tBinder_pred\tRes_pred\n'

    if RUN_CAP and RUN_PICAP:
        for ii in range(len(names_pi)):
            for jj in range(len(names_cap)):
                if names_pi[ii][0] == names_cap[jj][0]:
                    txt += names_pi[ii][0] + '\t'
                    txt += str(round(pi_pred[ii],4)) + '\t'
                    for kk in range(len(cap_pred[jj])):
                        txt += cap_pred[jj][kk][0] + ','
                    if len(cap_pred[jj]) < 1:
                        txt += 'n/a'
                    txt += '\n'
                    break;
    if not single:
        if not JSON:
            f = open(file,'a+')
            f.write(txt)
            f.close()

    if OUTPUT_CMD and RUN_CAP and RUN_PICAP:
        print("Total output:")
        print(txt)

    print("\nFin.")

    return names_cap, names_pi, cap_pred, pi_pred

    #return;

if __name__ == "__main__":

    #Maintain for reproducible values across both cpu and gpu
    torch.backends.cuda.matmul.allow_tf32 = True


    RUN_PICAP, RUN_CAP, HIGH_PL, PL_CUT, SINGLE, JSON = manage_flags(sys.argv)

    run_preprocess(HIGH_PL,PL_CUT)
    print("Preprocessing complete\n\n")
    names_cap, names_pi, cap_pred, pi_pred = run_it_all(RUN_CAP,RUN_PICAP,JSON=JSON)
