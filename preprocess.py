#preprocess all files in input_dir to output_dir
input_dir = "./input_pdb/"
output_dir = './pre_pdb/'


#load libraries
import os
import numpy as np
import pandas as pd
import torch
import esm
from tqdm import tqdm

import sys

from utils import cif_to_pdb

from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.teaching import *

from pyrosetta.rosetta.core.select.residue_selector import *
from pyrosetta.rosetta.core.simple_metrics.metrics import *
from pyrosetta.rosetta.core.simple_metrics.composite_metrics import *
from pyrosetta.rosetta.core.simple_metrics.per_residue_metrics import *

options = """
-beta
-ignore_unrecognized_res
-out:level 100
"""

#-out:level 100

pyrosetta.init(" ".join(options.split('\n')))

#required functions
def get_chain_seq(pose):
    """
    Gets pose sequence by cahin
    Args:
        pose : pyrosetta pose
    Returns:
        chains (arr) : list of all sequences by internal ordering
    """
    chains = [];
    for ii in range(pose.num_chains()):
        r = pose.chain_begin(ii+1)
        c = pose.pdb_info().pose2pdb(r)
        c = c.split(' ')
        while '' in c:
            c.remove('')
        c = c[-1]
        chains.append(c)
    return chains

def get_protchainXYZ(pose,chain_num):

    """
    Args:
        pose : rosetta pose
        chain_num (int): chain number (rosetta numbering)
    Returns:
        p_coor (arr): array of protein coordinates
        p_label (arr): array with residue numbering (pose numbering)
    """

    p_coor = []; #protein coordinates
    p_label = []; # [RESIDUE_NUMBER] - rosetta numbering

    for jj in range(pose.chain_begin(chain_num),pose.chain_end(chain_num)+1):

        res_number = jj;
        num_of_atoms = pose.residue(res_number).natoms()
        for i in range(num_of_atoms):
            atom_name = pose.residue(res_number).atom_name(i+1).strip()
            if atom_name.count('H')> 0:
                continue
            if atom_name.startswith('V')> 0:
                continue
            curr = np.array(pose.residue(res_number).atom(i+1).xyz())
            p_coor.append(curr)
            p_label.append(res_number)

    return p_coor, p_label

def get_chain_coor(pose,chain):

    """
    function to get all CB and CA atom positions of all residues and their local frame of reference

    Args:
        pose : rosetta pose
        chain : rosetta pose chain number (1,2,...)
    Returns:
        cb (arr): array of all CB coordinates - glycine just CA
        ca (arr): array of all CA coordinates
        frame (arr) : array of all local frame ~
            x' = ca - n , y' = (ca - n) x (ca - c) , z' = x' x y'
        ref (arr): array of PDB nomenclature for each residue
        beta (arr): array of BFactors/PLDDTs of the structure
    """

    start = pose.chain_begin(chain)
    end = pose.chain_end(chain)

    cb = np.zeros((end-start+1,3))
    ca = np.zeros((end-start+1,3))
    ref_pdb = []
    beta = []
    num_res = 0

    frame = np.zeros((end-start+1,3,3))

    for ii in range(start,end+1):

        res = pose.residue(ii);
        beta.append( float( pose.pdb_info().temperature(ii,1) ) )
        num_res += 1

        if (res.is_protein() == False):
            return [],[],[],[];

        ref_pdb.append(pose.pdb_info().pose2pdb(ii))


        #get atom coordinates
        xyz = res.xyz('N')
        n = np.array([xyz[0],xyz[1],xyz[2]])
        xyz = res.xyz('CA')
        a = np.array([xyz[0],xyz[1],xyz[2]])
        xyz = res.xyz('C')
        c = np.array([xyz[0],xyz[1],xyz[2]])
        b = a

        name = res.name1();
        if name != "G":
            xyz = res.xyz('CB')
            b = np.array([xyz[0],xyz[1],xyz[2]])

        #get reference frame
        ca_n = a - n;
        ca_n /= np.linalg.norm(ca_n);
        x_prime = ca_n;
        ca_c = a - c;
        ca_c /= np.linalg.norm(ca_c);
        y_prime = np.cross(ca_n,ca_c);
        y_prime /= np.linalg.norm(y_prime);
        z_prime = np.cross(x_prime,y_prime);
        z_prime /= np.linalg.norm(z_prime);

        #explcitly define as
        #        [ -x'- ]
        # ref =  [ -y'- ]
        #        [ -z'- ]
        ref = np.zeros((3,3));
        ref[0,:] = x_prime;
        ref[1,:] = y_prime;
        ref[2,:] = z_prime;

        #update
        cb[ii-start,:] = b;
        ca[ii-start,:] = a;
        frame[ii-start,...] = ref

    return cb, ca, frame, ref_pdb, beta

def rosetta_preprocess(f,output_dir):

    """
    function preprocess a specific file using pyrosetta to get coordinates and sequence
    outputs a file

    Args:
        f : pdb file (str)
        output_dir : where the output file will be dumped to (str)
    Returns:
        out_fasta : fasta sequence of all chains (arr)
        beta : Average B factor of entire structure
    """

    pose = pose_from_file(f)

    chains = get_chain_seq(pose)
    p = f.split('/')[-1].split('.')[0] #get the name of the file

    out_fasta = []
    #go thru all protein chains
    nc = pose.num_chains();
    num_res = 0;
    beta = [];

    for c in range(1,nc+1):

        #only protein chains allowed
        if pose.residue(pose.chain_begin(c)).is_protein() == False:
            continue;

        coor, label = get_protchainXYZ(pose,c)
        cb, ca, frame, ref_pdb, b = get_chain_coor(pose,c)
        beta.append(b)

        if len(cb) == 0:
            continue;

        seq = pose.chain_sequence(c)
        n = p + "_" + str(c)
        out_fasta.append([n,seq])

        #output the coor file
        np.savez(output_dir + n + ".npz",ca=ca,cb=cb,frame=frame,ref=ref_pdb)

    return out_fasta, beta

def rosetta_highPL_preprocess(f):

    """
    function preprocess a specific file using pyrosetta to get coordinates and sequence
    outputs a file alongside the BFactors of each residue

    Args:
        f : pdb file (str)
        output_dir : where the output file will be dumped to (str)
    Returns:
            All are arrays of arrays - each chain is the first entry
        out_fasta (arr) : fasta sequence of all chains (arr)
        ca_ (arr): array of all CA coordinates
        cb_ (arr): array of all CB coordinates - glycine just CA
        frame (arr) : array of all local frame ~
            x' = ca - n , y' = (ca - n) x (ca - c) , z' = x' x y'
        ref (arr): array of PDB nomenclature for each residue
        beta (arr): array of BFactors/PLDDTs of the structure
    """

    pose = pose_from_file(f)

    chains = get_chain_seq(pose)
    p = f.split('/')[-1].split('.')[0] #get the name of the file

    #print(ii,len(pdbs),f)
    out_fasta = []
    #go thru all protein chains
    nc = pose.num_chains();
    num_res = 0;


    ca_ = []
    cb_ = []
    f_ = []
    ref_ = []
    beta_ = []

    for c in range(1,nc+1):

        #only protein chains allowed
        if pose.residue(pose.chain_begin(c)).is_protein() == False:
            continue;

        coor, label = get_protchainXYZ(pose,c)
        cb, ca, frame, ref_pdb, b = get_chain_coor(pose,c)
        beta_.append(b)

        if len(cb) == 0:
            continue;

        seq = pose.chain_sequence(c)
        n = p + "_" + str(c)
        out_fasta.append([n,seq])

        ca_.append(ca)
        cb_.append(cb)
        f_.append(frame)
        ref_.append(ref_pdb)

    return out_fasta, ca_,cb_,f_,ref_, beta_


def esm_preprocess(fa,model,alphabet,batch_converter,output_dir, high_pl = False):

    """
    function preprocess a series of sequences with ESM

    Args:
        fa : fasta information [[name,seq]] (arr)
        model : ESM-2 Model (model)
        alphabet : ESM-2 alphabet (alphabet)
        batch_converter : ESM-2 preprocessor (converter)
        output_dir : where the output file will be dumped to (str)
    Returns:
        y : ESM-2 embedding
    """

    y = []

    for ii in range(len(fa)):
        #print(ii,fa[ii])

        batch_labels, batch_strs, batch_tokens = batch_converter([fa[ii]])
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        seq_rep = []
        for i, tokens_len in enumerate(batch_lens):
            seq_rep.append(token_representations[i, 1 : tokens_len - 1])

        y.append( seq_rep[0].numpy() )

        if not high_pl:
            #output to file
            name = fa[ii][0]
            np.save(output_dir + name + "_esm.npz",seq_rep[0].numpy())

    return y;

def run_preprocess(high_plddt=False,plddt_cut=70):
    """
    function preprocess all structures in the input directory (input_pdb/)
    stores intermediate values in the output_dir (pre_pdb/)

    Args:
        high_plddt (bool) : only use high-resolution residues
        plddt_cut (float) : cutoff value for resolution of residues
    Returns:
        void

    Notes:
        input_dir and output_dir are provided at the top of the file
        If issues exist, please consult that
    """

    #load ESM Model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    #get all the files
    ls = os.listdir(input_dir)

    print('\npreprocessing...')

    if high_plddt:

        for i in tqdm(range(len(ls))):
            ii = ls[i]

            if ('.pdb' in ii) or ('.cif' in ii):


                #Make the cif file into a pdb file
                if '.cif' in ii:
                    new_pdb = cif_to_pdb(input_dir + ii)
                    ii = new_pdb.split('/')[-1]


                p = ii.split('/')[-1].split('.')[0] #get the name of the file
                print(p)

                #try:
                if True:
                    fa, ca_,cb_,f_,ref_, beta_ = rosetta_highPL_preprocess(input_dir + ii)
                    es_ = esm_preprocess(fa,model,alphabet,batch_converter,output_dir, high_plddt)

                    num_res = 0

                    for kk in range(len(fa)):

                        ca,cb,fi,ref,es = [],[],[],[],[]
                        #print(fa[kk][1])
                        for jj in range(len(fa[kk][1])):
                            #print('\t',fa[kk][1][jj])


                            #only output the ones above plddt cutoff
                            if beta_[kk][jj] > plddt_cut:
                                num_res += 1;
                                ca.append(ca_[kk][jj])
                                cb.append(cb_[kk][jj])
                                fi.append(f_[kk][jj])
                                ref.append(ref_[kk][jj])
                                es.append(es_[kk][jj])



                        n = p + "_highPL_" + str(kk)
                        #print(len(fa[kk][1]),len(ca),len(cb),len(es))

                        np.savez(output_dir + n + ".npz",ca=np.array(ca),cb=np.array(cb),frame=np.array(fi),ref=np.array(ref))
                        np.save(output_dir + n + "_esm.npz.npy",es)

                    if num_res < 10:
                        print('Less than 10 residues were available for input protein structure above the requested plddt_cutoff of ' + str(plddt_cut))
                        print('Exiting...')
                        sys.exit(1)

                    #for i in range(len(fa)):
                    #    fasta.write('>' + fa[i][0] + '|' + str(beta[i]) + '\n' + fa[i][1] + '\n')


                #except:
                #    print("unable: ",ii)
                #break

    else:
        fasta = open(output_dir + 'fasta.fa','a+')
        for i in tqdm(range(len(ls))):
            ii = ls[i]

            if ('.pdb' in ii) or ('.cif' in ii):
                #Make the cif file into a pdb file
                if '.cif' in ii:
                    new_pdb = cif_to_pdb(input_dir + ii)
                    ii = new_pdb.split('/')[-1]

                p = ii.split('/')[-1].split('.')[0] #get the name of the file

                print('preprocessing:\t',p)

                try:
                    fa, beta = rosetta_preprocess(input_dir + ii, output_dir)
                    _ = esm_preprocess(fa,model,alphabet,batch_converter,output_dir)


                    for i in range(len(fa)):
                        fasta.write('>' + fa[i][0] + '|' + str(beta[i]) + '\n' + fa[i][1] + '\n')


                except:
                    print("unable: ",ii)
        fasta.close()


    print('making CSVs for file input')

    #only ouptut files that haven't been made yet
    ls = os.listdir(output_dir)
    out = ''

    cl = 'CLUST,PDB1|PDB2\n'

    done = [];
    for p in ls:
        #only grab npz
        if '.npz' not in p:
            continue;
        if 'esm' in p:
            continue;
        if 'DS_Store' in p:
            continue;
        #just double down

        #remove all high_plddt when in basic mode
        if not high_plddt:
            if "highPL" in p:
                continue;
        #remove all non-high_plddt if in high mode
        else:
            if "highPL" not in p:
                continue;

        name = p.split('.')[0];
        short_name = name[:name.rfind('_')]
        #if 'highPL' in short_name:
        #    short_name = short_name[:short_name.rfind('_')+1]
        #    print(short_name)

        if short_name in done:
            continue;
        done.append(short_name)


        chains = []
        for ii in ls:
            if 'esm' in ii:
                continue;
            #remove all high_plddt when in basic mode
            if not high_plddt:
                if "highPL" in ii:
                    continue;
            #remove all non-high_plddt if in high mode
            else:
                if "highPL" not in ii:
                    continue;

            if short_name in ii:
                n = ii.split('.')[0];
                c = n[n.rfind('_'):];
                chains.append(c)

                print(n,c)

        cl += name + '|' + name + '\n'

        out += name + ',' + name + ','
        for jj in range(len(chains)):
            out += output_dir + short_name + chains[jj] + '.npz'
            if jj + 1 == len(chains):
                continue;
            out += '|'
        out += ','
        for jj in range(len(chains)):
            out += output_dir + short_name + chains[jj] + '_esm.npz.npy'
            if jj + 1 == len(chains):
                continue;
            out += '|'
        out += ','

        out += ',,\n'

    output_file = output_dir + 'dataset'
    f = open(output_file + "_pdb.csv",'w+')
    f.write(out)
    f.close()

    f = open(output_file + "_clust.csv",'w+')
    f.write(cl)
    f.close();

    print('Outputed preprocessed files to: ', output_file + "_pdb  and " + output_file + "_clust .csv")

def preprocess_single(file):
    """
    function preprocess only a single structure in the input directory (input_pdb/)
    stores intermediate values in the output_dir (pre_pdb/)

    Args:
        file (str) : directory to the file
    Returns:
        void

    Notes:
        input_dir and output_dir are provided at the top of the file
        If issues exist, please consult that
    """

    #load ESM Model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    fasta = open(output_dir + 'single_fasta.fa','a+')

    print('preprocessing...')


    p = file.split('/')[-1].split('.')[0] #get the name of the file
    s = p;

    print('file:',file,'\n',p)
    fa, beta = rosetta_preprocess(file, output_dir)
    esm_preprocess(fa,model,alphabet,batch_converter,output_dir)

    for i in range(len(fa)):
        fasta.write('>' + fa[i][0] + '|' + str(beta[i]) + '\n' + fa[i][1] + '\n')

    fasta.close()

    print('making CSVs for file input')

    #only ouptut files that haven't been made yet
    ls = os.listdir(output_dir)

    out = ''

    cl = 'CLUST,PDB1|PDB2\n'

    done = [];
    for p in ls:
        if s not in p:
            continue;

        #only grab npz
        if '.npz' not in p:
            continue;
        if 'esm' in p:
            continue;
        if 'DS_Store' in p:
            continue;
        #just double down

        name = p.split('.')[0];
        short_name = name[:name.rfind('_')]

        if short_name in done:
            continue;
        done.append(short_name)


        chains = []
        for ii in ls:
            #print(ii)
            if 'esm' in ii:
                continue;
            if short_name in ii:
                n = ii.split('.')[0];
                c = n[n.rfind('_'):];
                chains.append(c)

        #print(chains)

        cl += name + '|' + name + '\n'

        out += name + ',' + name + ','
        for jj in range(len(chains)):
            out += output_dir + short_name + chains[jj] + '.npz'
            if jj + 1 == len(chains):
                continue;
            out += '|'
        out += ','
        for jj in range(len(chains)):
            out += output_dir + short_name + chains[jj] + '_esm.npz.npy'
            if jj + 1 == len(chains):
                continue;
            out += '|'
        out += ','

        out += ',,\n'

    output_file = output_dir + 'dataset_single'
    f = open(output_file + "_pdb.csv",'w+')
    f.write(out)
    f.close()

    f = open(output_file + "_clust.csv",'w+')
    f.write(cl)
    f.close();

    print('Outputted preprocessed files to: ', output_file + "_pdb  and " + output_file + "_clust .csv")

if __name__ == '__main__':
    run_preprocess()
    #return;
