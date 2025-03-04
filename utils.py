import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from scipy.spatial import distance_matrix as dm
from scipy.spatial.transform import Rotation as R
import math
import py3Dmol
from Bio.PDB import *
from colorama import Fore, Style

class CSV_Dataset(Dataset):
    def __init__(self, cluster_file,  pdb_file, root_dir, nn=[6,12,18,24], train=False,
                    use_clusters=False,val=False,use_af2=True,af2_likelihood=.65,return_name=False, return_pdb_ref=False):
        """
        Arguments:
            cluster_file (string): Path to csv file of cluster/pdb annotations
            pdb_file (string): Path to the csv file with pdb file annotations
            root_dir (string): Directory to where csv_files were initialized from (home dir)
            nn (array): list of nearest neighbors to be used
            train (bool): if we are in evaluation of performance at all - includes validation and test (return label)
            use_clusters (bool): do we evaluate the sequences on cluster based information
            val (bool): if we are in validation / test mode -> returns the cluster and pdb_name
        """
        self.pdb_data = pd.read_csv(pdb_file,header=None)
        self.cluster_data = pd.read_csv(cluster_file)
        #print(self.data)
        self.root_dir = root_dir
        self.train = train;
        self.val = val;

        self.use_clusters = use_clusters;
        self.clusters_epoch = [];

        #nearest neighbors
        self.nn = nn;
        self.use_af2 = use_af2
        self.af2_likelihood = af2_likelihood

        self.return_pdb_ref = return_pdb_ref;

        #distance encodings RBF = gaus(phi) = exp( (eps * (x - u) )^2 )
        self.rbf_dist_means = np.linspace(0,20,16)
        self.rbf_eps = (self.rbf_dist_means[-1] - self.rbf_dist_means[0]) / len(self.rbf_dist_means);

        self.return_name = return_name

        self.fail_state = [0,0,0,0,0,0,0,0,0]
        if self.return_name:
            self.fail_state.append(0)

    def __len__(self):
        if (self.use_clusters):
            return len(self.cluster_data)
        else:
            return len(self.pdb_data)

    def __getitem__(self,idx,pad=True):
        """
        Arguments:
            idx (int): CSV file index for training/testing
            pad (bool): whether to pad the output or not
        Returns:
            esm_ (array): ESM features in all chains
            cb (array): coordinates of CBs in all chains
            edges (array of arrays (4xn) ): Edges of NN with varying cutoffs
            edge_feats (array of arrays (4xn) ): Edge features of each nearest neighbors
                    RBF of distance, Direction encoding, orientation encoding
            binder_type (array): Is_smallMol_binder , is_carb_binder
            carbs_bound (array): array of all bound carbs if present in struct
        """

        if torch.is_tensor(idx):
            idx = idx.tolist();



        clust = ""
        pdb_name = ""
        coor_files = ""
        esm_files = ""
        af_files = ""
        sm_binder = False
        carb_binder = False
        ref_pdb = []

        #training - get the pdb thru clusters
        if self.use_clusters:
            clust = self.cluster_data.iloc[idx,0]
            pdbs = self.cluster_data.iloc[idx,1].split('|')[:-1] #ends with a | so we remove the null

            #get random pdb
            r = np.random.randint(0,len(pdbs),size=1)

            #print(r)
            #r = [0]
            pdb_name = pdbs[r[0]]


            print(clust, pdb_name)
            try:
                new_df = self.pdb_data[self.pdb_data[1] == pdb_name].values[0]
                #print(new_df,clust)
                if clust != new_df[0]:
                    print("Failure to match: ",clust,new_df[0],pdb_name)

                coor_files = new_df[2]
                esm_files = new_df[3]
                af_files = new_df[4]
                carb_binder = new_df[5]
                sm_binder = new_df[6]

            except:
                print("Failure to find: ",clust,pdb_name)
                return self.fail_state
        else:
            clust = self.pdb_data.iloc[idx,0]
            pdb_name = self.pdb_data.iloc[idx,1]
            coor_files = self.pdb_data.iloc[idx,2]
            esm_files = self.pdb_data.iloc[idx,3]
            af_files = self.pdb_data.iloc[idx,4]
            carb_binder = self.pdb_data.iloc[idx,5]
            sm_binder = self.pdb_data.iloc[idx,6]


        #print(idx,self.cluster_data.iloc[idx,:])

        #"Cluster,PDB,coor_files,esm_files,AF2_files,carb,sm";

        #load coor files together
        #try:
        ca,cb,frame,res_label = [],[],[],[]
        ca,cb,frame,res_label = self.load_coor(coor_files)

        if self.return_pdb_ref:
            ref_pdb = self.load_ref(coor_files)

        if type(res_label) == type(-1):
            return self.fail_state

        if (self.use_af2):
            #if af files exist
            if type(af_files) != type(np.nan):

                #see if we use the af file
                r = np.random.rand(0)
                if r < self.af2_likelihood:
                    ca_af,cb_af,frame_af = self.load_coor_af(coor_files)

                    if (len(ca) != len(ca_af)):
                        print("AF and PDB do not match!!!! ",pdb_name)
                    else:
                        ca,cb,frame = ca_af,cb_af,frame_af

        if bool(carb_binder):
            if len(np.shape(res_label)) > 1:
                res_label = torch.unsqueeze(torch.from_numpy(res_label[:,0]),1)
            else:

                res_label = torch.unsqueeze(torch.from_numpy(res_label),1)
        else:
            res_label = torch.unsqueeze(torch.from_numpy(res_label),1)
        esm_ = self.load_esm(esm_files)


        self.n_res = esm_.shape[0]

        #get the neighbors!
        edges,edge_feat = self.get_knn_info(cb,frame,self.nn)

        #make it torchy
        esm_ = torch.FloatTensor(esm_)
        #carb_oneHot = torch.IntTensor(carb_oneHot)
        cb = torch.FloatTensor(cb)
        #type_label = torch.IntTensor( [small_mol, is_na_binder, is_carb] )

        #return what is needed
        #return edge_feat
        if self.return_pdb_ref:
            return esm_, cb, edges, edge_feat, carb_binder, sm_binder, res_label, self.n_res, self.n_edge, pdb_name, ref_pdb

        if self.return_name:
            return esm_, cb, edges, edge_feat, carb_binder, sm_binder, res_label, self.n_res, self.n_edge, pdb_name

        return esm_, cb, edges, edge_feat, carb_binder, sm_binder, res_label, self.n_res, self.n_edge

    def load_carbs_oneHot(self,carbs):
        """
        Arguments:
            carbs (string): carb numbers seperated by "|"
        Returns:
            oneHot (np.array): array of carbohydrates bound in one-hot encoding
        """
        oneHot = np.zeros((len(carb_dict),))

        if type(carbs) == type(np.nan):
            return oneHot
        if carbs == "":
            return oneHot

        l = carbs.split('|')
        if len(l) == 0:
            return oneHot

        for ii in l:
            oneHot[int(ii)] = 1;
        return oneHot

    def load_esm(self,files):
        """
        Arguments:
            files (string): file names seperated by "|" to esm embedding files
        Returns:
            esm_ (np.array): array of all esm embeddings for all proteins
        """
        l = files.split('|')
        if "" in l:
            l.remove("")
        esm_ = [];

        for ii in l:
            #print(len(ii))
            curr = np.load(self.root_dir + ii)
            for jj in curr:
                esm_.append(jj)
        return np.array(esm_)

    def load_coor(self,files):
        """
        Arguments:
            files (string): file names seperated by "|" to coordinate files
        Returns:
            ca (np.array): array of all CA coor
            cb (np.array): array of all CB coor
            frame (np.array): array of all oriented frames
            label (np.array): Nres x 17 of all carbs bound
        """
        l = files.split('|')
        if "" in l:
            l.remove("")
        ca = [];
        cb = [];
        frame = []
        label = [];

        for ii in l:
            curr = np.load(self.root_dir + ii)
            ca_c = curr['ca']
            cb_c = curr['cb']
            frame_c = curr['frame']

            if self.train:
                try:
                    label_c = curr['label']
                except:
                    label_c = np.zeros((cb_c.shape[0],1))

            for jj in range(len(ca_c)):
                #REMOVE DUPLICATES!!!!
                #this is a very lazy unoptimized way to do this but it works
                skip_round = False;
                for kk in range(len(ca)):
                    if ca_c[jj][0] == ca[kk][0]:
                        if ca_c[jj][1] == ca[kk][1]:
                            if ca_c[jj][2] == ca[kk][2]:
                                skip_round=True;
                                break;
                if skip_round:
                    continue;

                ca.append(ca_c[jj])
                cb.append(cb_c[jj])
                frame.append(frame_c[jj])
                if self.train:
                    if jj >= len(label_c):
                        label.append(0)
                    else:
                        label.append(label_c[jj])

        ca = np.array(ca)
        cb = np.array(cb)
        frame = np.array(frame)

        try:
            label = np.stack(label)
        except:
            if self.train:
                return -1, -1, -1 ,-1
            else:
                return -1, -1, -1

        if self.train:
            return ca, cb, frame, label
        return ca, cb, frame


    def load_ref(self,files):
        """
        Arguments:
            files (string): file names seperated by "|" to coordinate files
        Returns:
            ref (np.array): reference pdb information
        """
        l = files.split('|')
        if "" in l:
            l.remove("")
        ref = []



        for ii in l:
            #print(self.root_dir + ii)
            curr = np.load(self.root_dir + ii)
            c_ref = curr['ref']
            for jj in range(len(c_ref)):

                ref.append(c_ref[jj])


        return ref


    def load_coor_af(self,files):
        """
        Arguments:
            files (string): file names seperated by "|" to coordinate files
        Returns:
            ca (np.array): array of all CA coor
            cb (np.array): array of all CB coor
            frame (np.array): array of all local frames
        """
        l = files.split('|')
        l.remove("")
        ca = [];
        cb = [];
        frame = []

        for ii in l:
            #print(self.root_dir + ii)
            curr = np.load(self.root_dir + ii)
            ca_c = curr['ca']
            cb_c = curr['cb']
            frame_c = curr['frame']

            for jj in range(len(ca_c)):
                ca.append(ca_c[jj])
                cb.append(cb_c[jj])
                frame.append(frame_c[jj])

        ca = np.array(ca)
        cb = np.array(cb)
        frame = np.array(frame)

        return ca, cb, frame

    def get_knn_info(self,coor,frame,num_neigh, eps=[1e-5,1e-5,1e-5] ):
        """
        Arguments:
            coor (arr): coordinates to be analyzed
            frame (arr): local frame information per residue
            num_neigh (arr): array of number of nearest neighbors
        Returns:
            edges (2d array): Edges of all nodes - first index is array num-neigh related
            edge_feats (2d array): Edge features of each edge above - first index is array num-neigh related
        """

        dist = dm(coor,coor);
        dist_sort = np.argsort(dist)
        edge1 = [];
        edge2 = [];
        feats = [];

        max_neigh = np.max(num_neigh)

        for i in range(len(num_neigh)):
            edge1.append([])
            edge2.append([])
            feats.append([])

        #go thru all coordinates
        for i in range(len(coor)):

            #1 - range = skip self
            for kk in range(1,max_neigh+1):

                #assert i != dist_sort[i,kk]
                if kk >= len(dist_sort[i]):
                    continue;
                #dont include self
                #Skip self if we get self - just double down and make sure
                if (dist[i,dist_sort[i,kk]] == 0):
                    continue;

                #get RBF
                #distances.append(dist_sort[i,kk])
                dist_from_rbf = dist[i,dist_sort[i,kk]] - self.rbf_dist_means;
                my_rbf = np.exp( -( dist_from_rbf / self.rbf_eps )**2 )

                #get orientation
                orient = np.matmul( frame[i],np.transpose(frame[kk]) )
                o = R.from_matrix(orient)
                quat = o.as_quat()

                #get direction
                vec = coor[dist_sort[i,kk]] - coor[i] + eps;
                vec /= np.linalg.norm(vec);
                direct = np.matmul( frame[i], vec)

                #put all info into a single array;
                val = [];
                for jj in my_rbf:
                    val.append(jj)
                for jj in quat:
                    val.append(jj);
                for jj in direct:
                    val.append(jj);

                #append our neighborhood info
                for jj in range(len(num_neigh)):
                    if kk <= num_neigh[jj]:
                        edge1[jj].append(i)
                        edge2[jj].append(dist_sort[i,kk])
                        feats[jj].append(val)

        fake_val = list( np.zeros((23,)) )
        #concatenate them and make them torch-worthy
        #get the num_edges

        n_edges = []
        for ii in edge1:
            n_edges.append(np.shape(ii)[0])
        self.n_edge = n_edges

        edges = [];
        for jj in range(len(num_neigh)):
            edges.append(torch.stack([ torch.LongTensor(edge1[jj]), torch.LongTensor(edge2[jj]) ]))
            feats[jj] = torch.FloatTensor(feats[jj])

        return edges, feats


def fix_edges(edges,feats,n_edges):
    """
    Arguments:
        edges (arr): list of padded edges (n_block, n_batch, 2, n_edge)
        feats (arr): list of edge_feats (n_block, n_batch, 2, n_edge)
        n_edges (arr): length of unpadded edges (n_block,n_edge)
    Returns:
        edges (2d array): Edges of all nodes - first index is array num-neigh related
        edge_feats (2d array): Edge features of each edge above - first index is array num-neigh related
    """

    new_edges = []
    new_feats = []

    sz_block = np.shape(edges)
    sz_batch = np.shape(edges[0])

    #Go thru each elem in batch
    for ii in range(sz_batch[0]):
        # go thru each block in batch
        new_edges.append([])
        new_feats.append([])

        #Go thru each block
        for jj in range(sz_block[0]):
            new_edges[ii].append(edges[jj][ii,:,:n_edges[jj][ii]])
            new_feats[ii].append(feats[jj][ii,:n_edges[jj][ii]])

    return new_edges, new_feats


def get_test_loader(test_file_cluster,  test_file_pdb, root_dir="../", train=0,
                batch_size=1, num_workers=0, test_cluster=False,
                knn=[6,12,18,24], pin_memory=True,return_pdb_ref=False):

    """
    Arguments:
        test_file_cluster (str): cluster file of directories of the test files used
        test_file_pdb (str): pdb file of directories of the test files used
        root_dir (str): directory used as basis of root
        train (int/bool): used when label is known for training
        batch_size (int): num of entries in a batch (1)
        num_workers (int): num threads
        test_cluster (bool): run through all pdbs or just thru cluster by cluster
        knn (int): KNN used for EGCLs with edges
        pin_memory (bool) : pin_memory
        return_pdb_ref (bool) : Return PDB numbering
    Returns:
        val loader (DataLoader): test dataloader
    """

    val_ds = CSV_Dataset(  test_file_cluster,  test_file_pdb, root_dir=root_dir, train=1,
        use_clusters=test_cluster,nn=knn, val=True, return_name=True, return_pdb_ref=return_pdb_ref)
    val_loader = DataLoader( val_ds, batch_size=1, num_workers=num_workers,
        pin_memory=pin_memory, shuffle=False )

    return val_loader


#Prediction / Inference code
def model_test_prot_env(loader, model, DEVICE='cpu'):
    """
    Picap Prediction
    Arguments:
        loader (dataloader): test dataloader
        model (str): picap loaded model
        DEVICE (str): cpu / gpu
    Returns:
        prot_pred (arr): predicted values of protein
        names (arr): pdb names associated with prot_pred
    """

    loop = tqdm(loader)
    prot_pred, prot_label = [], [];
    res_pred, res_label = [],[];
    names = []

    n_stuff = 0

    model.eval()
    for batch_idx, (node_feat, coor, edges, edge_feat, carb_binder, sm_binder, label_res, n_res, n_edge, name) in enumerate(loop):

        with torch.no_grad():

            coor = coor.to(device=DEVICE,dtype=torch.float32).squeeze()

            #exit the fail_state
            if len(coor.shape) < 2:
                continue;

            node_feat = node_feat.to(device=DEVICE,dtype=torch.float32).squeeze()
            #exit the fail_state
            if len(coor.shape) < 2:
                continue;

            pred_prot = model(node_feat, coor, edges, edge_feat,
                            is_batch=False, n_res=n_res, n_edge=n_edge)


            prot_pred.append(pred_prot.detach().cpu().numpy())
            names.append(name)

            n_stuff += 1

    return prot_pred, names

def model_test_res_env(loader, model, DEVICE='cpu',CUTOFF = 0.001):
    """
    Capsif2 Prediction
    Arguments:
        loader (dataloader): test dataloader
        model (str): picap loaded model
        DEVICE (str): cpu / gpu
        CUTOFF (float): cutoff value for inferring if a residue binds
    Returns:
        pred_res (arr): predicted residues of protein
        names (arr): pdb names associated with prot_pred
        res_label (arr): PDB code of residues predicted to bind
    """

    loop = tqdm(loader)
    pred_res, res_label = [], [];
    res_pred, res_label = [],[];
    names = []

    n_stuff = 0

    model.eval()
    for batch_idx, (node_feat, coor, edges, edge_feat, carb_binder, sm_binder, label_res, n_res, n_edge, name, ref_pdb) in enumerate(loop):

        with torch.no_grad():
            coor = coor.to(device=DEVICE,dtype=torch.float32).squeeze()

            #exit the fail_state
            if len(coor.shape) < 2:
                continue;

            node_feat = node_feat.to(device=DEVICE,dtype=torch.float32).squeeze()

            #exit the fail_state
            if len(coor.shape) < 2:
                continue;

            pred = model(node_feat, coor, edges, edge_feat,
                            is_batch=False, n_res=n_res, n_edge=n_edge)


            pred_res.append(pred.detach().cpu().numpy())
            c_p = pred.detach().cpu().numpy().reshape(-1)
            c_res = []
            for kk in range(len(c_p)):
                if c_p[kk] > CUTOFF:
                    c_res.append(ref_pdb[kk])

            res_label.append( c_res )
            names.append(name)

            n_stuff += 1



    return pred_res, names, res_label

### Notebook prediction utils ###
#stolen from https://github.com/ProteinDesignLab/protein_seq_des/blob/master/seq_des/util/data.py
def download_pdb(pdb, data_dir):
    """Function to download pdb -- either biological assembly or if that
    is not available/specified -- download default pdb structure
    Uses biological assembly as default, otherwise gets default pdb.

    Args:
        pdb (str): pdb ID.
        data_dir (str): path to pdb directory
    Returns:
        f (str): path to downloaded pdb
    """
    f = data_dir + "/" + pdb + ".pdb"
    print("Running")
    if not os.path.isfile(f):
        try:
            print("a")
            os.system("wget -O {}.gz https://files.rcsb.org/download/{}.pdb1.gz".format(f, pdb.upper()))
            os.system("gunzip {}.gz".format(f))
        except:
            print('b')
            f = data_dir + "/" + pdb + ".pdb"
            if not os.path.isfile(f):
                os.system("wget -O {} https://files.rcsb.org/download/{}.pdb".format(f, pdb.upper()))
    else:
        print('c')
        f = data_dir + "/" + pdb + ".pdb"
    if not os.path.isfile(f):
        os.system("wget -O {} https://files.rcsb.org/download/{}.pdb".format(f, pdb.upper()))
    return f

def visualize(pdb_file,r="a.b",width=600,height=500,colors=['lime','gray']):
    """
    Arguments:
        pdb_file (string): Path to pdb file to be shown
        r (string): residues predicted, (organized as NUM.CHAIN)
        color (array): colors for [protein, predicted_res]
    Returns:
        py3Dmol session with viewing the residues
    """


    with open(pdb_file) as ifile:
        system = "".join([x for x in ifile])

    view = py3Dmol.view(width=width, height=height)
    view.addModelsAsFrames(system)

    #print(r)
    if ("," in r):
        r = r.split(",")
    else:
        r = [r]

    i = 0
    for line in system.split("\n"):
        split = line.split()
        if len(split) == 0 or (split[0] != "ATOM" and split[0] != "HETATM"):
            continue
        if split[3] == "TIP3" or split[3] == "HOH":
            continue

        my_boi = split[5] + "." + split[4]
        idx = int(split[1])

        #show sidechains as sticks
        if (my_boi in r) and (split[2] != "N" and split[2] != "O" and split[2] != "C" and split[2] != "CA"):
            view.setStyle({'model': -1, 'serial': i+1}, {"stick": {'color': colors[0]}} )
        #color predicted backbone
        elif (my_boi in r):
            view.setStyle({'model': -1, 'serial': i+1}, {"cartoon": {'color': colors[0]}} )
        #color not-predicted backbone
        else:
            view.setStyle({'model': -1, 'serial': i+1}, {"cartoon": {'color': colors[1]}})

        #show the glycan in purple

        i += 1
    view.zoomTo()
    view.show()

def pred_res_to_str(pred):
    """
    Returns the canonical residue.chain string for use of notebook functions
    Arguments:
        pred (arr): residues predicted by cap2
    returns:
        txt (str): residues predicted by cap2 in a single string
    """
    txt = ''
    for jj in range(len(pred)):
        markymark = pred[jj][0].split(' ')
        txt += markymark[0] + '.' + markymark[1] + ','

    return txt

def cif_to_pdb(file):
    """
    There exists an issue with pyrosetta loading in cif files
    This function changes cif to pdb for input

    Arguments:
        file (string): Path to pdb file to edited
    Returns:
        out_file (string): output file
    Output:
        pdb file at out_file
    """

    parser = MMCIFParser()
    data = parser.get_structure('CAPS',file)

    #just change the extension to - super lazy
    #may cause errors on non-trivial cases
    out_file = file[:file.find('.cif')] + '.pdb'

    io = PDBIO()
    io.set_structure(data)
    io.save(out_file)
    return out_file


def output_structure_bfactor(file,res,out_file):
    """
    Outputs files for PDB for quick viewing of CAPSIF2 predictions

    Arguments:
        file (string): Path to pdb file to edited
        res (string): residues predicted, (organized as NUM.CHAIN)
        out_file (string): output pdb file with capsif2 labeled residues
    Returns:
        void
    Output:
        pdb file at out_file
    """

    if (len(res) < 1):
        res = '-1.A'
    res = res.split(',')

    #Create a parser adn read the structures
    parser = PDBParser()
    data = parser.get_structure('CAPS',file)

    #go thru all chains and residues and atoms
    models = data.get_models()
    models = list(models)
    for m in range(len(models)):
        chains = list(models[m].get_chains())
        for c in range(len(chains)):
            residues = list(chains[c].get_residues())
            for r in range(len(residues)):
                #check if its a binding residue
                temp = 1.00
                #its a predicted residue -> BFactor = 99.99
                my_res = str(residues[r].id[1]).strip() + "." + str(chains[c].id).strip()
                if my_res in res:
                    temp = 99.99

                atoms = list(residues[r].get_atoms())
                for a in range(len(atoms)):
                    atoms[a].set_bfactor(temp)
                    #print(chains[c].id,residues[r].id[1],atoms[a].name)
    #output the file
    io = PDBIO()
    io.set_structure(data)
    io.save(out_file)

    return;


if __name__ == "__main__":
    print("main")
