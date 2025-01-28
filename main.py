import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
from torch.utils.data import DataLoader
import time
import math
from typing import Dict
import random
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors # changed
from rdkit import RDLogger
from tqdm import tqdm
from matplotlib import pyplot as plt

class MyDataset(Dataset):
    def __init__(
        self,
        pathway_list: list[tuple[str, list[str]]],
        label_list: list[float],
    ):
        super().__init__()
        self.smiles: list[str] = [smi for smi, traj in pathway_list]
        self.trajs: list[list[str]] = [traj for smi, traj in pathway_list]
        self.inputs: list[tuple[str, list[str]]] = pathway_list
        self.labels: list[float] = label_list
        self.max_blocks: int = MAX_BLOCKS
        self.augment = False

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        '''see the data structure here!
        pathway = [<smiles1>, 'click|amide', <smiles2>, 'click|amide', <smiles3>, ... <smilesN>]
        label = scalar value
        '''
        label: float = self.labels[idx]
        smi: str = self.smiles[idx]
        pathway: list[str] = self.trajs[idx]

        block_smi_list: list[str] = list(pathway[0::2])
        reaction_list: list[str] = list(pathway[1::2])
        assert len(block_smi_list) == len(reaction_list) + 1
        assert len(block_smi_list) <= self.max_blocks
        assert set(reaction_list) <= {'click', 'amide'}

        # data representation (You maybe focus this part)
        fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)

        mol = Chem.MolFromSmiles(smi)
        fp = fpgen.GetFingerprint(mol)
        if True: # molecular graph (structural data)
            x = {}
            x['h'] = self.mol_to_atom_number_list(mol) # [N_h]
            x['adj']= torch.Tensor(self.mol_to_adj(mol)).long() # [N_h,N_h]
            x['e'] = self.mol_to_e(mol) # [N_h,N_h]
            fp = fpgen.GetFingerprint(mol)
            x['fp']= torch.as_tensor(fp, dtype=torch.float) #[fpSize]

        mask_idx = None
        if self.augment:
            mask_idx = random.choice(list(range(self.max_blocks)))

        reaction_type_to_index = {'click': 1, 'amide': 2, 'else': 3}
        if True: # make fragment sequence & reaction sequence (local property)
            block_seq = torch.zeros([self.max_blocks, 1024]) # [L, fpSize]
            for i, block_smi in enumerate(block_smi_list):
                if i == (len(block_smi_list)-1):
                    block_smi = block_smi[:-1]
                block_mol = Chem.MolFromSmiles(block_smi)
                fp = fpgen.GetFingerprint(block_mol)
                block_seq[i] = torch.as_tensor(fp, dtype=torch.float)
            if self.augment:
                block_seq[mask_idx] = 0

            reac_seq = torch.zeros(self.max_blocks-1, dtype=torch.int) # [L-1]
            for i, reaction in enumerate(reaction_list):
                if reaction not in reaction_type_to_index.keys():
                    reac_seq[i] = reaction_type_to_index['else']
                else:
                    reac_seq[i] = reaction_type_to_index[reaction]
        """ELSE: additional property data"""

        if True: # additional (global property)
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NHOHCount(mol)
            hba = Descriptors.NOCount(mol)
            n_ring = rdMolDescriptors.CalcNumAromaticRings(mol)
            prop = torch.tensor([mw, logp, tpsa, hbd, hba, n_ring], dtype=torch.float)

        sample = {'x': x, 'block_seq': block_seq, 'reac_seq': reac_seq, 'prop': prop, 'y': label}
        return sample

    def mol_to_atom_number_list(self, mol):
        atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        return atomic_numbers

    def mol_to_adj(self, mol):
        num_atoms = mol.GetNumAtoms()
        adj_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj_matrix[start, end] = 1
            adj_matrix[end, start] = 1
        return adj_matrix

    def mol_to_e(self, mol):
        num_atoms = mol.GetNumAtoms()
        # Initialize the edge feature matrix with zeros
        e_features = np.zeros((num_atoms, num_atoms), dtype=int)

        # Mapping from bond type to one-hot index
        bond_type_to_index = {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 1,
            Chem.rdchem.BondType.TRIPLE: 2,
            Chem.rdchem.BondType.AROMATIC: 3
        }

        # Populate the edge feature matrix
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()

            # Get the one-hot index for the current bond type
            bond_index = bond_type_to_index[bond_type]

            # Set the corresponding position to 1 for one-hot encoding
            e_features[start, end] = bond_index
            e_features[end, start] = bond_index # Undirected graph - set for both directions

        return torch.tensor(e_features, dtype=torch.long)

def my_collate_fn(batch: list[dict]):
    if True:
        max_atoms = max(data['x']['adj'].shape[0] for data in batch)
        h_padded = torch.zeros(len(batch), max_atoms, dtype=torch.long)
        adj_padded = torch.zeros(len(batch), max_atoms, max_atoms, dtype=torch.long)
        e_padded = torch.zeros(len(batch), max_atoms, max_atoms)

        for i, data in enumerate(batch):
            num_atoms_i = data['x']['adj'].shape[0]
            h_padded[i, :num_atoms_i] = torch.tensor(data['x']['h'])
            adj_padded[i, :num_atoms_i, :num_atoms_i] = data['x']['adj']
            e_padded[i, : num_atoms_i,: num_atoms_i] = data['x']["e"]
        x_fp = [data['x']['fp'] for data in batch]
    x = {'h':h_padded.long(), 'adj':adj_padded.float(), 'e':e_padded.long(), 'fp': torch.stack(x_fp, dim=0)}
    block_seq = [data['block_seq'] for data in batch]
    reac_seq = [data['reac_seq'] for data in batch]
    prop = [data['prop'] for data in batch]
    y = [data['y'] for data in batch]

    out = {}
    out["x"] = x
    out['block_seq'] = torch.stack(block_seq, dim=0)
    out['reac_seq'] = torch.stack(reac_seq, dim=0)
    out["prop"] = torch.stack(prop, dim=0)
    out["y"] = torch.tensor(y, dtype=torch.float)
    return out

class MPNNLayer(nn.Module):
    def __init__(self, hid_dim):
        super(MPNNLayer, self).__init__()
        self.make_message = nn.Linear(hid_dim * 3, hid_dim, bias=True)
        self.node_update = nn.Linear(hid_dim * 2, hid_dim, bias=True)

    def forward(self, h, adj, e):
        # h is the node feature matrix of shape [bs, N, hid_dim]
        # adj is the adjacency matrix of shape [bs, N, N]
        # e is the edge feature tensor of shape [bs, N, N, hid_dim]

        # make h repeated
        N = h.size(1)
        h_1 = h.unsqueeze(2).repeat(1, 1, N, 1) # [bs, N, 1, hid_dim]
        h_2 = h.unsqueeze(1).repeat(1, N, 1, 1) # [bs, 1, N, hid_dim]

        m = torch.cat([h_1, h_2, e], dim=3) # [bs, N, N, hid_dim * 3]
        m = self.make_message(m) # [bs, N, N, hid_dim]

        # multiply adj to get only "real edges"
        m = m * adj.unsqueeze(3)

        # sum up the message over node
        m = m.sum(1) # [bs, N, hid_dim]

        # Concatenate the summed message vectors with the node features
        h_cat = torch.cat((h, m), dim=2)

        # Update node features with the concatenated features
        h_updated = self.node_update(h_cat)

        return h_updated

class MyModel(nn.Module):
    def __init__(self, input_dim: int, hid_dim: int, n_layer: int):
        super().__init__()
        assert n_layer > 0

        """full mol"""
        self.emb_full_graph_x_h = nn.Embedding(130, hid_dim) # 118 atom types -> hid_dim,
        self.emb_full_graph_x_e = nn.Embedding(4, hid_dim) # 4 edge types -> hid_dim
        self.x_graph_1 = MPNNLayer(hid_dim)
        self.x_graph_2 = MPNNLayer(hid_dim)
        self.emb_full_fp = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(),
        )
        self.cat_x = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.ReLU(),
        )

        self.emb_block_seq = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(),
        )
        self.emb_reac_seq = nn.Sequential(
            nn.Embedding(5, hid_dim),
            nn.ReLU(),
        )
        self.cat_block = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.ReLU(),
        )

        self.norm = nn.BatchNorm1d(6)
        self.emb_prop = nn.Sequential(
            nn.Linear(6, hid_dim),
            nn.ReLU(),
        )

        self.cat_all = nn.Sequential(
            nn.Linear(hid_dim * 3, hid_dim),
            nn.ReLU(),
        )

        mlps = []
        for i in range(n_layer):
            mlps.append(nn.Linear(hid_dim, hid_dim))
            mlps.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlps)

        self.head = nn.Linear(hid_dim, 1)

    def forward(self, x: Dict, block_seq: torch.Tensor, reac_seq: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        '''
        input:
            x : {'h': [bs, N_h], 'adj': [bs, N_h, N_h], 'e': [bs, N_h, N_h, fpSize], 'fp': [bs, fpSize]}
            block_seq: [bs, max_len, fpSize]
            reac_seq: [bs, max_len - 1]
            prop: [bs, 6]
        '''
        if True:
            x_h, x_adj, x_e, x_fp = x['h'], x['adj'], x['e'], x['fp'] 
            x_h = self.emb_full_graph_x_h(x_h) # [bs, N_h] -> [bs, N_h, hid_dim]
            x_e = self.emb_full_graph_x_e(x_e) # [bs, N_h, N_h] -> [bs, N_h, N_h , hid_dim]
            x = self.x_graph_1(x_h, x_adj, x_e) # [bs, N_h, hid_dim]
            x = F.relu(x) 
            x = self.x_graph_2(x, x_adj, x_e) # [bs, N_h, hid_dim]
            x = x.sum(dim=1) # [bs, N_h, hid_dim] -> [bs, hid_dim]

            x_fp = self.emb_full_fp(x_fp) # [bs, fpSize] -> [bs, hid_dim]

            x = torch.cat([x, x_fp], dim=1) # [bs, 2 * hid_dim]
            x = self.cat_x(x) # [bs, 2 * hid_dim] -> [bs, hid_dim]

        if True:
            block_seq = self.emb_block_seq(block_seq)    # [bs, max_len, fpSize] -> [bs, max_len, hid_dim]
            block_seq = block_seq.sum(1)        # [bs, max_len, hid_dim -> [bs, hid_dim]
            reac_seq = self.emb_reac_seq(reac_seq) # [bs, max_len - 1] -> [bs, max_len-1, hid_dim]
            reac_seq = reac_seq.sum(1) # [bs, hid_dim]
            block = torch.cat([block_seq, reac_seq], dim=1) # [bs, 2 * hid_dim]
            block = self.cat_block(block) # [bs, hid_dim]

        """property"""
        if True: # adding prop with batch normalization
            prop = self.norm(prop) # [bs, 6]
            prop = self.emb_prop(prop) # [bs, 6] -> [bs, hid_dim]
            prop = F.relu(prop)

        x = torch.cat([x, block, prop], dim=1)
        x = self.cat_all(x)

        x = self.mlp(x)       # [bs, hid_dim] -> [bs, hid_dim]
        y = self.head(x)      # [bs, hid_dim] -> [bs, 1]
        return y.squeeze(1)   # [bs, 1] -> [bs,]

    def run_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if True:
            x = {}
            x['h'] = batch['x']['h'].to(self.device)
            x['adj'] = batch['x']['adj'].to(self.device)
            x['e'] = batch['x']['e'].to(self.device)
            x['fp'] = batch['x']['fp'].to(self.device)
        block_seq = batch['block_seq'].to(self.device)
        reac_seq = batch['reac_seq'].to(self.device)
        prop = batch['prop'].to(self.device)
        return self.forward(x, block_seq, reac_seq, prop)

    @torch.no_grad()
    def inference_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if True:
            x = {}
            x['h'] = batch['x']['h'].to(self.device)
            x['adj'] = batch['x']['adj'].to(self.device)
            x['e'] = batch['x']['e'].to(self.device)
            x['fp'] = batch['x']['fp'].to(self.device)
        block_seq = batch['block_seq'].to(self.device)
        reac_seq = batch['reac_seq'].to(self.device)
        prop = batch['prop'].to(self.device)
        return self.forward(x, block_seq, reac_seq, prop)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

def create_model():
    return MyModel(1024, hid_dim=hid_dim, n_layer=n_layer)

def single_epoch(model, data_loader, optimizer=None, device='cpu') -> float:
    total_loss = 0.0
    for batch in tqdm(data_loader, leave=False):
        y_pred = model.run_batch(batch)
        y_true = batch["y"].to(device)
        loss = F.mse_loss(y_pred, y_true, reduction="mean")
        total_loss += loss.item()
        if optimizer != None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return total_loss / len(data_loader)

if __name__ == "__main__":
    # Initial Set
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())

    model_path = f"./best_model.pt" # You should submit this file.
    ckpt_path = f"./last_model.pt"

    debug_mode = False
    use_cuda = True

    if use_cuda:
        device = 'cuda'
        num_workers = 2
    else:
        device = 'cpu'
        num_workers = 0
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)

    # Hyperparameters
    lr = 1e-4
    weight_decay = 1e-6
    hid_dim = 128
    n_layer = 2
    train_data_ratio = 0.8   
    step_size = 50
    gamma = 0.5

    n_epochs = 50
    batch_size = 512
    if debug_mode:
        n_epochs = 4
        batch_size = 16
    
    MAX_BLOCKS = 4

    # Load data
    data_file = "./aichem_2024_final_data.pkl"
    with open(data_file, 'rb') as f:
        raw_data = pickle.load(f)

    inputs = raw_data['train']['input']
    labels = raw_data['train']['label']

    # split dataset into train / valid (data is already shuffled)
    if debug_mode:
        Ntot = 1000
    else:
        Ntot = 30000

    Ntrain = int(Ntot * train_data_ratio)
    train_set = MyDataset(inputs[:Ntrain], labels[:Ntrain])
    valid_set = MyDataset(inputs[Ntrain:Ntot], labels[Ntrain:Ntot])
    print(f"length of train, valid set: {len(train_set), len(valid_set)}")

    # define data loader which batches dataset
    train_loader = DataLoader(train_set, batch_size, shuffle=True,
                            num_workers=num_workers, collate_fn=my_collate_fn)
    valid_loader = DataLoader(valid_set, batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=my_collate_fn)

    model = create_model()
    model.to(device)

    # train
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_loss_history, valid_loss_history = [], []
    best_loss = float('inf')
    for epoch_idx in range(1, n_epochs+1):
        tick_st = time.time()
        model.train()
        tloss = single_epoch(model, train_loader, optimizer, device=device)
        with torch.no_grad():
            model.eval()
            vloss = single_epoch(model, valid_loader, device=device)
        train_loss_history.append(tloss)
        valid_loss_history.append(vloss)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        tick_end = time.time()
        print(f"Epoch: {epoch_idx}\tTrain: {tloss:.4f}\tValid: {vloss:.4f}\tTime: {tick_end-tick_st:.2f}\tLearning Rate: {current_lr}")
        if vloss < best_loss:
            best_loss = vloss
            torch.save(model.state_dict(), model_path)
        torch.save(model.state_dict(), ckpt_path)

    # Plot
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(valid_loss_history, label='Validation Loss')
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.savefig('./loss_hisory')

    # Save the model
    model = create_model() # if required, it is allowed to modify here only.
    model.load_state_dict(torch.load(model_path, 'cpu', weights_only=True))
    model.to(device)
    model.eval()
    