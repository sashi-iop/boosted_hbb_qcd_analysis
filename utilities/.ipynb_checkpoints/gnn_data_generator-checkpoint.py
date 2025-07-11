# Import necessary modules
import os.path as osp
import torch
from torch_geometric.data import Dataset, Data  # PyG classes for creating graph datasets
import itertools
import numpy as np
import uproot  # To read ROOT files
import glob
import multiprocessing
from pathlib import Path
import yaml  # For reading definitions.yml
from tqdm.notebook import tqdm  # For progress bars in notebooks
import awkward as ak  # For handling jagged arrays
from .utils import get_file_handler  # Custom utility to get file handler for uproot


class GraphDataset(Dataset):
    def __init__(self, root, features, labels, spectators, transform=None, pre_transform=None,
                 n_events=-1, n_events_merge=1000, file_names=None, remove_unlabeled=True, remove_mass_pt_window=True):
        """
        Initialize parameters of graph dataset
        Args:
            root (str): Dataset root directory
            features (list): List of input features
            labels (list): List of label variables
            spectators (list): List of spectator variables (e.g., pt, mass)
            transform (callable): Optional transform on the data
            pre_transform (callable): Optional transform applied before saving
            n_events (int): Number of events to process (-1 = all)
            n_events_merge (int): How many events to merge per .pt file
            file_names (list): Custom list of input ROOT files
            remove_unlabeled (bool): Whether to remove samples with no labels
            remove_mass_pt_window (bool): Apply pt/mass window cuts
        """
        self.features = features
        self.labels = labels
        self.spectators = spectators
        self.n_events = n_events
        self.n_events_merge = n_events_merge
        self.file_names = file_names
        self.remove_unlabeled = remove_unlabeled
        self.remove_mass_pt_window = remove_mass_pt_window
        super(GraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """
        Returns the list of raw ROOT file names.
        Uses default path if file_names is None.
        """
        if self.file_names is None:
            return ['root://eospublic.cern.ch//eos/opendata/cms/datascience/HiggsToBBNtupleProducerTool/HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC/train/ntuple_merged_10.root']
        else:
            return self.file_names

    @property
    def processed_file_names(self):
        """
        Returns the list of processed `.pt` files in the processed directory.
        """
        proc_list = glob.glob(osp.join(self.processed_dir, 'data*.pt'))
        return_list = list(map(osp.basename, proc_list))
        return return_list

    def len(self):
        """
        Returns the number of processed `.pt` files.
        """
        return len(self.processed_file_names)

    def download(self):
        """
        Optional: implement file download logic here.
        """
        pass

    def process(self):
        """
        Converts raw ROOT files into graph datasets.
        Each sample is converted to a PyTorch Geometric `Data` object.
        Merges `n_events_merge` graphs into one `.pt` file.
        """
        for raw_path in self.raw_file_names:
            with uproot.open(raw_path, **get_file_handler(raw_path)) as root_file:

                # Access the TTree named 'deepntuplizer/tree'
                tree = root_file['deepntuplizer/tree']

                # Load input features as an awkward array
                feature_array = tree.arrays(self.features,
                                            entry_stop=self.n_events,
                                            library='ak')

                ###############################
                # Special handling of jet-level features (starting with 'fj')
                # Repeats the fj-level value for each particle in the event
                for feature in self.features:
                    if feature[:2] == 'fj':
                        new_values= []
                        for i in range(len(feature_array)):
                            if len(feature_array[i]['track_pt']) != 0:
                                values = [0]* len(feature_array[i]['track_pt'])
                                values[0] = feature_array[i][feature]
                                values = ak.Array(values)
                                new_values.append(values)
                            elif len(feature_array[i]['track_pt']) == 0:
                                values = []
                                values = ak.Array(values)
                                new_values.append(values)
                        feature_array = ak.with_field(feature_array, new_values, feature)
                ###############################

                # Load labels and spectators as numpy arrays
                label_array_all = tree.arrays(self.labels,
                                              entry_stop=self.n_events,
                                              library='np')

                spec_array = tree.arrays(self.spectators,
                                         entry_stop=self.n_events,
                                         library='np')
            
            # Construct binary classification labels: [QCD, Higgs]
            n_samples = label_array_all[self.labels[0]].shape[0]
            y = np.zeros((n_samples, 2))
            y[:, 0] = label_array_all['sample_isQCD'] * (label_array_all['label_QCD_b'] +
                                                         label_array_all['label_QCD_bb'] +
                                                         label_array_all['label_QCD_c'] +
                                                         label_array_all['label_QCD_cc'] +
                                                         label_array_all['label_QCD_others'])
            y[:, 1] = label_array_all['label_H_bb']

            # Stack spectators into a 2D array
            z = np.stack([spec_array[spec] for spec in self.spectators], axis=1)

            # Apply pT and mass cuts if required
            if self.remove_mass_pt_window:
                feature_array = feature_array[(z[:, 0] > 40) & (z[:, 0] < 200) & (z[:, 1] > 300) & (z[:, 1] < 2000)]
                y = y[(z[:, 0] > 40) & (z[:, 0] < 200) & (z[:, 1] > 300) & (z[:, 1] < 2000)]
                z = z[(z[:, 0] > 40) & (z[:, 0] < 200) & (z[:, 1] > 300) & (z[:, 1] < 2000)]

            n_samples = y[:, 0].shape[0]
            
            for i in tqdm(range(n_samples)):
                # Start a new list of graphs every `n_events_merge` samples
                if i % self.n_events_merge == 0:
                    datas = []

                # Skip unlabeled samples (sum of both class labels == 0)
                if self.remove_unlabeled:
                    if np.sum(y[i:i+1], axis=1) == 0:
                        continue

                # Skip samples with < 2 particles (no edges)
                n_particles = len(feature_array[self.features[0]][i])
                if n_particles < 2:
                    continue

                # Create edge index: all directed pairs of particles (fully connected, excluding self-loops)
                pairs = np.stack([[m, n] for (m, n) in itertools.product(range(n_particles), range(n_particles)) if m != n])
                edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()

                # Create node features from per-particle feature arrays
                x = torch.tensor([feature_array[feat][i].to_numpy() for feat in self.features], dtype=torch.float).T

                # Spectator values for the event (e.g., pt, mass)
                u = torch.tensor(z[i], dtype=torch.float)

                # Construct the PyG Data object
                data = Data(x=x, edge_index=edge_index, y=torch.tensor(y[i:i+1], dtype=torch.long))
                data.u = torch.unsqueeze(u, 0)  # Add batch dimension to spectator info

                # Apply optional filtering or transformation
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                # Add to batch list
                datas.append([data])

                # Save batch to file once `n_events_merge` graphs are collected
                if i % self.n_events_merge == self.n_events_merge-1:
                    datas = sum(datas, [])  # Flatten list of lists
                    torch.save(datas, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))

    def get(self, idx):
        """
        Loads a processed `.pt` file by index.
        """
        p = osp.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(p, weights_only=False)
        return data


if __name__ == "__main__":
    # Argument parser for running this script directly
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset path", required=True)
    parser.add_argument("--n-events", type=int, default=-1, help="number of events (-1 means all)")
    parser.add_argument("--n-events-merge", type=int, default=1000, help="number of events to merge")
    args = parser.parse_args()

    # Load feature, spectator, label definitions from YAML
    with open('definitions(cloned).yml') as file:
        definitions = yaml.load(file, Loader=yaml.FullLoader)

    features = definitions['features']
    spectators = definitions['spectators']
    labels = definitions['labels']

    # Create dataset object
    gdata = GraphDataset(args.dataset, features, labels, spectators,
                         n_events=args.n_events,
                         n_events_merge=args.n_events_merge)
