# Import necessary libraries
import tensorflow
import tensorflow.keras as keras
import numpy as np
import uproot  # for reading ROOT files
from .utils import to_np_array, get_file_handler  # custom utility functions
import awkward as ak  # for working with variable-length arrays (jagged arrays)


class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_files, features, labels, spectators, batch_size=1024, n_dim=60,
                 remove_mass_pt_window=False, remove_unlabeled=True, return_spectators=False,
                 max_entry=20000, scale_mass_pt=[1, 1]):
        'Initialization of the generator'

        # Basic parameters
        self.batch_size = batch_size
        self.labels = labels
        self.list_files = list_files
        self.features = features
        self.spectators = spectators
        self.return_spectators = return_spectators
        self.scale_mass_pt = scale_mass_pt
        self.n_dim = n_dim  # max number of particles per event
        self.remove_mass_pt_window = remove_mass_pt_window
        self.remove_unlabeled = remove_unlabeled
        self.max_entry = max_entry

        # Bookkeeping lists
        self.global_IDs = []     # unique index for all events across files
        self.local_IDs = []      # local index within each file
        self.file_mapping = []   # mapping from each sample to its file
        self.open_files = [None]*len(self.list_files)  # open file handles (lazy opening)

        # Build index mapping for all events
        running_total = 0
        for i, file_name in enumerate(self.list_files):
            with uproot.open(file_name, **get_file_handler(file_name)) as root_file:
                self.open_files.append(root_file)
                tree = root_file['deepntuplizer/tree']
                tree_length = min(tree.num_entries, max_entry)  # limit entries per file

            self.global_IDs.append(np.arange(running_total, running_total + tree_length))
            self.local_IDs.append(np.arange(0, tree_length))
            self.file_mapping.append(np.repeat([i], tree_length))
            running_total += tree_length

        # Flatten all indices across files
        self.global_IDs = np.concatenate(self.global_IDs)
        self.local_IDs = np.concatenate(self.local_IDs)
        self.file_mapping = np.concatenate(self.file_mapping)

        # Setup initial indexing order
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.global_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Get batch-wise sample indices and file mappings
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        files = self.file_mapping[index * self.batch_size:(index + 1) * self.batch_size]

        unique_files = np.unique(files)

        # Handle case where no valid files found in batch
        if len(unique_files) == 0:
            return None

        # Get start and stop indices for each unique file in the batch
        starts = np.array([min(indexes[files == i]) for i in unique_files])
        stops = np.array([max(indexes[files == i]) for i in unique_files])

        # Open necessary files and close unused ones
        for ifile, file_name in enumerate(self.list_files):
            if ifile in unique_files:
                if self.open_files[ifile] is None:
                    self.open_files[ifile] = uproot.open(file_name, **get_file_handler(file_name))
            else:
                if self.open_files[ifile] is not None:
                    self.open_files[ifile].close()
                    self.open_files[ifile] = None

        # Generate batch data
        X, y = self.__data_generation(unique_files, starts, stops)

        # Return None if data couldn't be retrieved
        if X is None or len(X) == 0:
            return None

        return X, y

    def on_epoch_end(self):
        'Shuffle or reset indices after each epoch'
        self.indexes = self.local_IDs  # no shuffling, can add if needed

    def __data_generation(self, unique_files, starts, stops):
        'Generates data for the current batch by reading ROOT files'
        Xs = []
        ys = []
        zs = []


        # Loop through each file to read the required entries
        for ifile, start, stop in zip(unique_files, starts, stops):

            # Depending on configuration, get (X, y) or (X, [y, z])
            if self.return_spectators:
                X, [y, z] = self.__get_features_labels(ifile, start, stop)
            else:
                X, y = self.__get_features_labels(ifile, start, stop)

            # Skip if file gave empty/invalid data
            if X is None or y is None or len(X) == 0:
                continue

            # Append to list
            Xs.append(X)
            ys.append(y)
            if self.return_spectators:
                zs.append(z)

        # Raise error if no valid data was retrieved from all files
        if len(Xs) == 0:
            raise ValueError("Error: No data was generated. Check filtering conditions or file contents!")

        # Combine all samples across files
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)

        if self.return_spectators:
            z = np.concatenate(zs, axis=0)
            return X, [y, z]

        return X, y

    def __get_features_labels(self, ifile, entry_start, entry_stop):
        'Loads and processes a specific slice of data from one file'

        # Ensure file is open
        if self.open_files[ifile] is None:
            root_file = uproot.open(self.list_file[ifile], **get_file_handler(self.list_file[ifile]))
        else:
            root_file = self.open_files[ifile]

        # Load the tree
        tree = root_file['deepntuplizer/tree']

        # Read feature arrays (jagged format)
        feature_array = tree.arrays(self.features,
                                    entry_start=entry_start,
                                    entry_stop=entry_stop+1,
                                    library='ak')

        ###############################
        # Handle jet-level features (e.g., 'fj_pt') by copying their value to every particle
        for feature in self.features:
            if feature[:2] == 'fj':
                new_values = []
                for i in range(len(feature_array)):
                    if len(feature_array[i]['track_pt']) != 0:
                        values = [feature_array[i][feature]] * len(feature_array[i]['track_pt'])
                        values = ak.Array(values)
                        new_values.append(values)
                    elif len(feature_array[i]['track_pt']) == 0:
                        values = []
                        values = ak.Array(values)
                        new_values.append(values)
                feature_array = ak.with_field(feature_array, new_values, feature)
        ###############################

        # Load labels as numpy arrays
        label_array_all = tree.arrays(self.labels,
                                      entry_start=entry_start,
                                      entry_stop=entry_stop+1,
                                      library='np')

        # Convert jagged particle features to fixed-size arrays
        X = np.stack([to_np_array(feature_array[feat], max_n=self.n_dim, pad=0) for feat in self.features], axis=2)
        n_samples = X.shape[0]

        # Binary labels: [QCD, Higgs]
        y = np.zeros((n_samples, 2))
        y[:, 0] = label_array_all['sample_isQCD'] * (label_array_all['label_QCD_b'] +
                                                     label_array_all['label_QCD_bb'] +
                                                     label_array_all['label_QCD_c'] +
                                                     label_array_all['label_QCD_cc'] +
                                                     label_array_all['label_QCD_others'])
        y[:, 1] = label_array_all['label_H_bb']

        # Read spectator variables if needed (e.g., pt, mass)
        if self.remove_mass_pt_window or self.return_spectators:
            spec_array = tree.arrays(self.spectators,
                                     entry_start=entry_start,
                                     entry_stop=entry_stop+1,
                                     library='np')
            z = np.stack([spec_array[spec] for spec in self.spectators], axis=1)

        # Apply pt and mass cuts if required
        if self.remove_mass_pt_window:
            X = X[(z[:, 0] > 40) & (z[:, 0] < 200) & (z[:, 1] > 300) & (z[:, 1] < 2000)]
            y = y[(z[:, 0] > 40) & (z[:, 0] < 200) & (z[:, 1] > 300) & (z[:, 1] < 2000)]
            z = z[(z[:, 0] > 40) & (z[:, 0] < 200) & (z[:, 1] > 300) & (z[:, 1] < 2000)]

        # Remove events with no label (e.g., all zero rows in `y`)
        if self.remove_unlabeled:
            X = X[np.sum(y, axis=1) == 1]
            if self.return_spectators:
                z = z[np.sum(y, axis=1) == 1]
            y = y[np.sum(y, axis=1) == 1]

        # Return (X, y, z) or (X, y)
        if self.return_spectators:
            return X, [y, z / self.scale_mass_pt]  # normalize spectators
        return X, y
