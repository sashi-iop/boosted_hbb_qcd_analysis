# Import necessary libraries
import uproot                   # For reading ROOT files
import numpy as np              # For numerical operations
import awkward as ak            # For jagged array support from ROOT trees

# ======================================================================================
# Utility to get correct file handler for local or remote ROOT files (e.g. via XRootD)
# Fixes memory issues or enables multithreading depending on the file source
# ======================================================================================
def get_file_handler(file_name):
    xrootd_src = file_name.startswith("root://")  # check if file is on remote (EOS)
    if not xrootd_src:
        # For local files, use multithreaded file source to avoid memory mapping issues
        return {"file_handler": uproot.MultithreadedFileSource}
    elif xrootd_src:
        # For XRootD files (remote), use multithreaded XRootD source for performance
        return {"xrootd_handler": uproot.source.xrootd.MultithreadedXRootDSource}
    return {}  # default fallback (unused)

# ======================================================================================
# Utility to find the index and value of the element in `array` closest to `value`
# ======================================================================================
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

# ======================================================================================
# Convert awkward array to numpy array with fixed max size and padding
# max_n: maximum number of entries per event
# pad: value used for padding shorter arrays
# ======================================================================================
def to_np_array(ak_array, max_n=100, pad=0):
    return ak.fill_none(ak.pad_none(ak_array, max_n, clip=True, axis=-1), pad).to_numpy()

# ======================================================================================
# Load features, labels, and spectators from a ROOT file, apply filtering
# Can apply pT/mass window and remove unlabeled samples
# ======================================================================================
def get_features_labels(file_name, features, spectators, labels, remove_mass_pt_window=True, entry_stop=None):
    # Open ROOT file and access TTree
    root_file = uproot.open(file_name, **get_file_handler(file_name))
    tree = root_file['deepntuplizer/tree']

    # Load feature, spectator, and label arrays
    feature_array = tree.arrays(features, entry_stop=entry_stop, library='np')
    spec_array = tree.arrays(spectators, entry_stop=entry_stop, library='np')
    label_array_all = tree.arrays(labels, entry_stop=entry_stop, library='np')

    # Stack feature and spectator arrays along axis=1
    feature_array = np.stack([feature_array[feat] for feat in features], axis=1)
    spec_array = np.stack([spec_array[spec] for spec in spectators], axis=1)

    # Number of jets (events)
    njets = feature_array.shape[0]

    # Construct 2-class label array: [QCD, Higgs]
    label_array = np.zeros((njets, 2))
    label_array[:, 0] = label_array_all['sample_isQCD'] * (
        label_array_all['label_QCD_b'] +
        label_array_all['label_QCD_bb'] +
        label_array_all['label_QCD_c'] +
        label_array_all['label_QCD_cc'] +
        label_array_all['label_QCD_others']
    )
    label_array[:, 1] = label_array_all['label_H_bb']

    # ==================================================================================
    # Optional: remove samples outside pt/mass window
    # spec_array[:, 0] = fj_pt, spec_array[:, 1] = fj_mass
    # ==================================================================================
    if remove_mass_pt_window:
        mask = (
            (spec_array[:, 0] > 40) & (spec_array[:, 0] < 200) &
            (spec_array[:, 1] > 300) & (spec_array[:, 1] < 2000)
        )
        feature_array = feature_array[mask]
        label_array = label_array[mask]
        spec_array = spec_array[mask]

    # ==================================================================================
    # Remove samples where no label is assigned (both QCD and H_bb are zero)
    # ==================================================================================
    label_mask = np.sum(label_array, axis=1) == 1
    feature_array = feature_array[label_mask]
    spec_array = spec_array[label_mask]
    label_array = label_array[label_mask]

    return feature_array, label_array, spec_array

# ======================================================================================
# Convert list of per-particle features into image using 2D histograms
# Used for CNN-based jet image models
# Inputs:
#   feature_array: shape = (n_samples, n_features, n_particles)
#   n_pixels: resolution of the image
#   img_ranges: eta/phi coordinate ranges (default = [-0.8, 0.8])
# Output:
#   img: shape = (n_samples, n_pixels, n_pixels, 1)
# ======================================================================================
def make_image(feature_array, n_pixels=224, img_ranges=[[-0.8, 0.8], [-0.8, 0.8]]):
    wgt = feature_array[:, 0]  # particle transverse momentum (ptrel)
    x = feature_array[:, 1]    # eta relative to jet axis
    y = feature_array[:, 2]    # phi relative to jet axis

    img = np.zeros(shape=(len(wgt), n_pixels, n_pixels))  # initialize image array

    # For each event, make 2D histogram of particle positions (eta, phi) weighted by pT
    for i in range(len(wgt)):
        hist2d, xedges, yedges = np.histogram2d(x[i], y[i],
                                                bins=[n_pixels, n_pixels],
                                                range=img_ranges,
                                                weights=wgt[i])
        img[i] = hist2d

    # Add channel dimension for CNN input
    return np.expand_dims(img, axis=-1)
