import matplotlib.pyplot as plt
import numpy as np
import mne
fname = '/media/jrking/INSERM/data/MEG/s23_pf120155/run_01_filt-0-35_sss_raw.fif'
raw = mne.io.Raw(fname, preload=True)
raw.pick_channels(['STI101'])

diff = raw._data[:, 1:] - raw._data[:, :-1]


onsets = np.where((diff > 0) & (diff < 64))[1]
offsets = np.where((diff < 0) & (diff > -64))[1]

durations = [offsets[i] - onsets[i] for i in range(len(onset))]

[np.sum(durations == i) for i in np.unique(durations)]
