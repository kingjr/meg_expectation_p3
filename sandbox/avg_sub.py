import pickle
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import mne
from meeg_preprocessing.utils import setup_provenance
from scripts.config import (
    data_path, epochs_types, epochs_params,
    subjects,
    results_dir,
    analyses,
    open_browser)

report, run_id, results_dir, logger = setup_provenance(script=__file__,
                                                       results_dir=results_dir)


epoch_types = epochs_types[1]
epoch_params = epochs_params[0]

for analysis in analyses:
    print(analysis['name'])
    evoked_list = list()
    for subject in subjects:
        eptyp_name = epoch_params['name'] + epoch_types

        # Load evokeds
        # FIXME make paths in config to avoid dealing with file
        save_dir = op.join(data_path, 'MEG', subject, 'evokeds')
        pkl_fname = op.join(save_dir, '%s-cluster_sensors_%s.pickle' % (
            eptyp_name, analysis['name']))
        with open(pkl_fname, 'rb') as f:
            evoked, sub, analysis = pickle.load(f)
        evoked.data = np.mean(sub['X'], axis=0)
        evoked.pick_types(meg=True, eeg=False, copy=False)
        evoked_list.append(evoked)

    evoked.data  =np.mean([e.data for e in evoked_list], axis=0)
    fig0 = evoked.plot_topomap(times = np.linspace(0,.600,20))
    report.add_figs_to_section(fig0, ('%s: %s' % (eptyp_name, analysis['name'])))

## TODO: Save these?

report.save(open_browser=open_browser)
