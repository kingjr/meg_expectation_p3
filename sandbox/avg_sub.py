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
    open_browser)
from p300.analyses import analyses

#report, run_id, results_dir, logger = setup_provenance(script=__file__,
#                                                       results_dir=results_dir)


epochs_type = epochs_types[1]
epochs_param = epochs_params[0]

for analysis in analyses:
    print(analysis['name'])
    evoked_list = list()
    subevoked_list = list()
    for subject in subjects:
        eptyp_name = epochs_param['name'] + epochs_type

        # Load evokeds
        # FIXME make paths in config to avoid dealing with file
        load_dir = op.join(data_path, 'MEG', subject, 'evokeds')
        pkl_fname = op.join(load_dir, '%s-cluster_sensors_%s.pickle' % (
            eptyp_name, analysis['name']))
        with open(pkl_fname, 'rb') as f:
            evoked, sub, analysis = pickle.load(f)
        evoked.data = np.mean(sub['X'], axis=0)
        picks = pick_types(evoked.info, meg=True, eeg=False, misc=False, stim=False)
        evoked.pick_types(picks)
        evoked_list.append(evoked)
        subevoked_list.append(sub['X'][:, picks, :])
    # mean coefficient value across subjects
    evoked.data = np.mean([e.data for e in evoked_list], axis=0)
    fig0 = evoked.plot_topomap(times = np.linspace(0,.600,20))
    # plot each sub condition
    sub_evokeds = np.mean(subevoked_list, axis=0)  # mean across subjects
    for sub in sub_evokeds:
        evoked.data = sub
        evoked.plot(show=False)
    plt.show()
    # manual plot
    from jr.plot import plot_sem, pretty_plot
    chan = 102
    cmap = plt.get_cmap('RdBu_r')
    colors = cmap(np.linspace(0, 1, len(sub_evokeds)))
    fig, ax = plt.subplots(1, 1, figsize=[8, 2])
    subevoked_list = np.array(subevoked_list)
    for sub, color in enumerate(colors):
        plot_sem(evoked.times, subevoked_list[:, sub, chan, :], color=color)
        ax.axvline(0, color='dimgray')
    pretty_plot(ax)
    plt.show()
    report.add_figs_to_section(fig0, ('%s: %s' % (eptyp_name, analysis['name'])))

## TODO: Save these?

report.save(open_browser=open_browser)
