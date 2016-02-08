import pickle
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

report, run_id, results_dir, logger = setup_provenance(script=__file__,
                                                       results_dir=results_dir)


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
        evoked.pick_types(meg=True, eeg=False, misc=False, stim=False)
        # evoked.pick_types(picks)
        evoked_list.append(evoked)
        picks = mne.pick_types(evoked.info, meg=True, eeg=False,
                               misc=False, stim=False)
        subevoked_list.append(sub['X'][:, picks, :])
        # subevoked_list.append(sub['X'])

    # mean coefficient value across subjects
    evoked.data = np.mean([e.data for e in evoked_list], axis=0)
    # fig0 = evoked.plot_topomap(times=np.linspace(0, .600, 20))
    # plot each sub condition
    sub_evokeds = np.mean(subevoked_list, axis=0)  # mean across subjects
    chan_mag = mne.pick_types(evoked.info, meg='mag')
    fig, ax = plt.subplots(1, 1, figsize=[8, 2])
    cmap = plt.get_cmap('RdBu_r')
    colors = cmap(np.linspace(0, 1, len(sub_evokeds)))
    for sub, color in zip(sub_evokeds, colors):
        evoked.data = sub
        ax.plot(evoked.times, np.std(sub[chan_mag, :], axis=0))  # ~GFP
        ax.set_ybound(lower=0e-14, upper=6e-14)
        # evoked.plot_image(show=False)
        # evoked.plot_joint(times=[.100, .200, .300], show=False)

    plt.show()

    # differences between sub conditions
    evoked.data = sub_evokeds[1]-sub_evokeds[0]
    # diff = sub_evokeds[1][chan_mag, :]-sub_evokeds[0][chan_mag, :]
    # ax.plot(evoked.times, np.std(diff, axis=0), color=color)  # ~GFP
    evoked.plot_topomap(show=False)
    evoked.plot_joint(times=np.linspace(0, .500, 10), ts_args=dict(gfp=True))
    plt.show()

    # average of sub conditions
    evoked.data = np.mean(sub_evokeds, axis=0)
    evoked.plot_topomap(show=False, times=np.linspace(0, .600, 20))
    plt.show()

    # manual plot
    from jr.plot import plot_sem, pretty_plot
    chan = 182
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
