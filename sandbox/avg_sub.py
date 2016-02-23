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
from jr.plot import plot_sem, pretty_plot

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
    # evoked.data = np.mean([e.data for e in evoked_list], axis=0)
    # fig0 = evoked.plot_topomap(times=np.linspace(-.500, .700, 19))

    # PLOT
    # plot each sub condition using PLOT
    sub_evokeds = np.mean(subevoked_list, axis=0)  # mean across subjects
    chan_mag = mne.pick_types(evoked.info, meg='mag')
    cmap = plt.get_cmap('RdBu_r')
    colors = cmap(np.linspace(0, 1, len(sub_evokeds)))
    subevoked_list = np.array(subevoked_list)

    if analysis['name']=='abs_soa':
        # get rid of absent condition, correct colors so not white at 50 ms
        colors = cmap(np.linspace(0, 1, len(sub_evokeds)-1)+.1)
        soa_subevoked_list = subevoked_list[:,1:6,:,:]
        fig, ax = plt.subplots(1, 1, figsize=[8, 2])

        plt.figure(facecolor="white")
        for sub, color in enumerate(colors):
            # subjects, conditions, sensors, times
            subevoked = soa_subevoked_list[:, sub, chan_mag, :]
            gfp = np.std(subevoked, axis=1)
            plot_sem(evoked.times, gfp, color=color)
            ax.axvline(0, color='dimgray')
        pretty_plot(ax)
        plt.title('SOA')
        plt.legend(['17 ms', '33 ms', '50 ms', '67 ms', '83 ms'])
    if analysis['name']=='pas_pst':
        colors = cmap(np.linspace(0, 1, len(sub_evokeds)))
        plt.figure(facecolor="white")
        for sub, color in enumerate(colors):
            # subjects, conditions, sensors, times
            subevoked = subevoked_list[:, sub, chan_mag, :]
            gfp = np.std(subevoked, axis=1)
            plot_sem(evoked.times, gfp, color=color)
            ax.axvline(0, color='dimgray')
        pretty_plot(ax)
        plt.title('PAS (target-present trials only)')
        plt.legend(['PAS 0', 'PAS 1', 'PAS 2', 'PAS 3'])
        plt.show()
    if analysis['name']=='local':
        fig, ax = plt.subplots(1, 1)
        colors = cmap(np.linspace(0, 1, len(sub_evokeds)))
        plt.figure(facecolor="white")
        for sub, color in enumerate(colors):
            # subjects, conditions, sensors, times
            subevoked = subevoked_list[:, sub, chan_mag, :]
            gfp = np.std(subevoked, axis=1)
            plot_sem(evoked.times, gfp, color=color)
            ax.axvline(0, color='dimgray')
        pretty_plot(ax)
        plt.title('Local Context (middle SOAs)')
        plt.legend(['Local Unseen', 'Local Seen'])
        plt.show()


    # differences between sub conditions
    # evoked.data = sub_evokeds[1]-sub_evokeds[0]
    # diff = sub_evokeds[1][chan_mag, :]-sub_evokeds[0][chan_mag, :]
    # ax.plot(evoked.times, np.std(diff, axis=0), color=color)  # ~GFP
    # evoked.plot_topomap(show=False)
    # evoked.plot_joint(times=np.linspace(0, .400, 14), title=analysis['name'],
    #                   ts_args=dict(gfp=True))
    # plt.show()

    # average of sub conditions
    # evoked.data = np.mean(sub_evokeds, axis=0)
    # evoked.plot_topomap(show=False, times=np.linspace(0, .600, 20))
    # plt.show()

## TODO: Save these?

report.save(open_browser=open_browser)
