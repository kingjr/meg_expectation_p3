import numpy as np
import matplotlib.pyplot as plt
import os.path as op
import mne
from meeg_preprocessing.utils import setup_provenance
# from mne.datasets import sample
from p300.conditions import get_events
from scripts.config import (data_path, epochs_params, subjects, epochs_types,
                            results_dir,open_browser)

report, run_id, results_dir, logger = setup_provenance(script=__file__,
                                                       results_dir=results_dir)

# data_path = sample.data_path()
# fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
# evoked = mne.read_evokeds(fname, condition='Left Auditory',
#                           baseline=(None, 0))
#
# # plot with bads
# evoked.plot(exclude=[])
# # compute interpolation (also works with Raw and Epochs objects)
# evoked.interpolate_bads(reset_bads=False)
# # plot interpolated (previous bads)
# evoked.plot(exclude=[])

epoch_params = epochs_params[0]
epoch_type = epochs_types[0]

for subject in subjects:

#def artefact_rej(name,subject,epochs):

    print(subject)
    # Extract events from mat file
    bhv_fname = op.join(data_path, 'behavior',
                        '{}_behaviorMEG.mat'.format(subject[-2:]))
    eptyp_name = epoch_params['name'] + epoch_type
    #eptyp_name=name
    # Get MEG data
    epo_fname = op.join(data_path, 'MEG', subject,
                        '{}-{}-epo.fif'.format(eptyp_name, subject))
    epochs = mne.read_epochs(epo_fname)

    epochs.crop(-.200, .600)

    # XXX probably move this correction before saving the epoch data
    sel_ch = np.where([ch == 'EEG064' for ch in epochs.ch_names])[0]
    if len(sel_ch):
        epochs.info['chs'][sel_ch[0]]['kind'] = 3
    # -------------------------------------------------------------

    # Get events specific to epoch definition (stim or motor lock)
    events = get_events(epochs.events)

    # Identify bad EEG channels
    # XXX Specific to Gabriela's data XXX I think this refers to EEG064, not this, right?
    epochs_eeg = epochs.pick_types(meg=False, eeg=True, copy=True)
    evoked_bad = epochs_eeg.average() #to plot before
    data = epochs_eeg._data
    chan_deviation = np.median(np.std(data, axis=2), axis=0)


    def mad(x, axis=None):
        center = np.median(x, axis=axis)
        return np.median(np.abs(x - center), axis=axis)

    threshold = np.median(chan_deviation) + 10 * mad(chan_deviation)
    ax.plot(chan_deviation)
    ax.axhline(threshold, color='r')
    report.add_figs_to_section(fig, ('%s (%s): Artifact thresholding' % (
         subject, eptyp_name)))
    bad_channels = np.where(chan_deviation > threshold)[0]
    #plt.show()

    # Interpolate
    epochs_eeg.info['bads'] += [epochs_eeg.ch_names[ch] for ch in bad_channels]
    epochs_eeg.interpolate_bads(reset_bads=False) #eeg only to plot
    evoked_eeg = epochs_eeg.average()

    epochs.interpolate_bads(reset_bads=False)

    # check
    # evoked_bad = epochs_eeg.average()
    # epochs_eeg.info['bads'] = [epochs_eeg.ch_names[ch] for ch in bad_channels]
    # epochs_eeg.interpolate_bads(reset_bads=False)
    # evoked_good = epochs_eeg.average()
    # evoked_bad.plot(show=False)
    # evoked_good.plot(show=False)
    # evoked_bad.plot_topomap(show=False)
    # evoked_good.plot_topomap()

    fig1 = evoked_bad.plot_topomap(show=False)
    report.add_figs_to_section(fig1, ('%s (%s): BEFORE artifact rejection' % (
        subject, eptyp_name)))
    fig2 = evoked_eeg.plot_topomap(show=False)
    report.add_figs_to_section(fig2, ('%s (%s): AFTER artifact rejection' % (
        subject, eptyp_name)))

    report.save(open_browser=open_browser)

    #return epochs

    # # Identify bad trials
    # # --- normalize all channel types
    # ch_types = [dict(meg=False, eeg=True),
    #             dict(meg='mag', eeg=False),
    #             dict(meg='grad', eeg=False)]
    # epochs_norm = epochs.copy()
    # for ch_type in ch_types:
    #     pick = mne.pick_types(epochs.info, **ch_type)
    #     data = epochs_norm._data[:, pick, :]
    #     ntrial, nchan, ntime = data.shape
    #     mean = np.median(np.median(data, axis=1), axis=0)
    #     std = mad(np.median(data, axis=1), axis=0)
    #     data -= np.tile(mean, [ntrial, nchan, 1])
    #     data /= np.tile(std, [ntrial, nchan, 1])
    #     epochs_norm._data[:, pick, :] = data
    #
    # # TODO eeg is downweighted
    # pick = mne.pick_types(epochs.info, eeg=True, meg=True)
    # data = epochs_norm._data
    # trial_deviation = np.median(np.std(data, axis=2), axis=1)
    # threshold = np.median(trial_deviation) + 10 * mad(trial_deviation)
    # plt.plot(trial_deviation)
    # plt.axhline(threshold, color='r')
    # plt.show()


    #
    # # TODO put this into a function and relaunch the mask
    # # subtraction and the basic univariate topographies
    # # Later: add the ICA correction: maybe for later, not now
