import numpy as np
import matplotlib.pyplot as plt
import os.path as op
import mne
from mne.io import set_eeg_reference
from meeg_preprocessing.utils import setup_provenance
# from mne.datasets import sample
from p300.conditions import get_events
from scripts.config import (data_path, epochs_params, epochs_types,
                            results_dir,open_browser)
from jr.stats import robust_mean

report, run_id, results_dir, logger = setup_provenance(script=__file__,
                                                       results_dir=results_dir)

def artefact_rej(eptyp_name,subject,epochs):
    print('Artefact Rejection')
    epochs_ = epochs.copy()
    # Only define artefact rejection based on this shorter epoch
    # (but keep the whole trial)
    epochs_.crop(-.200, .600)

    # XXX Specific to Gabriela's data
    # XXX probably move this correction before saving the epoch data
    sel_ch = np.where([ch == 'EEG064' for ch in epochs_.ch_names])[0]
    if len(sel_ch):
        epochs_.info['chs'][sel_ch[0]]['kind'] = 3
    # -------------------------------------------------------------

    # Get events specific to epoch definition (stim or motor lock)
    events = get_events(epochs_.events)

    # Identify bad EEG channels
    epochs_eeg = epochs_.pick_types(meg=False, eeg=True, copy=True)
    data = epochs_eeg._data
    chan_deviation = np.median(np.std(data, axis=2), axis=0)


    def mad(x, axis=None):
        center = np.median(x, axis=axis)
        return np.median(np.abs(x - center), axis=axis)

    # Define the threshold, find bad channels
    chan_threshold = np.median(chan_deviation) + 10 * mad(chan_deviation)
    bad_channels = np.where(chan_deviation > chan_threshold)[0]

    # Correct uncropped data
    epochs.info['bads'] += [epochs_eeg.ch_names[ch] for ch in bad_channels]
    epochs.interpolate_bads(reset_bads=False)

    #  Average rereference for EEG data
    # TODO HOW DO I DO THIS FOR EEG ONLY??
    from mne import pick_types
    picks = pick_types(epochs.info, eeg=True, meg=False)
    data = epochs._data[:, picks, :]
    # mean across channel
    #ref = np.mean(epochs._data, axis=1)
    ref = np.mean(data, axis=1)
    data -= np.tile(ref, [len(picks),1, 1]).transpose(1,0,2)  # probably needs a transpose here
    epochs._data[:, picks, :] = data

    # Identify bad trials
    # --- normalize all channel types
    ch_types = [dict(meg=False, eeg=True),
                dict(meg='mag', eeg=False),
                dict(meg='grad', eeg=False)]
    epochs_norm = epochs.copy()
    epochs_norm.crop(-.200, .600)
    for ch_type in ch_types:
        pick = mne.pick_types(epochs_.info, **ch_type)
        data = epochs_norm._data[:, pick, :]
        ntrial, nchan, ntime = data.shape
        mean = np.median(np.median(data, axis=1), axis=0)
        std = mad(np.median(data, axis=1), axis=0)
        data -= np.tile(mean, [ntrial, nchan, 1])
        data /= np.tile(std, [ntrial, nchan, 1])
        epochs_norm._data[:, pick, :] = data

    # TODO eeg is downweighted
    pick = mne.pick_types(epochs_.info, eeg=True, meg=True)
    data = epochs_norm._data
    trial_deviation = np.median(np.std(data, axis=2), axis=1)
    trial_threshold = np.median(trial_deviation) + 10 * mad(trial_deviation)

    # Find the good trials, only keep those
    good_trials = np.where(trial_deviation < trial_threshold)[0]
    epochs=epochs[good_trials]

    # Plot

    #TODO need to change to double check these plots, add to return statement?

    # Channel rejection check
    # fig, ax = plt.subplots(1)
    # ax.plot(chan_deviation)
    # ax.axhline(chan_threshold, color='r')
    # report.add_figs_to_section(fig, ('%s (%s): Artifact thresholding' % (
    #      subject, eptyp_name)))

    # Topo from before channel rejection
    # evoked_bad.data = robust_mean(epochs_eeg._data, axis = 0)
    # fig1 = evoked_bad.plot_topomap(show=False)
    # report.add_figs_to_section(fig1, ('%s (%s): BEFORE artifact rejection' % (
    #     subject, eptyp_name)))

    # Topo after channel rejection
    # Interpolate EEG only (just for plotting)
    # epochs_eeg.info['bads'] += [epochs_eeg.ch_names[ch] for ch in bad_channels]
    # epochs_eeg.interpolate_bads(reset_bads=False)
    # evoked_good = robust_mean(epochs_eeg._data, axis=0)
    # fig2 = evoked_good.plot_topomap(show=False)
    # report.add_figs_to_section(fig2, ('%s (%s): AFTER artifact rejection' % (
    #     subject, eptyp_name)))

    # Trial rejection check
    # fig3, ax = plt.subplots(1)
    # ax.plot(trial_deviation)
    # ax.axhline(trial_threshold, color='r')
    # report.add_figs_to_section(fig3, ('%s (%s): Trial Rejection' % (
    #      subject, eptyp_name)))

    return epochs  # [fig1, fig2, ]
