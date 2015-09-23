# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import mkl
import sys
import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.io import Raw
from mne.io.pick import _picks_by_type as picks_by_type # XXX users should not need this
from mne.preprocessing import read_ica
from mne.viz import plot_drop_log

from meeg_preprocessing.utils import setup_provenance, set_eog_ecg_channels

from p300.conditions import get_events, events_select_condition

from scripts.wip_artefacts import artefact_rej

from scripts.config import (
    data_path,
    subjects,
    runs,
    results_dir,
    raw_fname_filt_tmp,
    eog_ch,
    ecg_ch,
    events_fname_filt_tmp,
    chan_types,
    epochs_params,
    epochs_types,
    use_ica,
    open_browser
)

report, run_id, results_dir, logger = setup_provenance(
    script=__file__, results_dir=results_dir)


mne.set_log_level('INFO')

if len(sys.argv) > 1:
    subjects = [sys.argv[1]]
    mkl.set_num_threads(1)

for subject in subjects:
    print(subject)
    this_path = op.join(data_path, 'MEG', subject)
    all_epochs = [list(), list()] # XXX should be generic to take len(epochs_params)

    icas = list()
    if use_ica is True:
        for chan_type in chan_types:
            icas.append(read_ica(
                op.join(this_path, '{}-ica.fif'.format(chan_type['name']))))
    for r in runs:
        fname = op.join(this_path, raw_fname_filt_tmp.format(r))
        if not op.isfile(fname):
            logger.info('Could not find %s. Skipping' % fname)
            continue

        raw = Raw(fname)

        # Set ECG EOG channels XXX again?
        set_eog_ecg_channels(raw, eog_ch=eog_ch, ecg_ch=ecg_ch)

        # # Interpolate bad channels
        # if 'eeg' in [i['name'] for i in chan_types] and len(raw.info['bads']) > 0:
        #     raw.interpolate_bad_channels()

        # Get events identified in run_extract_events
        events = mne.read_events(
            op.join(this_path, events_fname_filt_tmp.format(r)))

        # Epoch data for each epoch type
        for ep, epochs_list in zip(epochs_params, all_epochs):
            # Select events
            sel = events_select_condition(events[:,2], ep['events'])
            events_sel = events[sel,:]

            # Only keep parameters applicable to mne.Epochs() XXX CLEAN UP JR!
            ep_epochs = {key:v for key, v in ep.items() if key in ['event_id',
                                                               'tmin', 'tmax',
                                                               'baseline',
                                                               'reject',
                                                               'decim']}
            # Epoch raw data
            epochs = mne.Epochs(raw=raw, preload=True, events=events_sel,
                                **ep_epochs)

            # Redefine t0 if necessary
            if 'time_shift' in ep.keys():
                epochs.times += ep['time_shift']
                epochs.tmin += ep['time_shift']
                epochs.tmax += ep['time_shift']

            # ICA correction
            if use_ica is True:
                for ica in icas:
                    ica.apply(epochs)

            # Append runs
            epochs_list.append(epochs)

    for name, epochs_list in zip([ep['name'] for ep in epochs_params], all_epochs):
        # Concatenate runs
        epochs = mne.epochs.concatenate_epochs(epochs_list)

        # Artefact rejection (EEG channels then trials)
        epochs = artefact_rej(name,subject,epochs)

        # Save
        epochs.save(op.join(this_path, '{}-{}-epo.fif'.format(name, subject)))

        # Plot
        #-- % dropped
        report.add_figs_to_section(
            plot_drop_log(epochs.drop_log), '%s (%s): total dropped'
                          % (subject, name), 'Drop: ' + name)
        #-- % trigger channel
        ch = mne.pick_channels(epochs.ch_names, ['STI101'])
        epochs._data[:, ch,:] = (2.*(epochs._data[:,ch,:] > 1) +
                                 1.*(epochs._data[:,ch,:] < 8192))

        # fig = mne.viz.plot_image_epochs(epochs, ch, scalings=dict(stim=1),
        #                                 units=dict(stim=''), vmin=1, vmax=3,
        #                                 show=False)
        # report.add_figs_to_section(fig, '%s (%s): triggers' % (subject, name),
        #                            'Triggers: ' + name)

        #-- % evoked
        evoked = epochs.average()
        fig = evoked.plot(show=False)
        report.add_figs_to_section(fig, '%s (%s): butterfly' % (subject, name),
                                   'Butterfly: ' + name)
        times = np.arange(epochs.tmin, epochs.tmax,
                          (epochs.tmax - epochs.tmin) / 20)
        for ch_type in chan_types:
            ch_type = ch_type['name']
            if ch_type in ['eeg', 'meg', 'mag']:
                if ch_type == 'meg':
                    ch_type = 'mag'
            fig = evoked.plot_topomap(times, ch_type=ch_type)
            report.add_figs_to_section(
                fig, '%s (%s): topo %s' % (subject, name, ch_type),
                                        'Topo: ' + name)

report.save(open_browser=open_browser)
