import copy
import warnings
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

import mne
from mne.io import Raw
from mne.io.pick import _picks_by_type as picks_by_type
from mne.preprocessing import read_ica
from mne.viz import plot_drop_log

from meeg_preprocessing.utils import setup_provenance, set_eog_ecg_channels

from p300.conditions import get_events

from toolbox.jr_toolbox.utils import find_in_df

from config import (
    data_path,
    subjects,
    results_dir,
    events_fname_filt_tmp,
    epochs_contrasts,
    open_browser,
    contrasts,
    chan_types,
)

# force separation of magnetometers and gradiometers
if 'meg' in [i['name'] for i in chan_types]:
    chan_types = [dict(name='mag'), dict(name='grad')] + \
                 [dict(name=i) for i in chan_types if i != 'meg']

report, run_id, results_dir, logger = setup_provenance(script=__file__,
                                                       results_dir=results_dir)

mne.set_log_level('INFO')

for subject in subjects:
    # Extract events from mat file
    bhv_fname = op.join(data_path, 'behavior',
                        '{}_behaviorMEG.mat'.format(subject[-2:]))


    # Apply contrast on each type of epoch
    all_epochs = [[]] * len(epochs_contrasts)
    for epoch_params in epochs_contrasts:
        ep_name = epoch_params['name']

        # Get MEG data
        epo_fname = op.join(data_path, 'MEG', subject,
                            '{}-{}-epo.fif'.format(ep_name, subject))
        epochs = mne.read_epochs(epo_fname)
        # Get events specific to epoch definition (stim or motor lock)
        events = get_events(epochs.events)

        # name of evoked file
        ave_fname = op.join(data_path, 'MEG', subject,
                            '{}-{}-contrasts-ave.fif'.format(ep_name, subject))

        # Apply each contrast
        all_evokeds = list()
        for contrast in contrasts:
            evokeds = list()

            # Find trials
            sel = find_in_df(events, contrast['include'], contrast['exclude'])
            # Select condition
            key = contrast['include'].keys()[0]
            for value in contrast['include'][key]:
                # Find included trials
                subsel = [sel[i] for i in np.where(events[key][sel]==value)[0]]
                # Evoked data
                evoked = epochs[subsel].average()
                evoked.comment = contrast['name']+str(value)
                # keep for contrast
                evokeds.append(evoked)
                # keep for saving
                all_evokeds.append(evoked)

            # Apply contrast
            if np.min([e.nave for e in evokeds]) == 0:
                warnings.warn('%s: no epochs in %s for %s' % (subject, ep_name,
                                                              contrast['name']))
                continue

            avg = copy.deepcopy(evokeds[0])
            avg.data *= 0.
            # if only two condition, only plot the difference between the two
            if len(evokeds) == 2:
                avg.data = evokeds[0].data
                evokeds = [evokeds[1]]
            else:
                # else plot all conditions - average across conditions
                for e in evokeds:
                    avg.data = avg.data + e.data / len(evokeds)

            # Plot
            fig, ax = plt.subplots(len(evokeds), len(chan_types))
            ax = np.reshape(ax, len(evokeds) * len(chan_types))
            for e, evoked in enumerate(evokeds):
                for ch, chan_type in enumerate(chan_types):
                    delta = copy.deepcopy(evoked) - copy.deepcopy(avg)
                    picks = [i for k, p in picks_by_type(evoked.info) for i in p
                                                        if k in chan_type['name']]
                    mM = abs(np.percentile(delta.data[picks,:], 99.))
                    ax_ind = e * len(chan_types)+ ch
                    ax[ax_ind].imshow(delta.data[picks,:], vmin=-mM, vmax=mM,
                                      interpolation='none', aspect='auto',
                                      cmap='RdBu_r', extent=[min(delta.times),
                                               max(delta.times), 0, len(picks)]
                                       )
                    ax[ax_ind].plot([0, 0], [0, len(picks)], color='black')
                    ax[ax_ind].set_title(chan_type['name'] + ': ' +
                                         str(contrast['include'][key][e]))
                    ax[ax_ind].set_xlabel('Time')
                    ax[ax_ind].set_adjustable('box-forced')
                    ax[ax_ind].autoscale(False)

            report.add_figs_to_section(fig, ('%s (%s) %s'
                % (subject, ep_name, contrast['name'])), contrast['name'])

            # Topo
            plot_times = np.linspace(avg.times.min(), avg.times.max(), 10)
            delta = evokeds[-1] - avg
            fig = delta.plot_topomap(plot_times, chan_type='mag', sensors=False,
                                    contours=False)
            report.add_figs_to_section(fig, ('%s (%s) %s: (topo)'
                % (subject, ep_name, contrast['name'])), contrast['name'])

        # Save all_evokeds
        mne.write_evokeds(ave_fname, all_evokeds)

report.save(open_browser=open_browser)
