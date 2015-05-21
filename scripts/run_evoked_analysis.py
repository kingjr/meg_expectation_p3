import pickle
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.io.pick import _picks_by_type as picks_by_type
from meeg_preprocessing.utils import setup_provenance

from p300.conditions import get_events

from toolbox.jr_toolbox.utils import build_analysis

from config import (
    data_path,
    subjects,
    results_dir,
    epochs_params,
    epochs_types,
    analyses,
    chan_types,
    open_browser,
)

report, run_id, results_dir, logger = setup_provenance(script=__file__,
                                                       results_dir=results_dir)

mne.set_log_level('INFO')

# force separation of magnetometers and gradiometers
if 'meg' in [i['name'] for i in chan_types]:
    chan_types = [dict(name='mag'), dict(name='grad')] + \
                 [dict(name=i['name'])
                  for i in chan_types
                  if i['name'] != 'meg']

for subject in subjects:
    print(subject)
    # Extract events from mat file
    bhv_fname = op.join(data_path, 'behavior',
                        '{}_behaviorMEG.mat'.format(subject[-2:]))

    # Apply analysis on each type of epoch
    all_epochs = [[]] * len(epochs_params)
    for epoch_params in epochs_params:
        for epoch_type in epochs_types:
            eptyp_name = epoch_params['name'] + epoch_type
            print(eptyp_name)

            # Get MEG data
            epo_fname = op.join(data_path, 'MEG', subject,
                                '{}-{}-epo.fif'.format(eptyp_name, subject))
            epochs = mne.read_epochs(epo_fname)
            # Get events specific to epoch definition (stim or motor lock)
            events = get_events(epochs.events)

            # Apply each analysis
            for analysis in analyses:
                print(analysis['name'])
                coef, evokeds = build_analysis(
                    analysis['conditions'], epochs, events,
                    operator=analysis['operator'])

                # Prepare plot delta (subtraction, or regression)
                fig1, ax1 = plt.subplots(1, len(chan_types))
                if type(ax1) not in [list, np.ndarray]:
                    ax1 = [ax1]
                # Prepare plot all conditions at top level of analysis
                fig2, ax2 = plt.subplots(len(evokeds['coef']), len(chan_types))
                ax2 = np.reshape(ax2, len(evokeds['coef']) * len(chan_types))

                # Save all_evokeds
                save_dir = op.join(data_path, 'MEG', subject, 'evokeds')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                pkl_fname = op.join(save_dir, '%s-cluster_sensors_%s.pickle' % (
                    eptyp_name, analysis['name']))

                with open(pkl_fname, 'wb') as f:
                    pickle.dump([coef, evokeds, analysis, events], f)

                # Loop across channels
                for ch, chan_type in enumerate(chan_types):
                    # Select specific types of sensor
                    info = coef.info
                    picks = [i for k, p in picks_by_type(info)
                             for i in p if k in chan_type['name']]
                    # ---------------------------------------------------------
                    # Plot coef (subtraction, or regression)
                    # adjust color scale
                    mM = np.percentile(np.abs(coef.data[picks, :]), 99.)

                    # plot mean sensors x time
                    ax1[ch].imshow(coef.data[picks, :], vmin=-mM, vmax=mM,
                                   interpolation='none', aspect='auto',
                                   cmap='RdBu_r', extent=[min(coef.times),
                                   max(coef.times), 0, len(picks)])
                    # add t0
                    ax1[ch].plot([0, 0], [0, len(picks)], color='black')
                    ax1[ch].set_title(chan_type['name'] + ': ' + coef.comment)
                    ax1[ch].set_xlabel('Time')
                    ax1[ch].set_adjustable('box-forced')

                    # ---------------------------------------------------------
                    # Plot all conditions at top level of analysis
                    # XXX only works for +:- data
                    mM = np.median([np.percentile(abs(e.data[picks, :]), 80.)
                                    for e in evokeds['coef']])

                    for e, evoked in enumerate(evokeds['coef']):
                        ax_ind = e * len(chan_types) + ch
                        ax2[ax_ind].imshow(evoked.data[picks, :], vmin=-mM,
                                           vmax=mM, interpolation='none',
                                           aspect='auto', cmap='RdBu_r',
                                           extent=[min(coef.times),
                                           max(evoked.times), 0, len(picks)])
                        ax2[ax_ind].plot([0, 0], [0, len(picks)], color='k')
                        ax2[ax_ind].set_title(chan_type['name'] + ': ' +
                                              evoked.comment)
                        ax2[ax_ind].set_xlabel('Time')
                        ax2[ax_ind].set_adjustable('box-forced')

                # Save figure
                report.add_figs_to_section(fig1, ('%s (%s) %s: COEF' % (
                    subject, eptyp_name, analysis['name'])), analysis['name'])

                report.add_figs_to_section(fig2, ('%s (%s) %s: CONDITIONS' % (
                    subject, eptyp_name, analysis['name'])), analysis['name'])


report.save(open_browser=open_browser)
