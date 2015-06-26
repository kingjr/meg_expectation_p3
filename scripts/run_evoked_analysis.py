import pickle
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
from meeg_preprocessing.utils import setup_provenance

from p300.conditions import get_events

from toolbox.jr_toolbox.utils import (nested_analysis, meg_to_gradmag,
                                      share_clim)

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

from itertools import product

for subject, epoch_params, epoch_type in product(subjects, epochs_params,
                                                 epochs_types):
    # Extract events from mat file
    bhv_fname = op.join(data_path, 'behavior',
                        '{}_behaviorMEG.mat'.format(subject[-2:]))
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
        coef, sub = nested_analysis(
            epochs._data, events, analysis['condition'],
            function=analysis.get('erf_function', None),
            query=analysis.get('query', None),
            single_trial=analysis.get('single_trial', False),
            y=analysis.get('y', None),
            n_jobs=1)
        evoked = epochs.average()
        evoked.data = coef

        # Save all_evokeds
        # FIXME make paths in config to avoid dealing with file
        save_dir = op.join(data_path, 'MEG', subject, 'evokeds')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pkl_fname = op.join(save_dir, '%s-cluster_sensors_%s.pickle' % (
            eptyp_name, analysis['name']))
        with open(pkl_fname, 'wb') as f:
            pickle.dump([evoked, sub, analysis], f)

        # Plot coef
        fig0 = evoked.plot_topomap(show=False)
        share_clim(fig0.get_axes())
        # FIXME: add loop across sensors for topo

        report.add_figs_to_section(fig0, ('%s (%s) %s: COEF topo' % (
            subject, eptyp_name, analysis['name'])), analysis['name'])

        fig1 = evoked.plot_image(show=False)
        report.add_figs_to_section(fig1, ('%s (%s) %s: COEF' % (
            subject, eptyp_name, analysis['name'])), analysis['name'])

        # Plot subcondition
        fig2, ax2 = plt.subplots(len(meg_to_gradmag(chan_types)),
                                 len(sub['X']), figsize=[19, 10])
        X_mean = np.mean([X for X in sub['X']], axis=0)
        for e, (X, y) in enumerate(zip(sub['X'], sub['y'])):
            evoked.data = X - X_mean
            evoked.plot_image(axes=ax2[:, e], show=False,
                              titles=dict(grad='grad (%.2f)' % y,
                                          mag='mag (%.2s)' % y))
        for chan_type in range(len(meg_to_gradmag(chan_types))):
            share_clim(ax2[chan_type, :])
        print(analysis['name'])
        report.add_figs_to_section(fig1, ('%s (%s) %s: CONDITIONS' % (
            subject, eptyp_name, analysis['name'])), analysis['name'])

report.save(open_browser=open_browser)
