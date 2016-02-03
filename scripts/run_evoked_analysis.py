import pickle
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import warnings

import mne
from meeg_preprocessing.utils import setup_provenance

from p300.conditions import get_events
from p300.analyses import analyses

from toolbox.jr_toolbox.utils import (meg_to_gradmag, share_clim)
from jr.stats import nested_analysis

from scripts.config import (
    data_path,
    subjects,
    baseline,
    results_dir,
    epochs_params,
    epochs_types,
    chan_types,
    open_browser,
)

report, run_id, results_dir, logger = setup_provenance(script=__file__,
                                                       results_dir=results_dir)

mne.set_log_level('INFO')

from itertools import product
error_log = list()
for subject, epochs_param, epochs_type in product(subjects, epochs_params,
                                                  epochs_types):

    eptyp_name = epochs_param['name'] + epochs_type
    print(subject, eptyp_name)

    # Get MEG data
    if baseline:
        epo_fname = op.join(data_path, 'MEG', subject,
                            '{}-{}-epo.fif'.format(eptyp_name, subject))
    elif not baseline:
        epo_fname = op.join(data_path, 'MEG', subject,
                            'nobl-{}-{}-epo.fif'.format(eptyp_name, subject))
    epochs = mne.read_epochs(epo_fname)
    # Get events specific to epoch definition (stim or motor lock)
    events = get_events(epochs.events)

    # Apply each analysis
    for analysis in analyses:
        try:
            coef, sub = nested_analysis(
                epochs._data, events, analysis['condition'],
                function=analysis.get('erf_function', None),
                query=analysis.get('query', None),
                single_trial=analysis.get('single_trial', False),
                y=analysis.get('y', None),
                n_jobs=1)
        # If there are 0 trials in one of the conditions, keep going anyways,
        # but throw an error
        except RuntimeError:
            error_msg = 'Problem with %s, %s!' % (subject, analysis['name'])
            warnings.warn(error_msg)
            error_log.append(error_msg)
            coef, sub = None, None
        evoked = epochs.average()
        evoked.data = coef

        # # Save all_evokeds
        # FIXME make paths in config to avoid dealing with file
        save_dir = op.join(data_path, 'MEG', subject, 'evokeds')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if baseline:
            pkl_fname = op.join(save_dir, '%s-cluster_sensors_%s.pickle' % (
                eptyp_name, analysis['name']))
        elif not baseline:
            pkl_fname = op.join(save_dir, 'nobl-%s-cluster_sensors_%s.pickle'
                                % (eptyp_name, analysis['name']))
        with open(pkl_fname, 'wb') as f:
            pickle.dump([evoked, sub, analysis], f)

        # Only plot if there are trials to be plotted
        if evoked.data is not None:
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
            report.add_figs_to_section(fig2, ('%s (%s) %s: CONDITIONS' % (
                subject, eptyp_name, analysis['name'])), analysis['name'])

if error_log:
    warnings.warn(error_log)

report.save(open_browser=open_browser)
