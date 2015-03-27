import os.path as op
import numpy as np
import mne
from mne.io import Raw
from mne.io.pick import _picks_by_type as picks_by_type
from mne.preprocessing import read_ica
from mne.viz import plot_drop_log
from mne.decoding import GeneralizationAcrossTime
from toolbox.jr_toolbox.utils import resample_epochs, decim

import pickle
import warnings

from p300.conditions import get_events

from meeg_preprocessing.utils import setup_provenance, set_eog_ecg_channels

from scripts.config import (
    data_path,
    subjects,
    results_dir,
    events_fname_filt_tmp,
    epochs_params,
    open_browser,
    contrasts,
    decoding_preproc,
    decoding_params
)


report, run_id, results_dir, logger = setup_provenance(script=__file__,
                                                       results_dir=results_dir)

mne.set_log_level('INFO')

for subject in subjects:

    # Extract events from mat file
    bhv_fname = op.join(data_path, 'behavior',
                        '{}_behaviorMEG.mat'.format(subject[-2:]))


    # Apply contrast on each type of epoch
    all_epochs = [[]] * len(epochs_params)
    for epoch_params, preproc in zip(epochs_params, decoding_preproc):
        ep_name = epoch_params['name']

        # Get MEG data
        epo_fname = op.join(data_path, 'MEG', subject,
                            '{}-{}-epo.fif'.format(ep_name, subject))
        epochs = mne.read_epochs(epo_fname)
        events = get_events(epochs.events)

        # preprocess data for memory issue
        if 'resample' in preproc.keys():
            epochs = resample_epochs(epochs, preproc['resample'])
        if 'decim' in preproc.keys():
            epochs = decim(epochs, preproc['decim'])
        if 'crop' in preproc.keys():
            epochs.crop(preproc['crop']['tmin'],
                        preproc['crop']['tmax'])

        # Apply each contrast
        for contrast in contrasts:
            # Find excluded trials
            exclude = np.any([events[x['cond']]==ii
                                for x in contrast['exclude']
                                    for ii in x['values']],
                            axis=0)

            # Select condition
            include = list()
            cond_name = contrast['include']['cond']
            for value in contrast['include']['values']:
                # Find included trials
                include.append(events[cond_name]==value)
            sel = np.any(include,axis=0) * (exclude==False)
            sel = np.where(sel)[0]

            # reduce number or trials if too many
            if len(sel) > 400:
                import random
                random.shuffle(sel)
                sel = sel[0:400]

            if len(sel) == 0:
                warnings.warn('%s: no epoch in %s for %s.' % (subject, ep_name,
                                                              contrast['name']))
                continue

            y = np.array(events[cond_name].tolist())

            # Apply contrast
            gat = GeneralizationAcrossTime(**decoding_params)
            # gat = GeneralizationAcrossTime(n_jobs=-1)
            gat.fit(epochs[sel], y=y[sel])
            gat.score(epochs[sel], y=y[sel])

            # Plot
            fig = gat.plot_diagonal(show=False)
            report.add_figs_to_section(fig, ('%s %s: %s'
                % (subject, ep_name, contrast['name'])), subject)

            fig = gat.plot(show=False)
            report.add_figs_to_section(fig, ('%s %s: %s'
                % (subject, ep_name, contrast['name'])), subject)

            # Save contrast
            pkl_fname = op.join(data_path, 'MEG', subject,
                                '%s-%s-decod_%s.pickle' % (ep_name, subject,
                                contrast['name']))

            with open(pkl_fname, 'w') as f:
                pickle.dump([gat, contrast], f)

report.save(open_browser=open_browser)
