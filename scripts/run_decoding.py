import os.path as op
import numpy as np
# import pickle
import mne
from mne.decoding import GeneralizationAcrossTime

from meeg_preprocessing.utils import setup_provenance
from toolbox.jr_toolbox.utils import resample_epochs, decim

from scripts.config import (
    data_path,
    epochs_params,
    open_browser,
    subjects,
    preproc,
    analyses,
    results_dir,
    epochs_types
)
from p300.conditions import get_events


report, run_id, _, logger = setup_provenance(
    script=__file__, results_dir=results_dir)

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

    # preprocess data for memory issue
    if 'resample' in preproc.keys():
        epochs = resample_epochs(epochs, preproc['resample'])
    if 'decim' in preproc.keys():
        epochs = decim(epochs, preproc['decim'])
    if 'crop' in preproc.keys():
        epochs.crop(preproc['crop']['tmin'],
                    preproc['crop']['tmax'])

    # Apply each analysis
    for analysis in analyses:
        logger.info('%s: %s: %s: %s' % (subject, epoch_params['name'], epoch_type,
                                        analysis['name']))
        # check if nested analysis
        if isinstance(analysis['condition'], list):
            continue
        query, condition = analysis['query'], analysis['condition']
        sel = range(len(events)) if query is None \
            else events.query(query).index
        sel = [ii for ii in sel if ~np.isnan(events[condition][sel][ii])]
        y = np.array(events[condition], dtype=np.float32)

        print analysis['name'], np.unique(y[sel]), len(sel)

        if len(sel) == 0:
            logger.warning('%s: no epoch in %s %s for %s.' % (
                subject, epoch_params['name'], epoch_type, analysis['name']))
            continue

        # Apply analysis
        gat = GeneralizationAcrossTime(clf=analysis['clf'],
                                       cv=analysis['cv'],
                                       scorer=analysis['scorer'],
                                       n_jobs=-1)
        gat.fit(epochs[sel], y=y[sel])
        gat.score(epochs[sel], y=y[sel])

        # Save analysis
        # TODO !
        #
        # # Save classifier results
        # with open(pkl_fname, 'wb') as f:
        #     pickle.dump([gat, analysis, sel, events], f)

        # Plot
        fig = gat.plot_diagonal(show=False)
        report.add_figs_to_section(fig, '%s %s %s %s: (diagonal)' % (
            subject, epoch_params['name'], epoch_type, analysis['name']),
            analysis['name'])

        fig = gat.plot(vmin=np.min(gat.scores_),
                       vmax=np.max(gat.scores_), show=False)
        report.add_figs_to_section(fig, '%s %s %s %s: GAT' % (
            subject, epoch_params['name'], epoch_type, analysis['name']),
            analysis['name'])

report.save(open_browser=open_browser)
