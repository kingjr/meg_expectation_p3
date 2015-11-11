import os.path as op
import numpy as np
import os
import pickle
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
    baseline,
    results_dir)

from p300.conditions import get_events
from p300.analyses import analyses

report, run_id, _, logger = setup_provenance(
    script=__file__, results_dir=results_dir)

from itertools import product

# for subject, epoch_params, epoch_type in product(subjects, epochs_params,
#                                                 epochs_types):
for subject, epoch_params in product(subjects, epochs_params):
    print(subject)
    # Only use unmasked data for now
    epoch_type = ''
    eptyp_name = epoch_params['name'] + epoch_type
    print(eptyp_name)

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
        logger.info('%s: %s: %s: %s' % (subject, epoch_params['name'],
                                        epoch_type, analysis['name']))
        # check if nested analysis
        if isinstance(analysis['condition'], list):
            continue
        query, condition = analysis['query'], analysis['condition']
        sel = range(len(events)) if query is None \
            else events.query(query).index
        # TODO change analyses so that nan gets thrown out in query, change this to be a warning if you see a nan here
        # sel = [ii for ii in sel if ~np.isnan(events[condition][sel[ii]])]
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
        save_dir = op.join(data_path, 'MEG', subject, 'decoding')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if baseline:
            pkl_fname = op.join(save_dir, '%s-gat_scores_%s.pickle' % (
                eptyp_name, analysis['name']))
        elif not baseline:
            pkl_fname = op.join(save_dir, 'nobl_%s-gat_scores_%s.pickle' % (
                eptyp_name, analysis['name']))
        # Save classifier results
        with open(pkl_fname, 'wb') as f:
            # pickle.dump([gat, analysis, sel, events], f)
            pickle.dump([gat.scores_, analysis, sel, events], f)
        # TODO : Just save gat.scores for now, but need to also save gat

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
