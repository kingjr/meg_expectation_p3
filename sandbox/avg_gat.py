import os.path as op
import pickle
import numpy as np
from meeg_preprocessing.utils import setup_provenance
from scripts.config import (
    data_path,
    results_dir,
    epochs_types,
    epochs_params,
    subjects,
    baseline,
    open_browser)
from p300.analyses import analyses
from jr.plot import pretty_gat, pretty_decod

report, run_id, results_dir, logger = setup_provenance(script=__file__,
                                                       results_dir=results_dir)

epoch_types = epochs_types[0]
epoch_params = epochs_params[0]
eptyp_name = epoch_params['name'] + epoch_types

for analysis in analyses:
    print(analysis['name'])
    scores_list = list()

    for subject in subjects:
        save_dir = op.join(data_path, 'MEG', subject, 'decoding')
        if baseline:
            pkl_fname = op.join(save_dir, '%s-gat_%s.pickle' % (
                                eptyp_name, analysis['name']))
        elif not baseline:
                pkl_fname = op.join(save_dir, 'nobl_%s-gat_%s.pickle' % (
                                    eptyp_name, analysis['name']))
        with open(pkl_fname, 'rb') as f:
            gat, analysis, sel, events = pickle.load(f)
        scores_list.append(gat.scores_)

    scores = np.mean(scores_list, axis=0)
    gat.scores_ = scores

    # PLOT
    fig = gat.plot_diagonal(show=False)
    report.add_figs_to_section(fig, 'Across Ss %s %s %s: (diagonal)' % (
        epoch_params['name'], epoch_types, analysis['name']),
        analysis['name'])

    fig = gat.plot(vmin=np.min(gat.scores_),
                   vmax=np.max(gat.scores_), show=False)
    report.add_figs_to_section(fig, 'Across Ss %s %s %s: GAT' % (
        epoch_params['name'], epoch_types, analysis['name']),
        analysis['name'])

    # STATS
    chance = analysis['chance']
    from toolbox.jr_toolbox.utils import stats
    p_values = stats(scores - chance)
    sig = p_values < .05
    decod = [np.diag(score) for score in scores]
    times = gat.train_times_['times']
    pretty_decod(decod, times=times, chance=chance, ax=ax, sig=np.diag(sig))

report.save(open_browser=open_browser)
