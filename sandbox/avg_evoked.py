import pickle
import os.path as op
import numpy as np
from meeg_preprocessing.utils import setup_provenance
from scripts.config import (
    data_path, epochs_types, epochs_params,
    subjects,
    results_dir,
    open_browser,
    baseline)
from p300.analyses import analyses

report, run_id, results_dir, logger = setup_provenance(script=__file__,
                                                       results_dir=results_dir)


epoch_types = epochs_types[1]
epoch_params = epochs_params[0]

for analysis in analyses:
    print(analysis['name'])
    evoked_list = list()
    for subject in subjects:
        eptyp_name = epoch_params['name'] + epoch_types

        # Load evokeds
        # FIXME make paths in config to avoid dealing with file
        save_dir = op.join(data_path, 'MEG', subject, 'evokeds')
        if baseline:
            pkl_fname = op.join(save_dir, '%s-cluster_sensors_%s.pickle' % (
                eptyp_name, analysis['name']))
        if not baseline:
            pkl_fname = op.join(save_dir, 'nobl_%s-cluster_sensors_%s.pickle'
                                % (eptyp_name, analysis['name']))
        with open(pkl_fname, 'rb') as f:
            evoked, sub, analysis = pickle.load(f)
        evoked.pick_types(meg=True, eeg=False, copy=False)
        evoked_list.append(evoked)

    evoked.data = np.mean([e.data for e in evoked_list], axis=0)

    # TODO Save here!
    save_dir = op.join(data_path, 'MEG/gm/evokeds')
    if baseline:
        pkl_fname = op.join(save_dir, '%s-cluster_sensors_%s.pickle' % (
            eptyp_name, analysis['name']))
    elif not baseline:
        pkl_fname = op.join(save_dir, 'nobl_%s-cluster_sensors_%s.pickle'
                            % (eptyp_name, analysis['name']))
    with open(pkl_fname, 'wb') as f:
        pickle.dump(evoked, f)

    fig0 = evoked.plot_topomap(show=False, scale=1,
                               times=np.linspace(0, .500, 20))
    report.add_figs_to_section(fig0, ('%s: %s' % (eptyp_name,
                                                  analysis['name'])))

report.save(open_browser=open_browser)
