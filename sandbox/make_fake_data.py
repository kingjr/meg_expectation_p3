import pickle
import os
import os.path as op
import numpy as np
import copy

from scripts.config import data_path, analyses

subjects = ['s5_sg120518', 's6_sb120316', 's7_jm100109',
            's8_pe110338', 's9_df130078', 's10_ns110383', 's11_ts100368',
            's12_aa100234', 's13_jn120580', 's14_ac130389']

eptyp_name = 'stim_lock'
for analysis in analyses:
    print analysis
    # load first subject
    load_dir = op.join(data_path, 'MEG', 's4_sa130042', 'evokeds')
    pkl_fname = op.join(load_dir, '%s-cluster_sensors_%s.pickle' % (
        eptyp_name, analysis['name']))
    with open(pkl_fname, 'r') as f:
        coef, evokeds, analysis, events = pickle.load(f)

    # add noise and save other subjects
    for subject in subjects:
        print subject
        # add noise
        coef_ = copy.deepcopy(coef)
        coef_.data = coef.data + coef.data / 3. * np.random.randn(
            *coef.data.shape)

        # save as a different subject
        save_dir = op.join(data_path, 'MEG', subject, 'evokeds')
        if not op.exists(save_dir):
            os.makedirs(save_dir)
        pkl_fname = op.join(save_dir, '%s-cluster_sensors_%s.pickle' % (
            eptyp_name, analysis['name']))
        with open(pkl_fname, 'wb') as f:
            pickle.dump([coef_, evokeds, analysis, events], f)
