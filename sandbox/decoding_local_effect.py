import os
import os.path as op
import numpy as np
import mne
from mne.decoding import GeneralizationAcrossTime
import pickle
import matplotlib.pyplot as plt
from jr.gat import scorer_auc, scorer_spearman
from p300.conditions import get_events
from scripts.config import epochs_params, subjects
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.cross_validation import KFold

# File name
from scripts.config import data_path, baseline
# data_path = op.join('/media', 'DATA', 'Pro', 'Projects', 'Paris', 'Other',
#                    'Gabriela', 'meg_expectation_p3', 'data')
all_scores = list()
for subject in subjects:
    print subject
    epoch_params = epochs_params[0]
    epoch_type = '-unmasked'
    eptyp_name = epoch_params['name'] + epoch_type
    if baseline:
        epo_fname = op.join(data_path, 'MEG', subject,
                            '{}-{}-epo.fif'.format(eptyp_name, subject))
    elif not baseline:
        epo_fname = op.join(data_path, 'MEG', subject,
                            'nobl-{}-{}-epo.fif'.format(eptyp_name, subject))

    # Load data
    epochs = mne.read_epochs(epo_fname)
    events = get_events(epochs.events)
    epochs.pick_types(meg='mag')
    epochs.decimate(2)
    epochs.crop(0, .7)

    # Apply each analysis
    # sel = events.query('soa > 17 and soa < 83 and not local_undef ' +
    #                   'and not pas_undef').index
    sel = events.query('not soa_undef and not local_undef ' +
                        'and not pas_undef').index
    single_factor = False
    if single_factor:
        factors = ['soa']
        y = np.ravel(np.array(events[factors], float))
        # when we predict only one factor, we can use the LR
        clf = None  # by default: StandardScaler + LogisticRegression
        scorer = scorer_auc
        scorer = scorer_spearman
        gat = GeneralizationAcrossTime(clf=clf, scorer=scorer,
                                       predict_method='predict_proba',
                                       n_jobs=-1)
        gat.fit(epochs[sel], y=y[sel])
        scores = gat.score(epochs[sel], y=y[sel])
        factors_scores = [scores]
    else:
        sel = events.query('soa > 17 and soa < 83 and not local_undef ' +
                           'and not pas_undef').index
        factors = ['local_seen', 'pas', 'soa']
        y = np.array(events[factors].values, float)
        # y is 1245, 3 (values for each trial, each factor)
        # we have to use another regression: Ridge should be ok
        # RidgeCV(alphas=[.01, .1, 1., 10])  # better but slower
        clf = make_pipeline(StandardScaler(), Ridge())
        # We need to specify the cross validation because it cannot be
        # stratified anymore.
        # cv = KFold(len(y[sel]))
        gat = GeneralizationAcrossTime(clf=clf, scorer=scorer_spearman,
                                       n_jobs=-1)
        # fits all factors at once
        gat.fit(epochs[sel], y=y[sel])
        # predict all factors at once
        y_pred = gat.predict(epochs[sel])
        # score each factor separately
        factors_scores = list()
        for ii, factor in enumerate(factors):
            print factor
            # y_pred shape: train time, test time, trials, factor
            gat.y_pred_ = y_pred[:, :, :, ii]
            scores = gat.score(y=y[sel, ii])
            # gat.plot()
            factors_scores.append(scores)
    all_scores.append(factors_scores)

for ii, this_factor in enumerate(factors):  # for each factor
    # mean beta coefficient across subjects
    gat.scores_ = np.mean([sbj_factor_score[ii]
                           for sbj_factor_score in all_scores],
                          axis=0)
    gat.plot(title=this_factor)

    # Save
    save_dir = op.join(data_path, 'MEG', 'decoding_all')
    if not op.exists(save_dir):
        os.makedirs(save_dir)
    if baseline:
        pkl_fname = op.join(save_dir, '%s-gat_%s.pickle' % (
            eptyp_name, this_factor))
    elif not baseline:
        pkl_fname = op.join(save_dir, 'nobl_%s-gat_%s.pickle' % (
            eptyp_name, this_factor))
    # Save classifier results
    with open(pkl_fname, 'wb') as f:
        pickle.dump([gat, sel, events], f)
