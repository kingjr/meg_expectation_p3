import os.path as op
import numpy as np
import mne
from mne.decoding import GeneralizationAcrossTime
from jr.gat import scorer_auc, scorer_spearman
from p300.conditions import get_events
from scripts.config import epochs_params, subjects
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV

# File name
from scripts.config import data_path
data_path = op.join('/media', 'DATA', 'Pro', 'Projects', 'Paris', 'Other',
                    'Gabriela', 'meg_expectation_p3', 'data')
all_scores = list()
for subject in subjects:
    epoch_params = epochs_params[0]
    epoch_type = '-unmasked'
    eptyp_name = epoch_params['name'] + epoch_type
    epo_fname = op.join(data_path, 'MEG', subject,
                        '{}-{}-epo.fif'.format(eptyp_name, subject))
    # TODO change to no baseline 'nobl-'

    # Load data
    epochs = mne.read_epochs(epo_fname)
    events = get_events(epochs.events)
    epochs.pick_types(meg='mag')
    epochs.decimate(2)
    epochs.crop(0, .7)

    # Apply each analysis
    sel = events.query('soa > 17 and soa < 83 and not local_undef' +
                       'and not pas_undef').index
    single_factor = False
    if single_factor:
        y = np.array(events['local_seen'], float)
        # when we predict only one factor, we can use the LR
        clf = None  # by default: StandardScaler + LogisticRegression
        scorer = scorer_auc
        gat = GeneralizationAcrossTime(clf=clf, scorer=scorer,
                                       predict_method='predict_proba',
                                       n_jobs=-1)
        gat.fit(epochs[sel], y=y[sel])
        scores = gat.score(epochs[sel], y=y[sel])
    else:
        factors = ['local_seen', 'pas', 'soa']
        y = np.array(events[factors].values, float)
        # we have to use another regression: Ridge should be ok
        # RidgeCV(alphas=[.01, .1, 1., 10])  # better but slower
        clf = make_pipeline(StandardScaler(), Ridge())
        gat = GeneralizationAcrossTime(clf=clf, scorer=scorer_spearman,
                                       n_jobs=-1)
        # fits all factors at onces
        gat.fit(epochs[sel], y=y[sel])
        # predict all factors at once
        y_pred = np.array(gat.predict(epochs[sel]))
        # score each factor separately
        factors_scores = list()
        for ii, factor in enumerate(factors):
            gat.scores_ = y_pred[:, :, :, ii]  # train time, test time, trials, factor
            scores = gat.score(y=y[sel])
            factors_scores.append(scores)
    all_scores.append(scores)
