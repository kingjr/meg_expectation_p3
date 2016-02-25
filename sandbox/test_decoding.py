# cd /media/DATA/Pro/Projects/Paris/Other/Gabriela/meg_expectation_p3
import os.path as op
import numpy as np
from sklearn.pipeline import make_pipeline
import mne
from mne.decoding import GeneralizationAcrossTime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from jr.gat import scorer_spearman
from p300.conditions import get_events

data_path = op.join('/media/DATA/Pro/Projects/Paris/Other/Gabriela/',
                    'meg_expectation_p3/data/')
# JR harddrive file
epo_fname = op.join(data_path, 'MEG', 's4_sa130042',
                    'stim_lock-unmasked-s4_sa130042-epo.fif')
# Gabriela recently sent file: with baseline correction
epo_fname = op.join(data_path, 'for_jr-epo.fif')

# Load data
epochs = mne.read_epochs(epo_fname)
events = get_events(epochs.events)
epochs.pick_types(meg='mag')
epochs.decimate(2)
epochs.crop(0, .7)
events.reset_index()  # XXX /!\ This is a necessary correction!
sel = events.query('not soa_undef and not local_undef and not pas_undef').index
max(sel)
factors = ['local_seen', 'pas', 'soa']
y = np.array(events[factors].values, float)

# Test GAT
clf = make_pipeline(StandardScaler(), Ridge())
gat = GeneralizationAcrossTime(clf=clf, scorer=scorer_spearman, n_jobs=-1)
gat.fit(epochs[sel], y=y[sel])
y_pred = gat.predict(epochs[sel])  # predict all factors at once
for ii, factor in enumerate(factors):  # score each factor separately
    gat.y_pred_ = y_pred[:, :, :, ii]
    scores = gat.score(y=y[sel, ii])
    gat.plot()

# Check correct CV
for train, test in gat._cv_splits:
    if test in train:
        raise ValueError('Duplicate!')

# Check duplicate data points in data
n = len(sel)
equals = 0
for chan in range(102):
    for t in range(len(epochs.times)):
        epochs._data.shape
        equals += len(np.unique(epochs._data[sel][:, chan, t])) - n
print(equals)

# Check manually written GAT


def manual_gat(X, y):
    from sklearn.cross_validation import KFold
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    n_samples, n_feature, n_time = X.shape
    cv = KFold(n_samples, 2)
    # fit and predict with CV
    y_pred = np.zeros((n_time, n_time, n_samples, 3))
    for train, test in cv:
        for T in range(n_time):
            print T
            for t in range(n_time):
                clf.fit(X[train, :, T], y=y[train])
                y_pred[T, t, test, :] = clf.predict(X[test, :, t])
    # score
    score = np.zeros((n_time, n_time))
    for ii in range(3):
        for T in range(n_time):
            for t in range(n_time):
                score[T, t] = mean_squared_error(y[:, ii], y_pred[T, t, :, ii])
    plt.matshow(score)
    plt.show()

# on MEG data
manual_gat(epochs._data[sel], y[sel, :])

# on synthetic data
n_samples = 100
X = np.random.rand(n_samples, 50, 5)
y = np.random.rand(n_samples, 3)
manual_gat(X, y)
