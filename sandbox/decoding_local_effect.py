import os.path as op
import numpy as np
import mne
from mne.decoding import GeneralizationAcrossTime
from jr.gat import scorer_auc
from p300.conditions import get_events
from scripts.config import epochs_params, subjects

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
    sel = events.query('soa > 17 and soa < 83').index
    y = np.array(events['local_seen'], float)

    gat = GeneralizationAcrossTime(scorer=scorer_auc,
                                   predict_method='predict_proba',
                                   n_jobs=-1)
    gat.fit(epochs[sel], y=y[sel])
    scores = gat.score(epochs[sel], y=y[sel])
    all_scores.append(scores)
