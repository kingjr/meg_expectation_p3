import os.path as op
import numpy as np
import mne
from mne.stats.regression import linear_regression
from p300.conditions import get_events
from scripts.config import epochs_params, subjects

# File name
from scripts.config import data_path
# data_path = op.join('/media', 'DATA', 'Pro', 'Projects', 'Paris', 'Other',
#                    'Gabriela', 'meg_expectation_p3', 'data')
all_regressions = list()
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
    epochs.pick_types(meg=True)
    epochs.decimate(2)
    epochs.crop(0, .7)

    # Apply each analysis
    sel = events.query('soa > 17 and soa < 83 and not local_undef' +
                       'and not pas_undef').index
    factors = ['local_seen', 'pas', 'soa']
    y = np.array(events[factors].values, float)
    evokeds = linear_regression(epochs[sel], y[sel], factors)
    all_regressions.append(evokeds)

# average results for plotting
for this_factor in evokeds:  # for each factor
    # mean beta coefficient across subjects
    evoked.data = np.mean([evoked[this_factor].beta.data
                           for evoked in all_regressions], axis=0)

# plot mean across subjects
for this_factor in evokeds:
    print(this_factor)
    # XXX what scale should it be??
    evokeds[this_factor].beta.plot_joint(
        title=this_factor, times=[0., .100, .200, .300],
        ts_args=dict(gfp=True), topomap_args=dict(sensors=False))
