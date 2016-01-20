"""
WIP
This scripts aims at isolating the evoked response of the target and of the
without the complex subtraction.
"""
import os.path as op
import numpy as np
import mne
from mne.stats.regression import linear_regression_raw
from collections import Counter
from p300.conditions import get_events
from p300.conditions import extract_events
data_path = op.join('/media/DATA/Pro/Projects/Paris/Other/Gabriela/',
                    'meg_expectation_p3/data/MEG/')
subject = 's4_sa130042'
run = 1
# --- read data
# XXX ideally, we'd need to first combine all run (after head realignment)
raw_fname = op.join(data_path, subject, 'run_%.2i_filt-0-35_sss_raw.fif' % run)
raw = mne.io.Raw(raw_fname, preload=True)  # Take first run

# --- extract events
events = extract_events(subject, raw)
# XXX my computer is too slow, let's only take magnetometers
# XXX ideally we may want to apply the analysis in ICA/PCA space with
# regularization
raw.pick_types(meg='mag')

events_bhv = get_events(events)  # get event meanings


# --- generate a selection of events (add mask events to explicitly mention
# their occurences)
# trial may have a target and a mask, or just a mask
events_bhv['event_id'] = np.nan * np.zeros(len(events_bhv))
# for date, row in df.T.iteritems():
n_trials = len(events_bhv)
for loc, event in events_bhv.iterrows():
    if loc < 2:
        events_bhv['local_seen'] = False  # XXX
    if event['type'] == 'stim':
        mask_event = event.copy()
        mask_event['event_id'] = 3
        mask_event['type'] = 'mask'
        mask_event['time_sample'] += (mask_event['soa'] * raw.info['sfreq'] //
                                      1000)
        events_bhv.loc[len(events_bhv)] = mask_event
    if event['type'] == 'stim':
        events_bhv['event_id'][loc] = 0
    elif event['type'] == 'motor1':
        events_bhv['event_id'][loc] = 1
    elif event['type'] == 'motor2':
        events_bhv['event_id'][loc] = 2
    if not event['present'] and event['type'] == 'stim':
        events_bhv.drop(loc)
    if loc >= n_trials:
        break
event_id = dict(stim=0, mask=3, motor1=1, motor2=2)

events_regression = np.vstack((
    events_bhv['time_sample'],
    np.zeros_like((events_bhv['time_sample'])),
    events_bhv['event_id'])).T


def find_clean_events(events):
    """
    The functions doesn't allow duplicate events or nan values.
    """
    # remove nan values
    to_be_removed = list()
    for ii in range(3):
        bad = np.where(np.isnan(events[:, ii]))[0]
        if len(bad):
            to_be_removed.append(bad)
    # remove duplicates
    duplicates = {k for k, v in Counter(events[:, 0]).items() if v > 1}
    bad = np.where(np.in1d(events[:, 0], list(duplicates)))[0]
    if len(bad):
        to_be_removed.append(bad)
    to_be_removed = np.hstack(to_be_removed)
    return [ii for ii in range(len(events)) if ii not in to_be_removed]

good_events = find_clean_events(events_regression)
good_events = good_events[1:]

# main function
covariates = dict(local=events_bhv['local_seen'].loc[good_events].astype(int))
evokeds = linear_regression_raw(
    raw, events=events_regression[good_events, :],
    event_id=event_id, reject=None, tmin=-.050, tmax=.600, decim=1,
    covariates=covariates)

# plot
for evoked in evokeds:
    print(evoked)
    # XXX what scale should it be??
    evokeds[evoked].plot_topomap()
