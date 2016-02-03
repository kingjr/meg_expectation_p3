"""
WIP
This scripts aims at isolating the evoked response of the target and of the
without the complex subtraction.
"""
import os.path as op
import numpy as np
import mne
from mne.stats.regression import linear_regression_raw
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
raw_events = extract_events(subject, raw)
# XXX my computer is too slow, let's only take magnetometers
# XXX ideally we may want to apply the analysis in ICA/PCA space with
# regularization
raw.pick_types(meg='mag')

events = get_events(raw_events)  # get event meanings

# --- generate a selection of events (add mask events to explicitly mention
# their occurences)
n_trials = len(events)
for loc, event in events.iterrows():
    if event['type'] == 'stim':
        mask_event = event.copy(deep=True)
        mask_event['event_id'] = 3
        mask_event['type'] = 'mask'
        soa = mask_event['soa'] if not np.isnan(mask_event['soa']) else 0.
        mask_event['time_sample'] += (soa * raw.info['sfreq'] // 1000)
        events.loc[len(events)] = mask_event
    # we'll remove every event for which one of the columns is nan
    if loc >= n_trials:
        break

# creates additional column to respect events 3-column structure
events['ignore'] = 0

# Drop meaningless events


def drop(cond):
    events.drop(events.index[cond], inplace=True)

drop(events['run_trial_id'] == 0)  # no context
drop(events['missed_m1'])  # missed responses
drop(events['missed_m2'])
drop((-events['present']) & (events['type'] == 'stim'))  # absent trials

# Ensure continous values (avoid strings and booleans)
events['local_seen'] = events['local_seen'].astype(int)
events['event_id'] = 0
event_id = dict()
for ii, this_type in enumerate(('stim', 'mask', 'motor1', 'motor2')):
    event_id[this_type] = ii
    sel = events['type'] == this_type
    events['event_id'][sel] = ii

# Set covariates as 0 for events different from stim
sel = events['type'] != 'stim'
events['local_seen'][sel] = 0
events['soa'][sel] = 0

# remove duplicates
time_samples = list()
for loc, event in events.iterrows():
    if event['time_sample'] in time_samples:
        events.drop(loc, inplace=True)
    else:
        time_samples.append(event['time_sample'])

# main function
events.reset_index(inplace=True)
evokeds = linear_regression_raw(
    raw, events=events[["time_sample", "ignore", "event_id"]].values,
    event_id=event_id, reject=None, tmin=-.050, tmax=.600,
    # covariates=events[["local_seen", "soa"]])
    )

# plot
for this_factor in evokeds:
    print(this_factor)
    # XXX what scale should it be??
    evokeds[this_factor].plot_joint(
        title=this_factor, times=[0., .100, .200, .300],
        ts_args=dict(gfp=True), topomap_args=dict(sensors=False))
