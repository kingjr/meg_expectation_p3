# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import sys
import mkl
import os.path as op

import mne

import numpy as np

from copy import deepcopy

from meeg_preprocessing.utils import setup_provenance

from conditions import extract_events

from scripts.config import (
    data_path,
    runs,
    event_id,
    results_dir,
    events_fname_filt_tmp,
    events_params,
    raw_fname_filt_tmp,
    open_browser
)

report, run_id, results_dir, logger = setup_provenance(
    script=__file__, results_dir=results_dir)
#
# if len(sys.argv) > 1:
#     subjects = [sys.argv[1]]
#     mkl.set_num_threads(1)

subject='s20_ad120286'

this_path = op.join(data_path, 'MEG', subject)
for r in runs:
    fname = op.join(this_path, raw_fname_filt_tmp.format(r))
    if not op.isfile(fname):
        logger.info('Could not find %s. Skipping' % fname)
        continue
    raw = mne.io.Raw(fname)

    events_orig = mne.find_events(raw, stim_channel='STI101', verbose=True,
                             consecutive='increasing', min_duration=0.000,
                             shortest_event=1)
    print(len(events_orig))

    events=deepcopy(events_orig)

    for e in range(len(events)):
        if events_orig[e][2]==64:
            events[e][2]=512
        elif events_orig[e][2]==128:
            events[e][2]=256
        elif events_orig[e][2]==256:
            events[e][2]=128
        elif events_orig[e][2]==512:
            events[e][2]=64

    # TODO SAVE NAME??
    # Save
    # mne.write_events(
    #     op.join(this_path, events_fname_filt_tmp.format(r)), events)
    #
    # Plot
    fig = mne.viz.plot_events(
        events, raw.info['sfreq'], raw.first_samp, show=False,
        event_id=event_id)
    report.add_figs_to_section(fig, 'events run %i' % r, subject)
report.save(open_browser=open_browser)
