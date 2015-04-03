# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import sys
import mkl
import os.path as op

import mne

import numpy as np

from meeg_preprocessing.utils import setup_provenance

from p300.conditions import extract_events

from config import (
    data_path,
    subjects,
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

if len(sys.argv) > 1:
    subjects = [sys.argv[1]]
    mkl.set_num_threads(1)

for subject in subjects:
    this_path = op.join(data_path, 'MEG', subject)
    for r in runs:
        fname = op.join(this_path, raw_fname_filt_tmp.format(r))
        if not op.isfile(fname):
            logger.info('Could not find %s. Skipping' % fname)
            continue
        raw = mne.io.Raw(fname)

        # Deal with unwanted initial triggers
        if subject in events_params.keys():
            # r - 1 because run starts at 1 not 0
            events_param = events_params[subject][r - 1]
        else:
            events_param = dict()

        events = extract_events(raw)
        #print(len(events))

        # Save
        mne.write_events(
            op.join(this_path, events_fname_filt_tmp.format(r)), events)

        # Plot
        fig = mne.viz.plot_events(
            events, raw.info['sfreq'], raw.first_samp, show=False,
            event_id=event_id)
        report.add_figs_to_section(fig, 'events run %i' % r, subject)
report.save(open_browser=open_browser)
