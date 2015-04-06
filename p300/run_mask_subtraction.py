import os.path as op
import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.io.pick import _picks_by_type as picks_by_type

from toolbox.jr_toolbox.utils import find_in_df
from toolbox.jr_toolbox.utils import Evokeds_to_Epochs as avg2epo

from meeg_preprocessing.utils import setup_provenance

from p300.conditions import get_events

from scripts.config import (
    data_path,
    subjects,
    results_dir,
    events_fname_filt_tmp,
    epochs_params,
    open_browser,
    chan_types
)


report, run_id, results_dir, logger = setup_provenance(
    script=__file__, results_dir=results_dir)


# force separation of magnetometers and gradiometers
if 'meg' in [i['name'] for i in chan_types]:
    chan_types = [dict(name='mag'), dict(name='grad')] + \
                 [dict(name=i['name']) for i in chan_types
                                           if i['name'] != 'meg']

# Part 2. load all epochs ##############################################
for subject in subjects:
    epo_fname = op.join(data_path, 'MEG', subject,
                        '{}-{}-epo.fif'.format(ep['name'], subject))
    epochs = mne.read_epochs(epo_fname)
    events = get_events(epochs.events)
    # load mask
    mask_fname = op.join(data_path, 'MEG', subject,
            '{}-{}-mask-ave.fif'.format(ep['name'], subject))
    template = mne.read_evokeds(mask_fname, 'absent')

    # remove template from individual epochs
    n_epoch, n_all_chan, n_time = epochs._data.shape
    n_data_chan, n_time_template = template.data.shape
    data = np.zeros((n_epoch, n_data_chan, n_time_template))
    for soa in soas:
        sel = find_in_df(events, dict(soa_ttl=soa))
        for trial in sel:
            toi = np.arange(soa * sfreq / 1000,
                            n_time - ((83 - soa) * sfreq / 1000)).astype(int)
            data[trial, :, :] = epochs._data[trial, :, toi].transpose() - \
                                template.data
    epochs._data = data
    epochs.times = epochs.times[:n_time_template]
    epochs.tmax = max(epochs.times)
    # save to new epochs
    unmasked_fname = op.join(data_path, 'MEG', subject,
            '{}-unmasked-{}-epo.fif'.format(ep['name'], subject))
    epochs.save(unmasked_fname)

report.save(open_browser=open_browser)
