import os.path as op
import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.io.pick import _picks_by_type as picks_by_type

from toolbox.jr_toolbox.utils import find_in_df, Evokeds_to_Epochs

from p300.conditions import get_events

from scripts.config import (
    data_path,
    subjects,
    results_dir,
    events_fname_filt_tmp,
    epochs_params,
    open_browser
)


# XXX to be imported from config:
soas = [17, 33, 50, 67, 83]

# loop across subjects
for subject in subjects:

    # loop across epoch type: stim [and response lock? maybe not, or for later]
    ep = epochs_params[0]


    # PART 1. Get MEG data ##############################################
    # Load data
    epo_fname = op.join(data_path, 'MEG', subject,
                        '{}-{}-epo.fif'.format(ep['name'], subject))
    epochs = mne.read_epochs(epo_fname)

    # # Check that triggers look ok
    # STI101 = np.where([i == 'STI101' for i in epochs.ch_names])[0]
    # # mne.viz.plot_image_epochs(epochs, STI101, scalings=dict(stim=1),
    # #                           units=dict(stim=''), vmin=0, vmax=64)
    #
    # # Get events specific to epoch definition (stim or motor lock)
    events = get_events(epochs.events)

    # Apply each contrast
    all_evokeds = list()
    # only load masks from absent trials:
    sel = find_in_df(events, dict(present=False))
    epochs = epochs[sel]
    events = events.iloc[sel]

    # Realign
    # NB: The TTL setup has different SOA between trigger & mask: realign necessary
    sfreq = epochs.info['sfreq']
    n_time = len(epochs.times)
    evokeds = list()
    fig, ax = plt.subplots(len(soas) + 1, sharey=True)
    mag = [i[1] for i in picks_by_type(epochs.info) if i[0] == 'mag'][0]
    for i, soa in enumerate(soas):
        sel = find_in_df(events, dict(soa_ttl=soa))
        evoked = epochs[sel].average(picks=range(len(epochs.ch_names)))
        toi = np.arange(soa * sfreq / 1000,
                        n_time - ((83 - soa) * sfreq / 1000)).astype(int)
        evoked.data = evoked.data[:, toi]
        evoked.times = evoked.times[toi]
        ax[i].matshow(evoked.data[mag, :])
        evokeds.append(evoked)
    # create template
    template = Evokeds_to_Epochs(evokeds).average(picks=range(len(epochs.ch_names)))
    template.comment = 'mask'
    # plot
    ax[-1].matshow(evoked.data[mag, :])
    plt.show()

    # Save
    mask_fname = op.join(data_path, 'MEG', subject,
            '{}-{}-mask-ave.fif'.format(ep['name'], subject))
    template.save(mask_fname)

    # Part 2. load all epochs ##############################################
    epo_fname = op.join(data_path, 'MEG', subject,
                        '{}-{}-epo.fif'.format(ep['name'], subject))
    epochs = mne.read_epochs(epo_fname)
    events = get_events(epochs.events)
    # load mask
    mask_fname = op.join(data_path, 'MEG', subject,
            '{}-{}-mask-ave.fif'.format(ep['name'], subject))
    template = mne.read_evokeds(mask_fname, 'mask')

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
