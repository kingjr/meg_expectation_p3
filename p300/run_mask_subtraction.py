import numpy as np

def simulate_data():
    """Simulate data"""
    from mne import create_info, EpochsArray
    n_trial, n_chan, n_time = 100, 64, 200
    soas = np.array([17, 33, 50, 67, 83])
    event_soa = soas[np.random.randint(0, len(soas), n_trial)]
    event_present = np.random.randint(0, 2, n_trial)
    events = np.hstack((
        np.zeros((n_trial, 2), int),
        np.array(event_soa + 1e3 * event_present, int)[:, None]))
    ch_names = ['EEG_%i' % ch for ch in range(n_chan)]
    info = create_info(ch_names, 250, 'eeg')
    epochs = EpochsArray(np.zeros((n_trial, n_chan, n_time)),
                         info, events, tmin=-.100)
    return epochs


def subtract_mask(epochs, event_soa, event_present):
    """
    Parameters
    ----------
        epochs : Epochs
        event_soa : np.array, shape (n_trials,)
            Contains SOA ttl value [17, 33, 50, 67, 83] (include mask SOA)
        event_present : np.array, shape (n_trials,)
            Contains 0 and 1.
    """
    from mne import pick_types
    from jr.stats import robust_mean
    # Define SOA
    soas = np.array([17, 33, 50, 67, 83])
    n_trial, n_chan, n_time = epochs._data.shape

    for trial, (soa, present) in enumerate(zip(event_soa, event_present)):
        t = np.where(epochs.times >= soa / 1000.)[0][0]
        # add mask data
        epochs._data[trial, :, t] = -1
        # add target data
        if present:
            t0 = np.where(epochs.times >= 0.)[0][0]
            epochs._data[trial, :, t0] = 1

    # Create mask template
    sfreq = epochs.info['sfreq']
    # --- Align mask trial
    data_mask = list()
    for soa in soas:
        # select absent trials
        sel = np.where((event_soa == soa) & (event_present == 0))[0]
        mask = epochs._data[sel, :, :]
        # add zeros before and after
        preshift = (soas[-1] - soa) / 1000. * sfreq
        pre = np.zeros((len(sel), n_chan, preshift))
        postshift = soa / 1000. * sfreq
        post = np.zeros((len(sel), n_chan, postshift))
        mask = np.concatenate((pre, mask, post), axis=2)
        data_mask.append(mask)
        print(preshift, postshift)
    # --- Robust average across all mask
    data_mask = np.concatenate(data_mask, axis=0)
    mask_template = robust_mean(data_mask, axis=0)
    # plt.matshow(data_mask[:, 0, :])

    # Substract mask template
    picks = pick_types(epochs.info, meg=True, eeg=True, stim=False, misc=False)
    for soa in soas:
        # select all target present trials
        sel = np.where(event_soa == soa)[0]
        data = epochs._data[sel, :, :]
        # align mask to target-locked
        mask = mask_template
        preshift = (soas[-1] - soa) / 1000. * sfreq
        postshift = soa / 1000. * sfreq
        mask = mask[:, preshift:-postshift]
        # subtract target-locked mask onto target-locked data
        data[:, picks, :] -= np.tile(mask[picks, :], [len(sel), 1, 1])
        epochs._data[sel, :, :] = data
    # plt.matshow(epochs._data[:, 0, :], aspect='auto')
    return epochs
