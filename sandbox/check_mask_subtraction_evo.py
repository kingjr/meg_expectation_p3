"""
Test whether epochs_mask and epochs_unmask are different. No they arent.
Test OK...
"""
import numpy as np
import matplotlib.pyplot as plt
import mne
from p300.conditions import get_events
from p300.run_mask_subtraction import subtract_mask
from scripts.config import subjects
from mne import pick_types

mask_list, unmask_list, mean_all_unmask_list = list(), list(), list()
for subject in subjects:
    print subject
    path = '/Volumes/INSERM/data/MEG/%s/' % subject
    fname_mask = path + 'nobl-stim_lock-%s-epo.fif' % subject
    fname_unmask = path + 'nobl-stim_lock-unmasked-%s-epo.fif' % subject
    fname_mean_all_unmask = path + 'MEAN_ALL-nobl-stim_lock-unmasked-%s-epo.fif' % subject

    epochs_mask = mne.read_epochs(fname_mask)
    epochs_unmask = mne.read_epochs(fname_unmask)
    epochs_mean_all_unmask = mne.read_epochs(fname_mean_all_unmask)

    events = get_events(epochs_mask.events)
    absent = np.where(events['present'] == False)[0]
    evoked_mask = epochs_mask[absent].average()
    evoked_unmask = epochs_unmask[absent].average()
    evoked_mean_all_unmask = epochs_unmask[absent].average()

    #Only take the MEG channels (because diff # of EEG channels/subj)
    meg = pick_types(evoked_mask.info, eeg=False, meg=True)
    evoked_mask.data=evoked_mask.data[meg,:]
    evoked_unmask.data=evoked_unmask.data[meg,:]
    evoked_mean_all_unmask.data=evoked_unmask.data[meg,:]

    mask_list.append(evoked_mask.data)
    unmask_list.append(evoked_unmask.data)
    mean_all_unmask_list.append(evoked_unmask.data)

evoked_mask.data = np.mean(mask_list, axis=0)
evoked_unmask.data = np.mean(unmask_list, axis=0)
evoked_mean_all_unmask.data = np.mean(mean_all_unmask_list, axis=0)
plt.matshow(evoked_mask.data)
plt.matshow(evoked_unmask.data)
plt.matshow(evoked_mean_all_unmask.data)
# evoked_mask.plot(show=False)
# evoked_unmask.plot(show=False)
plt.show()



epochs_unmask = subtract_mask(epochs_mask,
                              events['soa_ttl'], events['present'])
