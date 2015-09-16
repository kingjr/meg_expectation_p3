import os.path as op
import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.io.pick import _picks_by_type as picks_by_type

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


report, run_id, results_dir, logger = setup_provenance(script=__file__,
                                                       results_dir=results_dir)


# force separation of magnetometers and gradiometers
if 'meg' in [i['name'] for i in chan_types]:
    chan_types = [dict(name='mag'), dict(name='grad')] + \
                 [dict(name=i['name']) for i in chan_types
                                           if i['name'] != 'meg']
# only run on stim lock
ep_name = 'stim_lock'

# loop across subjects
all_templates = list()
for subject in subjects:
    print(subject)
    # Load data
    epo_fname = op.join(data_path, 'MEG', subject,
                        '{}-{}-epo.fif'.format(ep_name, subject))
    epochs = mne.read_epochs(epo_fname)
    sfreq = epochs.info['sfreq']

    # # Get events specific to epoch definition (stim or motor lock)
    events = get_events(epochs.events)

    # Apply each contrast and Realign
    # NB: The TTL setup has different SOA between trigger & mask: realign necessary
    n_time = len(epochs.times)
    evokeds_abs, evokeds_pst = list(), list()
    soas = [17, 33, 50, 67, 83]
    for s, soa in enumerate(soas):
        # find absent trials
        sel = np.where((np.array(events.soa_ttl) == soa) &
                       (np.array(events.present) == False))[0]
        evoked_abs = epochs[sel].average(picks=range(len(epochs.ch_names)))
        # find present trials
        sel = np.where((np.array(events.soa_ttl) == soa) &
                       (np.array(events.present)))[0]
        evoked_pst = epochs[sel].average(picks=range(len(epochs.ch_names)))
        # XXX MNE uses trial number as a weight in subtractions
        evoked_abs.nave = 1
        evoked_pst.nave = 1

        toi = np.arange(soa * sfreq / 1000, n_time - ((83 - soa) * sfreq
                                                       / 1000)).astype(int)

        evoked_abs.data = evoked_abs.data[:, toi]
        evoked_abs.times = evoked_abs.times[0:len(toi)]
        evokeds_abs.append(evoked_abs)

        evoked_pst.data = evoked_pst.data[:, 0:len(toi)]
        evoked_pst.times = evoked_pst.times[0:len(toi)]
        evokeds_pst.append(evoked_pst)

    # Create templates
    #--- force mne to take all channels instead of meg and eeg
    picks = range(len(epochs.ch_names))

    template_abs = avg2epo(evokeds_abs).average(picks=picks)
    template_abs.comment = 'absent'
    template_abs.nave = 1 # XXX Stupid MNE

    template_pst = avg2epo(evokeds_pst).average(picks=picks)
    template_pst.comment = 'present'
    template_pst.nave = 1 # XXX Stupid MNE

    all_templates.append([template_abs, template_pst])

    ############################################################################
    # Plot
    # color min max from average
    # generic plotting function
    imshow = lambda ax, ep, picks, mM: ax.imshow(ep.data[picks, :], vmin=-mM,
                                             vmax=mM, interpolation='none',
                                             aspect='auto', cmap='RdBu_r',
                                            extent=[min(ep.times),
                                                    max(ep.times), 0,
                                                    len(picks)])

    # each figure corresponds to a sensor type
    for c, chan_type in enumerate(chan_types):
        # select channels
        picks = [i for k, p in picks_by_type(template_pst.info) for i in p
                                            if k in chan_type['name']]
        mM = abs(np.percentile(template_pst.data[picks,:], 99.))
        # figure soas x [(target mask), mask only, (target & mask) - mask]
        fig, axes = plt.subplots(len(soas) + 1, 3)
        # each line corresponds to a soa
        for s, soa in enumerate(soas):
            imshow(axes[s, 0], evokeds_pst[s], picks, mM)
            imshow(axes[s, 1], evokeds_abs[s], picks, mM)
            imshow(axes[s, 2], evokeds_pst[s] - template_abs, picks, mM)
            # imshow(axes[s, 2], evokeds_pst[s] - evokeds_abs[s], picks, mM)

        # plot average mask
        s = len(soas)
        imshow(axes[s, 0], template_pst, picks, mM)
        imshow(axes[s, 1], template_abs, picks, mM)
        imshow(axes[s, 2], template_pst - template_abs, picks, mM)

        report.add_figs_to_section(fig, subject + chan_type['name'], subject)
    # plt.show()


    ############################################################################
    # Save
    ave_fname = op.join(data_path, 'MEG', subject,
            '{}-{}-mask-ave.fif'.format(ep_name, subject))

    mne.write_evokeds(ave_fname, [template_pst, template_abs])

report.save(open_browser=open_browser)
