import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.io.pick import _picks_by_type as picks_by_type
from mne.epochs import EpochsArray
from mne.report import Report

from jr.stats import robust_mean
#from jr import OnlineReport



from p300.conditions import get_events
from scripts.config import data_path, subjects, chan_types, results_dir

#report = OnlineReport()
# XXX NEED TO CLEAN TO USE OnlineReport (save)
report = Report()
report.data_path = op.join(results_dir,'run_template_mask')

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
    soas = [17, 33, 50, 67, 83]
    ch_picks = range(len(epochs.ch_names))
    events = get_events(epochs.events)

    if False:
        # TO BE REMOVED ONCE CHECKED
        # Make fake data to check that it's working
        epochs._data = np.zeros_like(epochs._data)
        mask = np.zeros_like(epochs._data[1, ...])
        mask[:, np.where(epochs.times >= 0)[0]] = 1.
        for trial in range(len(epochs)):
            soa = np.array(events.soa_ttl)[trial]
            target_present = np.array(events.present)[trial]
            toi_mask = np.where(epochs.times >= (soa / 1e3))[0]
            epochs._data[trial, :, toi_mask] += 1.
            if target_present:
                toi_target = np.where(epochs.times >= 0)[0]
                epochs._data[trial, :, toi_target] += .5

        # plot evoked for present and absent conditions
        fig, axes = plt.subplots(2, 5)
        for s, soa in enumerate(soas):
            sel = np.where((np.array(events.soa_ttl) == soa) &
                           (np.array(events.present) == False))[0]
            evo = epochs[sel].average()
            axes[0, s].matshow(evo.data, vmin=-1, vmax=2.)
            axes[0, s].set_title(np.where(evo.data[0, :] > 0.)[0][0])
            sel = np.where((np.array(events.soa_ttl) == soa) &
                           (np.array(events.present) == True))[0]
            evo = epochs[sel].average()
            axes[1, s].matshow(evo.data, vmin=-1, vmax=2.)
            axes[1, s].set_title(np.where(evo.data[0, :] > 0.)[0][0])
        plt.show()
    # Apply each contrast and Realign
    # NB: The TTL setup has different SOA between trigger & mask:
    # realign necessary
    n_time = len(epochs.times)
    evokeds_abs, evokeds_pst = list(), list()
    for s, soa in enumerate(soas):
        # find absent trials
        sel = np.where((np.array(events.soa_ttl) == soa) &
                       (np.array(events.present) == False))[0]
        evoked_abs = epochs[sel].average(picks=ch_picks)
        # change classic averaging with robust averaging (hybrid median mean)
        evoked_abs.data = robust_mean(epochs._data[sel, :, :], axis=0,
                                      percentile=[10, 90])[ch_picks, :]

        # find present trials
        sel = np.where((np.array(events.soa_ttl) == soa) &
                       (np.array(events.present)))[0]
        evoked_pst = epochs[sel].average(picks=ch_picks)
        # change classic averaging with robust averaging (hybrid median mean)
        evoked_pst.data = robust_mean(epochs._data[sel, :, :], axis=0,
                                      percentile=[10, 90])[ch_picks, :]
        # XXX MNE uses trial number as a weight in subtractions
        evoked_abs.nave = 1
        evoked_pst.nave = 1

        # realign selected epochs to mask onset
        toi = np.arange(soa * sfreq / 1000,
                        n_time - ((83 - soa) * sfreq / 1000)).astype(int)

        evoked_abs.data = evoked_abs.data[:, toi]
        evoked_abs.times = evoked_abs.times[0:len(toi)]
        evokeds_abs.append(evoked_abs)

        evoked_pst.data = evoked_pst.data[:, 0:len(toi)]
        evoked_pst.times = evoked_pst.times[0:len(toi)]
        evokeds_pst.append(evoked_pst)

    # Create overall templates (average of mask-locked evo abs across SOA)
    # -- force mne to take all channels instead of meg and eeg
    events_ = np.zeros((5, 3), int)
    data = [evoked.data for evoked in evokeds_abs]
    template_abs = EpochsArray(data, epochs.info, events_,
                               tmin=epochs.times.min()).average(picks=ch_picks)
    template_abs.comment = 'absent'
    template_abs.nave = 1  # XXX Stupid MNE
    data = [evoked.data for evoked in evokeds_pst]
    template_pst = EpochsArray(data, epochs.info, events_,
                               tmin=epochs.times.min()).average(picks=ch_picks)
    template_pst.comment = 'present'
    template_pst.nave = 1  # XXX Stupid MNE

    all_templates.append([template_abs, template_pst])

    ###########################################################################
    # Plot
    # One must first locked the stims to the mask before allowing a subtraction
    evokeds_pst_nomask = [evo_.copy() for evo_ in evokeds_pst]
    # crop so that the template and the data have the same timing
    _template_abs = template_abs.copy()
    _template_abs.crop(-.450, .800)
    for s, soa in enumerate(soas):
        # lock to the mask
        start = -.450 + soa / 1e3
        # ensure that each soa contains the same amount of time samples
        stop = evokeds_pst_nomask[s].times[
            np.where(evokeds_pst_nomask[s].times >= start)[0][0] +
            len(_template_abs.times) - 1]
        evokeds_pst_nomask[s].crop(start, stop)
        evokeds_pst_nomask[s].times = _template_abs.times
        evokeds_pst_nomask[s] = evokeds_pst_nomask[s] - _template_abs

    # # Mmmmh, is this correct?...
    # # XXX TODO
    # fig, axes = plt.subplots(2, 5)
    # for s, evo in enumerate(evokeds_pst_nomask):
    #     axes[0, s].matshow(evo.data, vmin=-1, vmax=2.)
    #     axes[0, s].set_title(np.where(evo.data[0, :] > 0.)[0][0])
    # plt.show()

    # color min max from average
    # generic plotting function
    def imshow(ax, ep, picks, mM):
        ax.imshow(ep.data[picks, :], vmin=-mM, vmax=mM, interpolation='none',
                  aspect='auto', cmap='RdBu_r', extent=[min(ep.times),
                  max(ep.times), 0, len(picks)])

    # for s, soa in enumerate(soas):
    #     sel = np.where((np.array(events.soa_ttl) == soa) &
    #                    (np.array(events.present) == True))[0]
    #     evo = evokeds_pst_nomask[sel].average()
    #     plt.matshow(evo.data)
    # plt.show()

    # each figure corresponds to a sensor type
    for c, chan_type in enumerate(chan_types):
        # select channels
        picks = [i for k, p in picks_by_type(template_pst.info) for i in p
                 if k in chan_type['name']]
        mM = abs(np.percentile(template_pst.data[picks, :], 99.))
        # figure soas x [(target mask), mask only, (target & mask) - mask]
        fig, axes = plt.subplots(len(soas) + 1, 3)
        # each line corresponds to a soa
        for s, soa in enumerate(soas):
            imshow(axes[s, 0], evokeds_pst[s], picks, mM)
            imshow(axes[s, 1], evokeds_abs[s], picks, mM)
            imshow(axes[s, 2], evokeds_pst_nomask[s], picks, mM)

        # plot average mask
        s = len(soas)
        imshow(axes[s, 0], template_pst, picks, mM)
        imshow(axes[s, 1], template_abs, picks, mM)
        # incorrect: not saved at the correct time
        # XXXXXXXXXXXXXXXXXXxxx XXX TODO
        imshow(axes[s, 2], template_pst, picks, mM)

        report.add_figs_to_section(fig, subject + chan_type['name'], subject)
    # plt.show()

    ##########################################################################
    # Save
    ave_fname = op.join(data_path, 'MEG', subject,
                        '{}-{}-mask-ave.fif'.format(ep_name, subject))

    mne.write_evokeds(ave_fname, [template_pst, template_abs])

report.save()
