import os.path as op
import numpy as np
import warnings
import mne
from mne.io import Raw
from mne.io.pick import _picks_by_type as picks_by_type
from mne.preprocessing import read_ica
from mne.viz import plot_drop_log

from meeg_preprocessing.utils import setup_provenance, set_eog_ecg_channels

from p300.conditions import get_events

from config import (
    data_path,
    subjects,
    results_dir,
    events_fname_filt_tmp,
    epochs_params,
    open_browser,
    contrasts,
)


report, run_id, results_dir, logger = setup_provenance(script=__file__,
                                                       results_dir=results_dir)

mne.set_log_level('INFO')

for subject in subjects:
    # Extract events from mat file
    bhv_fname = op.join(data_path, 'behavior',
                        '{}_behaviorMEG.mat'.format(subject[-2:]))


    # Apply contrast on each type of epoch
    all_epochs = [[]] * len(epochs_params)
    for epoch_params in epochs_params:
        ep_name = epoch_params['name']

        # Get MEG data
        epo_fname = op.join(data_path, 'MEG', subject,
                            '{}-{}-epo.fif'.format(ep_name, subject))
        epochs = mne.read_epochs(epo_fname)
        # Get events specific to epoch definition (stim or motor lock)
        events = get_events(epochs.events)
        print(sum(events['pas']==0))
        print(sum(events['pas']==1))
        print(sum(events['pas']==2))
        print(sum(events['pas']==3))
        print(sum(events['local_context'] =='UU'))
        print(sum(events['local_context'] =='SU'))
        print(sum(events['local_context'] =='US'))
        print(sum(events['local_context'] =='SS'))
        # name of evoked file
        ave_fname = op.join(data_path, 'MEG', subject,
                            '{}-{}-contrasts-ave.fif'.format(ep_name, subject))

        # Apply each contrast
        all_evokeds = list()
        for contrast in contrasts:
            evokeds = list()

            # Find excluded trials
            exclude = np.any([events[x['cond']]==ii
                                for x in contrast['exclude']
                                    for ii in x['values']],
                             axis=0)

            # Select condition
            for value in contrast['include']['values']:
                # Find included trials
                include = events[contrast['include']['cond']]==value
                # Evoked data
                evoked = epochs[include * (exclude==False)].average()
                evoked.comment = contrast['name']+str(value)
                # keep for contrast
                evokeds.append(evoked)
                # keep for saving
                all_evokeds.append(evoked)

            # Apply contrast
            if np.min([e.nave for e in evokeds]) == 0:
                warnings.warn('%s: no epochs in %s for %s' % (subject, ep_name,
                                                              contrast['name']))
                continue
            diff = evokeds[0] - evokeds[-1]

            # Plot
            fig = diff.plot()
            report.add_figs_to_section(fig, ('%s (%s) %s: butterfly'
                % (subject, ep_name, diff.comment)), 'Butterfly: ' + ep_name)
            plot_times = np.linspace(diff.times.min(),
                                     diff.times.max(),
                                     10)
            fig = diff.plot_topomap(plot_times, ch_type='mag', sensors=False,
                                    contours=False)
            report.add_figs_to_section(fig, ('%s (%s) %s: (topo)'
                % (subject, ep_name, diff.comment)), 'Topo: ' + ep_name)

        # Save all_evokeds
        mne.write_evokeds(ave_fname, all_evokeds)

report.save(open_browser=open_browser)
