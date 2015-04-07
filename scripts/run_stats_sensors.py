import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import warnings

from toolbox.jr_toolbox.utils import cluster_stat, Evokeds_to_Epochs
from meeg_preprocessing.utils import setup_provenance

import mne

from scripts.config import (
    data_path,
    subjects,
    results_dir,
    epochs_contrasts,
    open_browser,
    contrasts,
    chan_types
)

# XXX uncomment
# report, run_id, results_dir, logger = setup_provenance(
#     script=__file__, results_dir=results_dir)


# XXX JRK: With neuromag, should have virtual sensor in the future. For now,
# only apply stats to mag.
if 'meg' in [i['name'] for i in chan_types]:
    import copy
    # find 'meg' ch_type
    i = [i for i, le_dict in enumerate(chan_types)
               if le_dict['name'] == 'meg'][0]
    meg_type = chan_types[i].copy()
    meg_type.pop('name', None)
    chan_types[i] = dict(name='mag', **meg_type)

# Apply contrast on each type of epoch
# XXX remove --------------------------
ep = epochs_contrasts[1]
contrast = contrasts[0]
chan_type = chan_types[0]
subjects = subjects[:15]
# XXX ---------------------------------
# for ep in epochs_contrasts:
#     for contrast in contrasts:
#         print(ep['name'] + ':' + contrast)
#         for chan_type in chan_types:

# Contrasts
evokeds = ([], []) # XXX JRK: only valid for binary contrasts
# Gather data across all subjects
for s, subject in enumerate(subjects):
    ave_fname = op.join(data_path, 'MEG', subject,
                        '{}-{}-contrasts-ave.fif'.format(ep['name'],
                                                         subject))
    key = contrast['include'].keys()[0]
    for v, value in enumerate(contrast['include'][key]):
        evoked = mne.read_evokeds(ave_fname, contrast['name'] +
                                             str(value))
        if evoked.nave == 0:
            print('No epochs for {} {}!'.format(subject,
                                                contrast['name']))
        picks = [evoked.ch_names[ii] for ii in
                 mne.pick_types(evoked.info, meg=chan_type['name'])]
        evoked.pick_channels(picks)
        evokeds[v].append(evoked)

# Stats
cluster = cluster_stat(evokeds, n_permutations=2 ** 11,
                       connectivity=chan_type['connectivity'],
                       threshold=dict(start=1., step=1.), n_jobs=-1)

# Plots
i_clus = np.where(cluster.p_values_ < .01)
fig = cluster.plot(i_clus=i_clus, show=False)
# report.add_figs_to_section(fig, '{}: {}: Clusters time'.format(
#     ep['name'], contrast['name']), ep['name'] + contrast['name'])

fig = cluster.plot_topomap(sensors=False, contours=False, show=False)
# report.add_figs_to_section(fig, '{}: {}: topos'.format(
#     ep['name'], contrast['name']), ep['name'] + contrast['name'])
# n = stats.p_values
# ax = plt.subplots(1,n)
# for i in range(n):
#     stats.plot_topo(i, axes=ax[i], title='Cluster #%s' % str(i),
#                     sensors=False, contours=False)
# XXX uncomment
# report.add_figs_to_section(fig, '{}: {}: Clusters'.format(ep['name'], contrast['name']),
#                            ep['name'] + contrast['name'])

# XXX Save in file

# XXX uncomment
# report.save(open_browser=open_browser)
