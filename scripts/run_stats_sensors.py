import os.path as op
import pickle
import numpy as np
import matplotlib.pyplot as plt

import warnings

from toolbox.jr_toolbox.utils import (cluster_stat, Evokeds_to_Epochs,
                                      save_to_dict)
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
contrast = contrasts[1]
chan_type = chan_types[0]
# subjects = subjects[:15]
# XXX ---------------------------------
# for ep in epochs_contrasts:
#     for contrast in contrasts:
#         print(ep['name'] + ':' + contrast)


# Contrasts
evokeds = list()
# Gather data across all subjects
for s, subject in enumerate(subjects):
    ave_fname = op.join(data_path, 'MEG', subject,
                        '{}-{}-contrasts-ave.pickle'.format(ep['name'],subject))
    delta, evoked = load_from_dict(ave_fname, contrast['name'], out_type=='list')[0]
    evokeds.append(evoked['current'])
    # XXX warning if subjects has missing condition


# select only evokeds that corresponds to the highest level of contrast
# XXX
#         for chan_type in chan_types:

# take first evoked to retrieve all necessary information
evoked = evokeds[0][0]
picks = [evoked.ch_names[ii] for ii in mne.pick_types(evoked.info, meg=chan_type['name'])]
evoked.pick_channels(picks)

# Stats
cluster = cluster_stat(evokeds, n_permutations=2 ** 11,
                       connectivity=chan_type['connectivity'],
                       threshold=dict(start=1., step=1.), n_jobs=-1)

# Plots
i_clus = np.where(cluster.p_values_ < .01)
fig = cluster.plot(i_clus=i_clus, show=False)
# report.add_figs_to_section(fig, '{}: {}: Clusters time'.format(
#     ep['name'], contrast['name']), ep['name'] + contrast['name'])

# plot T vales
fig = cluster.plot_topomap(sensors=False, contours=False, show=False)

# Plot contrasted ERF + select sig sensors
evoked = Evokeds_to_Epochs(evokeds[0]).average() - \
         Evokeds_to_Epochs(evokeds[-1]).average()
# Create mask of significant clusters
mask, _, _ = cluster._get_mask(i_clus)
# Define color limits
mM = np.percentile(np.abs(evoked.data), 99)
# XXX JRK: pass plotting function to config
evoked.plot_topomap(mask=mask.T, scale=1., sensors=False, contours=False,
                    times=np.linspace(min(evoked.times), max(evoked.times), 20),
                    vmin=-mM, vmax=mM, colorbar=True)

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

# Save contrast
pkl_fname = op.join(data_path, 'MEG/fsaverage', '%s-cluster_sensors.pickle' % (ep['name']))

# If file exist, load already save data and append new cluster
save_var = dict()
save_var[contrast['name']]=dict(evokeds=evokeds, cluster=cluster,
                                contrast=contrast, chan_type=chan_type, ep=ep,
                                subjects=subjects)
save_to_dict(pkl_fname, save_var, overwrite=True)
load_from_dict(pkl_fname, save_var, overwrite=True)

# XXX uncomment
# report.save(open_browser=open_browser)
