import os.path as op
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt

import warnings

import mne
from mne.channels import read_ch_connectivity

from toolbox.jr_toolbox.utils import (cluster_stat, Evokeds_to_Epochs)
from meeg_preprocessing.utils import setup_provenance


from scripts.config import (
    data_path,
    subjects,
    epochs_params,
    epochs_types,
    analyses,
    chan_types,
    results_dir,
    open_browser
)

# XXX uncomment
# report, run_id, results_dir, logger = setup_provenance(
#     script=__file__, results_dir=results_dir)


# XXX to be removed once subjects have been preprocessed
subjects = ['s5_sg120518', 's6_sb120316', 's7_jm100109',
            's8_pe110338', 's9_df130078', 's10_ns110383', 's11_ts100368',
            's12_aa100234', 's13_jn120580', 's14_ac130389']

# XXX JRK: With neuromag, should have virtual sensor in the future. For now,
# only apply stats to mag.
if 'meg' in [i['name'] for i in chan_types]:
    # find 'meg' ch_type
    i = [i for i, le_dict in enumerate(chan_types)
         if le_dict['name'] == 'meg'][0]
    meg_type = chan_types[i].copy()
    meg_type.pop('name', None)
    chan_types[i] = dict(name='mag', **meg_type)

# Apply contrast on each type of epoch
# XXX remove --------------------------
# subjects = subjects[:15]
# XXX ---------------------------------
# for ep in epochs_params:
ep = epochs_params[0]
#     for epoch_type in epochs_types:
epoch_type = epochs_types[0]
eptyp_name = ep['name'] + epoch_type
#         for contrast in contrasts:
#             print(ep['name'] + ':' + contrast)
#             for analysis in analyses:
analysis = analyses[1]

# Analyses
evokeds = list()
# Gather data across all subjects
for s, subject in enumerate(subjects):
    pkl_fname = op.join(data_path, 'MEG', subject, 'evokeds',
                        '%s-cluster_sensors_%s.pickle' % (
                            eptyp_name, analysis['name']))
    with open(pkl_fname) as f:
        coef, evoked, _, _ = pickle.load(f)
    evokeds.append(coef)

epochs = Evokeds_to_Epochs(evokeds)
# XXX warning if subjects has missing condition


# select only evokeds that corresponds to the highest level of contrast
#                   for chan_type in chan_types:
chan_type = chan_types[0]

# Take first evoked to retrieve all necessary information
picks = [epochs.ch_names[ii] for ii in mne.pick_types(
         epochs.info, meg=chan_type['name'])]

# Stats
epochs_ = epochs.copy()
epochs_.pick_channels(picks)

# XXX wont work for eeg
cluster = cluster_stat(epochs_, n_permutations=2 ** 10,
                       connectivity=ch_type['connectivity'],
                       threshold=dict(start=1., step=1.), n_jobs=-1)
# Run stats
from mne.stats import spatio_temporal_cluster_1samp_test

condition_names = 'Aud_L', 'Aud_R', 'Vis_L', 'Vis_R'
X = epochs.get_data()  # as 3D matrix

T_obs_, clusters, p_values, _ = \
    spatio_temporal_cluster_1samp_test(,
                                       out_type='mask',
                                       connectivity=connectivity)

# Plots
i_clus = np.where(cluster.p_values_ < .01)
fig = cluster.plot(i_clus=i_clus, show=False)
# report.add_figs_to_section(fig, '{}: {}: Clusters time'.format(
#     ep['name'], contrast['name']), ep['name'] + contrast['name'])

# plot T vales
fig = cluster.plot_topomap(sensors=False, contours=False, show=False)

# Plot contrasted ERF + select sig sensors
evoked = epochs.average()
evoked.pick_channels(picks)


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
pkl_fname = op.join(data_path,
                    'MEG/fsaverage', '%s-cluster_sensors_%.pickle' % (
                        ep['name'], analysis['name']))

with open(pkl_fname, 'wb') as f:
    pickle.dump([cluster, evokeds, analysis], f)

# XXX uncomment
# report.save(open_browser=open_browser)
