import os
import os.path as op

# PATHS #######################################################################
base_path = op.dirname(op.dirname(__file__))

data_path = op.join(base_path, 'data')

pass_errors = False

""" Data paths:
The directory structure should look like this

# MEG directory containing MEG data for each subject
./data/MEG/subject1
./data/MEG/...
./data/MEG/subjectN

# freesurfer directory containing MR data for each subject
./data/subjects/subject1
./data/subjects/...
./data/subjects/subjectN

# other stuff
./data/behavior/
"""

# REPORT ######################################################################
open_browser = True

# SUBJECTS ####################################################################
subjects = ['s4_sa130042', 's5_sg120518', 's6_sb120316', 's7_jm100109',
            's8_pe110338', 's9_df130078', 's10_ns110383', 's11_ts100368',
            's12_aa100234', 's13_jn120580', 's14_ac130389', 's15_nv110179',
            's16_mp130429', 's17_ft120490', 's18_rg110386', 's19_cd110147',
            's20_ad120286', 's21_jl130434', 's22_sl130503', 's23_pf120155',
            's24_cl120289', 's25_bb100103', 's26_sb130354']

exclude_subjects = ['s19_cd110147', 's15_nv110179']  # maxfilter error

subjects = [s for s in subjects if s not in exclude_subjects]

runs = list(range(1, 5))  # number of runs per subject

# FILTERING & BASELINING#######################################################
lowpass = 35
highpass = 0.75
filtersize = 16384
baseline = False

# FILENAMES ###################################################################
raw_fname_tmp = '{:s}_main{:d}_sss.fif'
trans_fname_tmp = 'run_{:02d}_sss-trans.fif'
raw_fname_filt_tmp = 'run_{:02d}_filt-%d-%d_sss_raw.fif' % (highpass, lowpass)
# XXX any reason why -eve. but _raw?
mri_fname_tmp = 'run_{:02d}_sss-trans.fif'
events_fname_filt_tmp = 'run_{:02d}_filt-%d-%d_sss-eve.fif' % (highpass,
                                                               lowpass)
fwd_fname_tmp = '{:s}-meg-fwd.fif'
inv_fname_tmp = '{:s}-meg-inv.fif'
cov_fname_tmp = '{:s}-meg-cov.fif'
src_fname_tmp = '{:s}-oct-6-src.fif'


# morph_mat_fname_tmp = '{}-morph_mat.mat'

results_dir = op.join(base_path, 'results')
if not op.exists(results_dir):
    os.mkdir(results_dir)

# SELECTION ###################################################################
from mne.channels import read_ch_connectivity
mag_connectivity, _ = read_ch_connectivity('neuromag306mag')
grad_connectivity, _ = read_ch_connectivity('neuromag306planar')

chan_types = [dict(name='mag', connectivity=mag_connectivity),
              dict(name='grad', connectivity=grad_connectivity),
              dict(name='eeg', connectivity=None)]

# ICA #########################################################################
use_ica = True
eog_ch = ['EOG061', 'EOG062']
ecg_ch = 'ECG063'
n_components = 'rank'
n_max_ecg = 4
n_max_eog = 2
ica_reject = dict(mag=5e-12, grad=5000e-13, eeg=300e-6)
ica_decim = 50

# EVENTS ######################################################################

events_params = [dict(name='stim_lock')]

# EPOCHS ######################################################################
# Generic epochs parameters for stimulus-lock and response-lock
# conditions
event_id = None
cfg = dict(event_id=event_id, decim=4)
# reject=dict(grad=4000e-12, mag=4e-11, eog=180e-5)

# Specific epochs parameters for stim-lock and response-lock conditions,
# depending on whether or not a beaseline is applied
if baseline:
    epochs_stim = dict(name='stim_lock', events='stim', tmin=-0.500,
                       tmax=1.000, baseline=[-.500, -.100], **cfg)
else:
    epochs_stim = dict(name='stim_lock', events='stim', tmin=-0.500,
                       tmax=1.000, baseline=None, **cfg)

epochs_motor1 = dict(name='motor1_lock', events='motor1', tmin=-0.500,
                     tmax=0.200, baseline=None, **cfg)
epochs_params = [epochs_stim, epochs_motor1]

# XXX JRK: Could add second motor response preproc here
# Redefined below to only do stim-locked epochs
# epochs_types = [dict(name='stim_lock'), dict(name='stim_lock-unmasked'),
#                 dict(name='resp_lock'), dict(name='resp_lock-unmasked')]
epochs_types = ['', '-unmasked']

# DECODING ####################################################################
# preprocessing for memory
decoding_preproc_S = dict(decim=8, crop=dict(tmin=0., tmax=0.700))
decoding_preproc_M1 = dict(decim=8, crop=dict(tmin=-0.600, tmax=0.100))
# XXX JRK: Could add second motor response preproc here
decoding_preproc = [decoding_preproc_S, decoding_preproc_M1]

# specify classifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
scaler = StandardScaler()
svc = SVC(C=1, kernel='linear', probability=True, class_weight='auto')
clf = Pipeline([('scaler', scaler), ('svc', svc)])

decoding_params = dict(n_jobs=-1, clf=clf, predict_type='predict_proba')

# FIXME!!!
preproc = decoding_preproc_S

# STATS #######################################################################
clu_sigma = 1e3
clu_n_permutations = 1024
clu_threshold = 0.05


# TO RUN TESTS ################################################################
use_ica = False  # XXX deal with bad chan first
# runs = [1]
epochs_params = [epochs_params[0]]
data_path = '/Volumes/INSERM/data'
# subjects=[subjects[0]]
