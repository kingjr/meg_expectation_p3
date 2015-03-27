import os
import os.path as op

# PATHS ########################################################################
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

# REPORT #######################################################################
open_browser = True

# SUBJECTS #####################################################################
subjects = ['s10_ns110383', 's13_jn120580', 's16_mp130429', 's19_cd110147',
            's22_sl130503', 's25_bb100103', 's5_sg120518', 's8_pe110338',
            's11_ts100368', 's14_ac130389', 's17_ft120490', 's20_ad120286',
            's23_pf120155', 's26_sb130354', 's6_sb120316', 's9_df130078',
            's12_aa100234', 's15_nv110179', 's18_rg110386', 's21_jl130434',
            's24_cl120289', 's4_sa130042', ' s7_jm100109']
# not included: s15_nv110179 (MaxFilter), s19_cd110147 (MaxFilter)

subjects = ['s23_pf120155'] # For now, only run on on subject

exclude_subjects = []

subjects = [s for s in subjects if s not in exclude_subjects]

runs = list(range(1, 2))  # number of runs per subject

# FILRERING ####################################################################
lowpass = 35
highpass = 0.75
filtersize = 16384

# FILENAMES ####################################################################
raw_fname_tmp = '{:s}_main_{:d}_sss.fif'
trans_fname_tmp = 'run_{:02d}_sss-trans.fif'
raw_fname_filt_tmp = 'run_{:02d}_filt-%d-%d_sss_raw.fif' % (highpass, lowpass)
# XXX any reason why -eve. but _raw?
mri_fname_tmp = 'run_{:02d}_sss-trans.fif'
events_fname_filt_tmp = 'run_{:02d}_filt-%d-%d_sss-eve.fif' % (highpass, lowpass)
fwd_fname_tmp = '{:s}-meg-fwd.fif'
inv_fname_tmp = '{:s}-meg-inv.fif'
cov_fname_tmp = '{:s}-meg-cov.fif'
src_fname_tmp = '{:s}-oct-6-src.fif'


# morph_mat_fname_tmp = '{}-morph_mat.mat'

results_dir = op.join(base_path, 'results')
if not op.exists(results_dir):
    os.mkdir(results_dir)

# SELECTION ####################################################################
ch_types_used = ['meg', 'eeg']

# ICA ##########################################################################
use_ica = True
eog_ch = ['EOG061', 'EOG062']
ecg_ch = 'ECG063'
n_components = 'rank'
n_max_ecg = 4
n_max_eog = 2
ica_reject = dict(mag=5e-12, grad=5000e-13, eeg=300e-6)
ica_decim = 50

# EVENTS #######################################################################
# Exceptional case: subject06 run 9 had a trigger test within the
# recording, need to start collecting events after that/
events_params = dict(subject06_ha=[dict()] * 9 +
                                  [dict(first_sample=140000)])

# EPOCHS #######################################################################
# Generic epochs parameters for stimulus-lock and response-lock
# conditions
event_id = None
cfg = dict(event_id=event_id, decim=4)
# reject=dict(grad=4000e-12, mag=4e-11, eog=180e-5)

# Specific epochs parameters for stim-lock and response-lock conditions
epochs_stim = dict(name='stim_lock', events='stim', tmin=-0.500,
                   tmax=1.100, baseline=[-.500, -.100], **cfg)
epochs_resp = dict(name='motor_lock', events='motor', tmin=-0.500,
                   tmax=0.200, baseline=None, **cfg)
epochs_params = [epochs_stim, epochs_resp]

# COV ##########################################################################
cov_method = ['shrunk', 'empirical']

# INVERSE ######################################################################
fsave_grade = 4
# fwd_fname_tmp = 'sub{:02d}-meg-oct-6-fwd.fif' # XXX check file name
make_inverse_params = {'loose': 0.2,
                       'depth': 0.8,
                       'fixed': False,
                       'limit_depth_chs': True}
snr = 3
lambda2 = 1.0 / snr ** 2
apply_inverse_params = {'method': "dSPM", 'pick_ori': None,
                        'pick_normal': None}

# MAIN CONTRASTS ###############################################################
# Here define your contrast of interest
contrasts = (
            dict(name='motor_side',
                 include=dict(cond='motor_side', values=['left', 'right']),
                 exclude=[]),
            )


# DECODING #####################################################################
# preprocessing for memory
decoding_preproc_S = dict(decim=2, crop=dict(tmin=0., tmax=0.700))
decoding_preproc_M = dict(decim=2, crop=dict(tmin=-0.600, tmax=0.100))
decoding_preproc = [decoding_preproc_S, decoding_preproc_M]

# specify classifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
scaler = StandardScaler()
svc = SVC(C=1, kernel='linear', probability=True, class_weight='auto')
clf = Pipeline([('scaler', scaler), ('svc', svc)])

decoding_params = dict(n_jobs=-1, clf=clf, predict_type='predict_proba')


# STATS ########################################################################
clu_sigma = 1e3
clu_n_permutations = 1024
clu_threshold = 0.05
