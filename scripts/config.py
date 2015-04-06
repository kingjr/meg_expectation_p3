import os
import os.path as op

# PATHS ########################################################################
base_path = op.dirname(op.dirname(__file__))

data_path = op.join(base_path, 'data')
#data_path = '/Volumes/INSERM/data'
data_path = '/media/jrking/INSERM/data' # XXX to be changed

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
# s14_ac130389, s17_ft120490 have two sss for certain blocks
# XXX JRK: This should be solvable!

exclude_subjects = ['s19_cd110147', 's15_nv110179'] # maxfilter error

subjects = [s for s in subjects if s not in exclude_subjects]

runs = list(range(1, 5))  # number of runs per subject

# FILRERING ####################################################################
lowpass = 35
highpass = 0.75
filtersize = 16384

# FILENAMES ####################################################################
raw_fname_tmp = '{:s}_main{:d}_sss.fif'
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

events_params = dict()

# EPOCHS #######################################################################
# Generic epochs parameters for stimulus-lock and response-lock
# conditions
event_id = None
cfg = dict(event_id=event_id, decim=4)
# reject=dict(grad=4000e-12, mag=4e-11, eog=180e-5)

# Specific epochs parameters for stim-lock and response-lock conditions
epochs_stim = dict(name='stim_lock', events='stim', tmin=-0.500,
                   tmax=1.000, baseline=[-.500, -.100], **cfg)
epochs_motor1 = dict(name='motor1_lock', events='motor1', tmin=-0.500,
                   tmax=0.200, baseline=None, **cfg)
# XXX JRK: Could add second motor response preproc here
epochs_params = [epochs_stim, epochs_motor1]


# MAIN CONTRASTS ###############################################################
# Here define your contrast of interest
contrasts = (
            dict(name='present_absent',
                 include=dict(present=[True, False]),
                 exclude=dict()),
            dict(name='seen_unseen',
                 include=dict(seen=[True, False]),
                 exclude=dict()),
            dict(name='pas',
                 include=dict(pas=[0, 1, 2, 3]),
                 exclude=dict()),
            dict(name='global',
                 include=dict(block=['visible', 'invisible']),
                 exclude=dict(soa=[17, 83])),
            dict(name='local',
                 include=dict(local_context=['S', 'U']),
                 exclude=dict()),
            dict(name='soa',
                 include=dict(soa=[17, 33, 50, 67, 83]),
                 exclude=dict()),
            dict(name='seen_X_soa',
                 include=dict(seen_X_soa=[seen + str(soa)
                                for seen in ['seen_', 'unseen_']
                                    for soa in [17, 33, 50, 67, 83]]),
                exclude=dict()),
            dict(name='motor1_finger',
                 include=dict(motor1=['left', 'right']),
                 exclude=[dict(missed_m1=True)]),
            )


# DECODING #####################################################################
# preprocessing for memory
decoding_preproc_S = dict(decim=2, crop=dict(tmin=0., tmax=0.700))
decoding_preproc_M1 = dict(decim=2, crop=dict(tmin=-0.600, tmax=0.100))
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


# STATS ########################################################################
clu_sigma = 1e3
clu_n_permutations = 1024
clu_threshold = 0.05


# TO RUN TESTS
subjects = [subjects[0]]
# runs = [runs[0]]
use_ica = False
