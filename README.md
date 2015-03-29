# meg_expectation_p3

## INSTALL
You'll need a bunch of libraries to make it work. There's probably some missing. We'll complete the documentation here

- add export PATH='~/anaconda/bin/:$PATH' to .bash_profile (see http://apple.stackexchange.com/questions/99835/how-to-create-bash-profile-and-profile)

- git
```sudo apt-get install git```

- MNE python (development version):
```
cd <where_you_want_to_put_the_mne_folder>
git clone https://github.com/mne-tools/mne-python
sudo python mne-python/setup.py install
```

- some meeg-preprocessing tools
```
cd <where_you_want_to_put_the_meegpreproc_folder>
git clone https://github.com/dengemann/meeg-preprocessing
sudo python install.py
```

- Then you can download this repository (the meeg_preprocessing folder needs to be in this one or you need to add its location to the paths defined in the scripts - otherwise it won't be recognized)
```
cd <where_you_want_to_put_your_analysis>
git clone https://github.com/kingjr/meg_expectation_p3.git
```

## DEVELOPMENT
The directory structure should look like this

- MEG directory containing MEG data for each subject
./data/MEG/subject1_id
./data/MEG/...
./data/MEG/subjectN_id

- MRI directory containing MR for each subject
./data/subjects/subject1_id/anat_subject1_id.nii
./data/subjects/...
./data/subjects/subjectN_id/anat_subjectN_id.nii

- freesurfer directory containing segmented MR data for each subject
./data/subjects/subject1_id/
./data/subjects/...
./data/subjects/subjectN_id/

- other stuff
./data/behavior/

- scripts
./p300/     all functions that are really specific to your experiment
./scripts/: all functions that are generic to a preprocessing pipeline
            You'll find in there the config.py file in which we ideally
            define all parameters used in scripts/*.py
./tool

- Once you've configured config.py (and p300/conditions.py) run:
```
ipython scripts/config.py # just to check that you don't have bugs here
ipython scripts/run_filtering.py # filtering
ipython scripts/run_ica.py # ica correction for EOG and ECG
ipython scripts/run_extract_events.py # identify triggers
ipython scripts/run_extract_epochs.py # segment stim lock and resposne lock data
ipython scripts/run_evoked_contrast.py # example of a single subject contrast
ipython --pylab tk scripts/run_decoding.py # example of a decoding
```
