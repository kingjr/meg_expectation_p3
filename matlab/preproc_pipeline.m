%% Define libraries

gen_path ='/Volumes/INSERM/data/';
addpath('/Users/Gabriela/Documents/2014-2015/2015 Spring/Neuroimaging II/matlab/fieldtrip-20150318/');

addpath([gen_path '../scripts/JR_toolbox/']); % for general use
addpath([gen_path '../scripts/JR_toolbox/ft_jr_art/']); % specific to artefact corrections
addpath([gen_path '../scripts/'])

% subjects={'s4_sa130042','s5_sg120518','s6_sb120316','s7_jm100109','s8_pe110338','s9_df130078','s10_ns110383','s11_ts100368','s12_aa100234','s13_jn120580','s14_ac130389','s16_mp130429','s17_ft120490','s18_rg110386','s20_ad120286','s21_jl130434','s22_sl130503','s23_pf120155','s24_cl120289','s25_bb100103','s26_sb130354'};
% not included: s15_nv110179 (MaxFilter), s19_cd110147 (MaxFilter)

subjects = {'s23_pf120155'}; % example subject

%% Run Maxfilter

% % Runs Maxfilter, automatically detecting bad channels and returns an SSS
% % file for each block that is saved in the sss folder
% 
% for s = 1:length(subjects)
%     subject=subjects{s}; %%need to use appropriate subject name with underscore
%     preproc_maxfilter(subject)
% end

%% Define trials and concatenate trials across blocks

% Preprocess the data, define trials, and append for one full data
% structure, saved in preprocessed folder

% Need to do manually for s14_ac130389 and s17_ft120490 because more than
% one SSS file per block (due to long blocks)

for s=1:length(subjects)
    subject.nip=subjects{s};
    preproc_define_trials(subject.nip, gen_path);
end


%% Check for extra or missing triggers

% Realign triggers and ttl_values so that behavioral and M/EEG
% structures are aligned, saved to aligned folder

for s=1:length(subjects)
    subject.nip=subjects{s};
    preproc_triggers(subject.nip, gen_path);
end


%% Artifact rejection: preproc_artifacts

% Manually discard trials with blinks or otherwise noisy trials
% Save these trials and noisy channels as structures in artifacts folder

%% Interpolate to fill in missing EEG electrodes & rereference to common average

% Uses eeg_bad_channels and neighborhood structure for each individual
% Saves a final EEG structure to reref folder

for s=1:length(subjects)
    subject=subjects{s};
    preproc_EEG(subject,gen_path)
end

%% ECG correction??


%% Mask Subtraction

for s=1:length(subjects)
    subject=subjects{s};
    preproc_mask(subject,gen_path)
end


%%%%%%%%%%%%%%% STOP HERE %%%%%%%%%%%%%%%%%%%%%

conditions={'16', '33', '50', '66', '83', 'pas1', 'pas2', 'pas3', 'pas4', '50invisible', '50visible', 'visblock_stan','visblock_dev', 'invisblock_stan','invisblock_dev','visblock_stan_pas','visblock_dev_pas', 'invisblock_stan_pas','invisblock_dev_pas','visblock_stan_pas2','visblock_dev_pas2', 'invisblock_stan_pas2','invisblock_dev_pas2','visblock_stan_50','visblock_dev_50', 'invisblock_stan_50','invisblock_dev_50'};
conditions={'50invisible_womask', '50visible_womask'};


%% Separate by condition and average
for s=1:length(subjects)
    subject=subjects{s};
    preproc_conditions(subject,gen_path,conditions,'MEG')
    preproc_conditions(subject,gen_path,conditions,'EEG')
end

%% Grand averages across subjects

preproc_grand_average(subjects,gen_path,conditions, 'MEG');
preproc_grand_average(subjects,gen_path,conditions, 'EEG');

%% Decoding