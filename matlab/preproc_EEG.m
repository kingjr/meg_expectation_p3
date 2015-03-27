function preproc_EEG(subject,gen_path)

clearvars -EXCEPT gen_path subject

%% Load data

disp(['Loading: ' subject])

load([gen_path 'aligned/' subject])
load([gen_path 'artifacts/' subject], 'eeg_reject_channels')

%% Define neighbors 
% Individually, for each subject - need for interpolation of bad channels

dataEEG=rmfield(data,'grad'); % remove .grad field to be able to interpolate EEG channels

cfg = [];
cfg.method ='distance';
%cfg.method = 'template';
cfg.layout = 'eeg_64_NM20884N.lay';
cfg.channel = 'EEG*';
%cfg.channel = 'all';
cfg.feedback = 'yes';
cfg.neighbourdist=5; %%% Is this a good setting??
neighbors=ft_prepare_neighbours(cfg, dataEEG);


%% Interpolate bad channels (defined in artifact rejection)

channels={}; % get a structure with bad channels that works for ft_channelrepair - there must be a way to do this without a loop??
for i=1:length(eeg_reject_channels)
    channels(end+1)=eeg_reject_channels{i};
end

disp(channels)

cfg=[]; 
cfg.badchannel= channels;
cfg.neighbours=neighbors;
cfg.method = 'spline';
dataEEG=ft_channelrepair(cfg,dataEEG);

%% Common average rereference

disp('Rereferencing')

for i=1:length(dataEEG.trial)
    dataEEG.trial{i}=ft_preproc_rereference(dataEEG.trial{i});
end

disp('Saving')

save([gen_path 'reref/' subject 'EEGreref'], 'dataEEG', 'neighbors')


return

