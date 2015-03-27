function preproc_define_trials(subject,gen_path)



for main = 1:4 % Loop across the sss files of individual blocks for this subject 

    file=[gen_path 'sss/' subject '_main' num2str(main) '_sss.fif'];

    %% define trial epochs
    
    cfg = []; % set FT parameters
    cfg.dataset = file;
    cfg.trialdef.eventtype = 'STI101'; % channel with triggers
    
    cfg.trialdef.prestim = 0.5; % take 500 ms before the trigger
    cfg.trialdef.poststim = 1.1; % take 1.1 ms after the trigger (need extra because shift for absent trials)
    cfg.trialdef.eventvalue = [1:60]; % take event codes 1 - 60
    
    cfg = ft_definetrial(cfg);
    % warning about not all EEG channels being digitized because of
    % EEG channel 064 in old version before deleted from montage

    %% preprocess the data channels
    
    %cfg.channel = 'all'
    cfg.continuous = 'yes';
    cfg.lpfilter = 'yes';
    cfg.lpfreq = 30; % 30 Hz low frequency filter
    cfg.demean = 'yes';
    cfg.baselinewindow = [-.200 0]; % 200 ms pre-stimulus baseline correction

    switch main
        case 1
            datab1pp=ft_preprocessing(cfg);
        case 2
            datab2pp=ft_preprocessing(cfg);
        case 3
            datab3pp=ft_preprocessing(cfg);
        case 4
            datab4pp=ft_preprocessing(cfg);
    end
end

%% Append all four blocks into one trial structure

cfg = [];
data = ft_appenddata(cfg, datab1pp, datab2pp, datab3pp, datab4pp);

%% Grad and elec fields

% grad and elec fields are defined for 'new' structures that didn't include EEG064 
% in the recording montage, but they get lost when concatenating 
% all trials together, so need to add them back on. Otherwise, for 'old' structures
% sometimes need to delete EEG064 and then manually define elec and grad 

try % new structure 
    data.grad = [datab1pp.grad];
    data.elec = [datab1pp.elec];
catch % just in case the elec and grad fields aren't in the first block
    cfg = [];
    cfg.channel = {'all' '-EEG064'};
    data=ft_selectdata(cfg,data);
    idx=find(strcmp('EEG064',data.hdr.orig.ch_names));
    data.hdr.orig.chs(idx).kind=0;
    [data.grad, data.elec]=mne2grad(data.hdr);
    disp('I didn''t have a grad or elec field, so they were added')
end

% double check that the elec field no longer includes EEG064 
% (for old structures only; otherwise resampling will not work)
try
data.elec.chanpos(61,:)=[];

data.elec.elecpos(61,:)=[];
data.elec.label(61)=[];
end

%% Resample and save
cfg=[];
cfg.detrend = 'no'; % must be defined; FT: not recommended if looking at evoked responses
data = ft_resampledata(cfg,data); %down sample from 1000 to 256 Hz

save([gen_path 'preprocessed/' subject 'pp'], 'data')

return
