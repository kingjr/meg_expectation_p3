function preproc_mask_subtract(subject,gen_path)

disp(['Loading: ' subject])

load([gen_path 'reref/' subject 'EEGreref'],'dataEEG','neighbors')
load([gen_path 'artifacts/' subject], 'eeg_reject_channels','reject_trials')
load([gen_path 'aligned/' subject],'trials')

%% (For now, take the original MEG data, but eventually take after ECG correction?)
load([gen_path 'aligned/' subject])
cfg = [];
cfg.channel = 'MEG';
dataMEG=ft_selectdata(cfg,data);


%% Append the EEG and MEG structures after previous specific preprocessing steps
cfg=[];
data=ft_appenddata(cfg,dataEEG,dataMEG);
data.grad=dataMEG.grad; % grad field gets deleted since not in EEG structure

%% Delete artifacted trials (in case any of them were mask-only)
% but also keep original structures because you want to do the subtraction
% from all trials and to maintain all trials throughout the pipeline

datawoart=data;
trialswoart=trials;
% from M/EEG structure
datawoart.time(reject_trials)=[];
datawoart.trial(reject_trials)=[];
datawoart.trialinfo(reject_trials)=[];
% from behavioral structure
trialswoart(reject_trials)=[];

%% Create a structure with mask-only trials

absent_trials=[]; % identify indices of mask-only trials
for i=1:length(trialswoart)
    if ~trialswoart(i).present
        absent_trials(end+1)=i;
    end
end

cfg.trials=absent_trials; % only keep the trials corresponding to those indices
datamask = ft_selectdata(cfg,datawoart);


%% Prepare Mask

% datamask_orig=datamask;

% figure out what the shortest trial length will be after shifting
% (later keep all trials to that same length to make sure that data from all
% trials is kept at all time points, as opposed to only having data from the
% time points that were shifted small amounts at the end of the average)

maxshift = round(.083/.004);
maxlength=size(datamask.trial{1},2)-maxshift; 

% Create a new data structure with mask-only trials and realign to mask 
% onset (based on variable SOA)

for i=1:length(absent_trials)
    index = absent_trials(i);
    soa = trialswoart(index).soa;
    shift = round(soa/.004); % to account for sampling rate now at 256 Hz
    %disp(shift)
    datamask.trial{i}=datamask.trial{i}(:,(shift+1:shift+maxlength));% shift the data to the left by the SOA (see note above)
    datamask.time{i}=datamask.time{i}(1:maxlength); % cut off the end of the time matrix
end

% compare pre- and post-shift to verify that it was done correctly
% figure; plot(datamask_orig.time{47}, datamask_orig.trial{47}(15,:)) % original trial 47 for EEG15
% figure; plot(datamask.time{47}, datamask.trial{47}(15,:)) % realigned trial 47 for EEG15 (83 ms SOA)
% trialswoart(absent_trials(47)).soa

% Average the mask-only trials to get wave to subtract
cfg = [];
cfg.vartrllength = 0; % make sure that all trials are the same length
avgmask = ft_timelockanalysis(cfg,datamask);
% Note: If using ECG correction for MEG, need to use the 
% preproc_timelockanalysis instead of the FT function.

%% Subtract the average mask from all other trials based on soa
datawomask=data;

for i=1:length(datawomask.trial) % for each trial
    soa = trials(i).soa;
    shift = round(soa/.004); % to account for sampling rate now at 256 Hz

    % at each channel and each time point, subtract avg mask-only activation
    % (shifted based on soa of target-present trial)
    datawomask.trial{i}(:,1+shift:end-21)=...
    datawomask.trial{i}(:,1+shift:end-21)-avgmask.avg(:,1:(end-shift)); 
    % the -21 accounts for the fact that the mask-only trials were shifted
    % and the ends got cut off in the average (by round(.083/.004)=21),
    % making it is a shorter structure.
    
    % cut extra time samples on both ends (where you couldn't do the mask subtraction)
    % beginning: the avg mask is a fixed length and we are shifting it to
    % various degrees based on SOA leaving a small part at the beginning
    % that wasn't subtracted for 16 ms trials and a longer part for 83 ms trials
    datawomask.trial{i}(:,1:22)=[]; % 22 corresponds to shift above, but took maximal value to make all trials the same length
    datawomask.time{i}(1:22)=[];
    % end: couldn't do the subtraction b/c the average mask was shorter than the trial)
    datawomask.trial{i}(:,end-22:end)=[]; % corresponds to the -21 above
    datawomask.time{i}(:,end-22:end)=[];

end

% double check that the difference is equal to the average mask (shifted by
% the SOA)
% figure; plot(data.time{1}, data.trial{11}(48,:)); title ('Before');
% figure; plot(datawomask.time{1}, datawomask.trial{11}(48,:)); title ('After');
% figure; plot(avgmask.time, avgmask.avg(48,:)); title('Average mask');
% figure; plot(datawomask.time{1}, data.trial{11}(48,23:end-23)-datawomask.trial{11}(48,:)); title ('Difference');
% trials(11).soa

disp('Saving')
save ([gen_path 'final/' subject],'data','datawomask','trials','avgmask')

return
