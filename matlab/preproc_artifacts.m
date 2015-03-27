load([gen_path 'aligned/' subject], 'data')

cfg = [];
cfg.channel = 'EOG061';% blinks
%cfg.method = 'trial';
%cfg.method ='channel';
cfg.method = 'summary'; % I used to use this, but it doesn't seem to work
%for this channel in the latest version of fieldtrip. I'm not going to
%spend too much time figuring it out right now, since we might get rid of 
%blinks with an ICA anyways.
cfg.alim = 1e-11;
cfg.latency=[0 0.7]; %%% even with a short artifact rejection epoch like
% this, we way too many trials to blinks in a number of subjects (because
% if they responded fast and blinked immediately after, they fall in this
% time window
cfg.eogscale = 1e-8;
cfg.keepchannel='yes';
dummy=ft_rejectvisual(cfg,data);

eog_reject_trials=[]; %there must be a more efficient way to do this than 
% to copy the trials from the output in the command window into this trial
% structure (and then to concatenate them all at the end into one structure
% with all of the artifacted trials)??

cfg = [];
%cfg.channel = 'MEGMAG';
%cfg.channel = 'MEGGRAD';
cfg.channel = 'EEG';
%cfg.method = 'trial';
cfg.method = 'summary';
cfg.alim = 1e-11;
cfg.megscale = 1;
cfg.eegscale =8e-8;
cfg.latency=[0 0.9];
cfg.keepchannel='no';
dummy=ft_rejectvisual(cfg,data);

eeg_reject_trials=[14, 16, 18, 25, 80, 82, 136, 156, 168, 169, 280, 281, 316, 338, 353, 366, 405, 414, 445, 447, 452, 461, 462, 464, 467, 469, 502, 508, 515, 516, 518, 521, 539, 540, 545, 551, 552, 561, 565, 573, 574, 575, 587, 600, 609, 613, 614, 615, 616, 617, 619, 621, 623, 627, 664, 673, 690, 697, 698, 699, 707, 720, 730, 757, 758, 759, 760, 761, 770, 774, 789, 792, 793, 794, 797, 798, 818, 908, 932, 949, 952, 953, 955, 956, 964, 974, 976, 977, 983, 990, 1003, 1019, 1020, 1021, 1035, 1042, 1052, 1053, 1061, 1062, 1064, 1082, 1110, 1125, 1130, 1131, 1133, 1135, 1136, 1137, 1143, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1213, 1216, 1218, 1220, 1221, 1222, 1223, 1224, 1231, 1238, 1239];
eeg_reject_channels=[];
for i=1:length(dummy.cfg.channel)
    if ~ismember(dummy.cfg.channel(i), dummy.label)
        eeg_reject_channels{end+1}=dummy.cfg.channel(i);
    end
end

%% For real we will need to do this, but I've just been doing EEG for now
% 
% mag_reject_trials=[];
% mag_reject_channels=[];
% for i=1:length(dummy.cfg.channel)
%     if ~ismember(dummy.cfg.channel(i), dummy.label)
%         mag_reject_channels{end+1}=dummy.cfg.channel(i);
%     end
% end
% 
% grad_reject_trials=[];
% grad_reject_channels=[];
% for i=1:length(dummy.cfg.channel)
%     if ~ismember(dummy.cfg.channel(i), dummy.label)
%         grad_reject_channels{end+1}=dummy.cfg.channel(i);
%     end
% end
% 
% meg_reject_trials=union(mag_reject_trials,grad_reject_trials);
% eeg_eog_reject_trials=union(eeg_reject_trials, eog_reject_trials);
% reject_trials=union(eeg_eog_reject_trials,meg_reject_trials);
% 
% reject_channels=[eeg_reject_channels mag_reject_channels grad_reject_channels];

% save([gen_path 'artifacts/' subject], 'eeg_r*', 'eog*', 'mag_r*', 'grad_r*', 'reject*') 
%% Since we just have EEG for now
reject_channels = eeg_reject_channels;
reject_trials = eeg_reject_trials;

save([gen_path 'artifacts/' subject], 'eeg_r*', 'reject*') 

