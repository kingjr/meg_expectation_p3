load('/neurospin/meg/meg_tmp/2013_Surprise/data/aa100234_maskdata.mat','datamaskEEG')
ft2mat = @(data) permute(reshape(cell2mat(data.trial),[size(data.trial{1}) length(data.trial)]),[3 1 2]);
time = datamaskEEG.time{1}(1:380);
datamaskEEG.trial = cellfun(@(x) x(:,1:380), datamaskEEG.trial,'uniformoutput',false);
datamask = ft2mat(datamaskEEG);

%% test: change baseline, change preprocessing (filtering), plot all soa for one subject
load([gen_path 'preprocessed/rg110386alldatapp'], 'data_nobsl')
load('/neurospin/meg/meg_tmp/2013_Surprise/data/behavior/rg110386.mat','trials')
figure();clf;hold on;
plot([trials([1:197 199:end]).ttl_value],'r')
plot(data_nobsl.trialinfo','b')
trials(198) = []; % remove extra trial;

% baseline correction
time = data_nobsl.time{1};
erf = ft2mat(data_nobsl);
bsl = mean(median(erf(:,:,time>-.500 & time < -.100),1),3);
erf_bsl = erf - repmat(bsl,[1247 1 359]);

% prepare layout
cfg = [];
cfg.layout= 'eeg_64_NM20884N.lay';% plot
layout = ft_prepare_layout(cfg);

% plot options
cfg_plot= [];
cfg_plot.zlim = [-3 3]*1e-6;
cfg_plot.layout = layout;
cfg_plot.fontsize=12;
cfg_plot.marker='off';
cfg_plot.label = ft_EEG.label;
cfg_plot.highlightchannel =  'EEG048';
cfg_plot.highlight          = 'on';
%% Plot all SOAs
soas = [.016 .033 .050 .066 .083];
toi = .05:0.017:0.3;% zoom
coi = 323:382;
figure('Color',[1 1 1]);clf;
for soa = 1:length(soas)
    for t = 1:length(toi)
        t
        % select subplot
        subaxis(length(soas),length(toi),(soa-1)*length(toi)+t, 'SpacingHorizontal',0, 'SpacingVertical',0);
        % select time & soa
        sel = [trials.soa]==soas(soa) & [trials.present]==1;
        topo = erf_bsl(sel,coi,find((time-.04)>=toi(t),1)+[-3:3]);
        % mean across subjects
        topo = squeeze(mean(topo,1));
        % plot
        my_plot_topo(topo, cfg_plot);
        if soa == 1
            title(toi(t));
        end
    end
end

clf;hold on;
colors = colorGradient([1 0 0],[0 1 0],5);
for soa = 1:5
    sel = [trials.soa]==soas(soa) & [trials.present]==0;
    %plot_eb(time,squeeze(erf_bsl(sel,382,:)),colors(soa,:));
    plot(time,mean(squeeze(erf_bsl(sel,382,:))),'linewidth',2,'color',colors(soa,:));
end

clf;hold on;
coi = 323:382;
for soa = 1:5
    sel = [trials.soa]==soas(soa) & [trials.present]==0;
subplot(5,1,soa)
    %plot_eb(time,squeeze(erf_bsl(sel,382,:)),colors(soa,:));
    imagesc(time,[],squeeze(median(squeeze(erf_bsl(sel,coi,:)))),[-1 1]*10^-5);
hold on;
plot([.160 .160],ylim,'k');
end


%% Plot all ERP before mask subtraction, mean across all subjects. target + mask and maks only
% so as to see whether we can detect a temporal shift of the erp according
% to the soa
subjects={'ad120286', 'rg110386','sa130042','pe110338','ts100368','ns110383','aa100234','jn120580', 'ac130389','bb100103','mp130429','sb130354','ft120490'};
mean_erp = [];
for s = length(subjects):-1:1
s
% load([gen_path 'preprocessed/' subjects{s}], 'data')
% load(['/neurospin/meg/meg_tmp/2013_Surprise/data/behavior/' subjects{s}],'trials')
load([gen_path 'reref/' subjects{s} 'EEGreref.mat'],'dataEEG', 'trials');

ft2mat = @(data) permute(reshape(cell2mat(data.trial),[size(data.trial{1}) length(data.trial)]),[3 1 2]);
time = dataEEG.time{1};

x = ft2mat(dataEEG);

for soa = 5:-1:1
    % mask only
    sel = [trials.soa]==soas(soa) & [trials.present]==0;
    mean_erp(:,:,s,soa,1) = squeeze(median(x(sel,:,:)));
    % target + mask
    sel = [trials.soa]==soas(soa) & [trials.present]==1;
    mean_erp(:,:,s,soa,2) = squeeze(median(x(sel,:,:)));
    
end
end


figure;clf;hold on;set(gcf,'color','w');
clim = [-5 5]*10^-6;
for soa = 1:5
    % mask only
    subplot(5,3,3*soa-2)
    imagesc(time-.040,[],squeeze(mean(mean_erp(:,:,:,soa,1),3)),clim);
axis([-.050 max(time-.040) ylim]);
hold on;plot([0 0],ylim,'k');grid on;
    % target + mask 
    subplot(5,3,3*soa-1)
    imagesc(time-.040,[],squeeze(mean(mean_erp(:,:,:,soa,2),3)),clim);
axis([-.050 max(time-.040) ylim]);
hold on;plot([0 0],ylim,'k');grid on;
% target + mask - mask
    subplot(5,3,3*soa)
    imagesc(time-.040,[],squeeze(mean(mean_erp(:,:,:,soa,2)-mean_erp(:,:,:,soa,1),3)),clim);
axis([-.050 max(time-.040) ylim]);
hold on;plot([0 0],ylim,'k');grid on;
end
export_fig('/neurospin/meg/meg_tmp/2013_Surprise/data/Topos/finals/eeg_soa_withMaskWithoutMask.png');


