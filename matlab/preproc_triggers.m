function preproc_triggers(subject,gen_path)

%% Load trial and trigger structures
disp(['Loading: ' subject])

load([gen_path 'preprocessed/' subject 'pp'])
triggers = data.trialinfo;

load([gen_path 'behavioral/' subject],'trials');
trial_ttls=[trials.ttl_value];

%% Compute the difference between the trial and trigger values
% 0 means they align, otherwise one or the other will need to be deleted

diff=[];
for x=1:min(length(trial_ttls),length(triggers))
    diff(x)=bsxfun(@minus, triggers(x), trial_ttls(x)); % find the difference between ttl value and trigger value for each trial
end

triggers_remove=[]; % keep track of the index of removed trials and triggers
trials_remove=[];
trials_corrected=trial_ttls; % preserve the original structures, and make all changes to this new one
triggers_corrected=triggers;

i=1;
while i<=length(diff) % while there are still values in both matrices to be compared
    disp(i)
    
    if diff(i)~=0 % if the ttl value and trigger value differ, there's a problem
        deleted_trial=0; % keep track of whether or not a fix has been made

        if i<length(trials_corrected) % check for an extra trial (if there is one to the right to compare w/ current trigger)
            if trials_corrected(i+1)==triggers_corrected(i) % extra trial
                trials_corrected(i)=[]; %remove the extra trial
                trials_remove(end+1)=i+length(trials_remove); % index it in the original trial structure
                deleted_trial=1; % note that a change was made
            end
        end

        if i<length(triggers_corrected) && deleted_trial ==0 %if that didn't work, check for an extra trigger
                if triggers_corrected(i+1)==trials_corrected(i)% extra trigger
                    triggers_corrected(i)=[]; % remove the extra trigger
                    triggers_remove(end+1)=i+length(triggers_remove); % index it in the original matrix structure
                    deleted_trial=1; % note that a change was made
                end
        end
        
        if deleted_trial == 0 % no trial or trigger removed b/c neither i+1 matches
            disp('There is no +1 fix, please align manually!')
            break
        end
        
        diff=[]; %recalculate the difference after removing an erroneous value
        for x=1:min(length(trials_corrected),length(triggers_corrected))
            diff(x)=bsxfun(@minus, triggers_corrected(x), trials_corrected(x));
        end
            
    elseif diff(i)==0  % if values match, move on to next index
        i=i+1;
    end
end

%% Remove extra triggers or trials on the end of the structure

if length(triggers_corrected)<length(trials_corrected) % extra trials
    start_index = length(triggers_corrected)+1+length(trials_remove); 
    %length(trig_corr)+1 is index of first extra value in corrected 
    %structure, need to transform that into index in original structure
    end_index = length(trials_corrected)+length(trials_remove);
    trials_remove=[trials_remove start_index:end_index];
elseif length(triggers_corrected)>length(trials_corrected) %extra triggers
    start_index = length(trials_corrected)+1+length(triggers_remove); % same reasoning as above
    end_index = length(triggers_corrected)+length(triggers_remove);
    triggers_remove=[triggers_remove start_index:end_index];
end

%% Double check w/ original structure indices before saving

diff=[]; %recalculate the difference in the original structures after removing erroneous values
triggers(triggers_remove)=[];
trial_ttls(trials_remove)=[];
for x=1:min(length(trials_corrected),length(triggers_corrected))
    diff(x)=bsxfun(@minus, triggers_corrected(x), trials_corrected(x));
end

plot([diff],'o') % double check that the final structures are aligned

disp(['Finished: ' subject]);
disp(trials_remove);
disp(triggers_remove);

%% Remove the mismatched trials
trials(trials_remove)=[];
data.time(triggers_remove)=[];
data.trial(triggers_remove)=[];
data.trialinfo(triggers_remove)=[];


save([gen_path 'aligned/' subject],'trials','data','trials_remove','triggers_remove')

