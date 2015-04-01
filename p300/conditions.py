
import mne
import numpy as np
import warnings
import scipy.io as sio

def events_select_condition(trigger, condition):
    """Function to handle events and event ids

    Parameters
    ----------
    trigger : np.array (dims = [n,1])
        The trigger values
    condition : str
        The set of events corresponding to a specific analysis.

    Returns
    -------
    selection : np.array (dims = [m, 1])
        The selected trigger.
    """
    if condition == 'stim_motor':
        selection = np.where(trigger > 0)[0]
    elif condition == 'stim':
        selection = np.where((trigger > 0) & (trigger <=60))[0]
    elif condition == 'motor':  # remove response not linked to stim
        selection = np.where(trigger > 60)[0]
    return selection

def get_events(events):
    """ Define trigger condition correspondance here"""
    import pandas as pd

    n_events = events.shape[0]

    # initialize future data frame
    events_data_frame = list()

    # Loop across all trials
    for ii in range(n_events): ## range does not include the listed max - does this matter here?
        event = dict()

        # Trigger order per trial: ttl_value, letter/number, PAS

        # Event type: stimulus or response trigger
        if events[ii, 2] < 64:
            event['type'] = 'stim'
        else:
            event['type'] = 'motor'

        # Stimulus category (absent/letter/digit)
        # XXX The stim definition may be simplifiable in base 2?
        if event['type'] == 'stim':
            if events[ii,2] in (range(1,5) or range(13,17) or range(25,29) or range(37,41) or range(49,53)):
                # 1-4 13-16 25-28 37-40 49-52
                event['target'] = 'absent'
            if events[ii,2] in (range(5,9) or range(17,21) or range(29,33) or range(41,45) or range(53,57)):
                # 5 7 17-20 29-32 41-44 54 56
                event['target'] = 'letter'
            if events[ii,2] in (range(9,13) or range(21,25) or range(33,37) or range(45,49) or range(57,60)):
                # 9 11 21-24 33-36 45-48 58 60
                event['target'] = 'number'
        else:
            event['target'] = 'none'

        # SOA for target-present trials
        if event['type'] == 'stim':
            if events[ii,2] in range (5,13):
                # 5, 7, 9, 11 (can be 5-12)
                event['soa'] = '16'
            if events[ii,2] in range(17,25):
                # 17-24
                event['soa'] = '33'
            if events[ii,2] in range(29,37):
                # 29-36
                event['soa'] = '50'
            if events[ii,2] in range(41,49):
                # 41-48
                event['soa'] = '66'
            if events[ii,2] in range(53,61):
                # 54,56,58,60 (can be 53-60)
                event['soa'] = '83'
        else:
            event['soa'] = 'none'

        # PAS
        if event['type'] == 'stim':
            if events[ii+2,2] == (2 ** 6):
                # 64
                event['pas'] = '1'
            if events[ii+2,2] == (2 ** 7):
                # 128
                event['pas'] = '2'
            if events[ii+2,2] == (2 ** 8):
                # 256
                event['pas'] = '3'
            if events[ii+2,2] == (2 ** 9):
                # 512
                event['pas'] = '4'

        else:
            event['pas'] = 'none'


        # Letter response (which right hand finger corresponds to letter response)
        if event['type'] == 'stim':
            if events[ii,2] in [1,2,5,6,9,10,13,14,17,18,21,22,25,26,29,30,33,34,37,38,41,42,45,46,49,50,53,54,57,58]:
                event['letter_resp'] ='left'
            if events[ii,2] in [3,4,7,8,11,12,15,16,19,20,23,24,27,28,31,32,35,36,39,40,43,44,47,48,51,52,55,56,59,60]:
                event['letter_resp'] = 'right'
        else:
            event['letter_resp'] = 'none'

        # Accuracy
        if event['type'] == 'stim':
            if event['letter_resp'] == 'right':
                if event['target'] == 'letter':
                    if events[ii+1,2] == (2 ** 13): #right response
                        event['accuracy'] = 'correct'
                    if events[ii+1,2] == (2 ** 14): #left response
                        event['accuracy'] = 'incorrect'
                if event['target'] == 'number':
                    if events[ii+1,2] == (2 ** 14):
                        event['accuracy'] = 'correct'
                    if events[ii+1,2] == (2 ** 13):
                        event['accuracy'] = 'incorrect'
            if event['letter_resp'] == 'left':
                if event['target'] == 'letter':
                    if events[ii+1,2] == (2 ** 14):
                        event['accuracy'] = 'correct'
                    if events[ii+1,2] == (2 ** 13):
                        event['accuracy'] = 'incorrect'
                if event['target'] == 'number':
                    if events[ii+1,2] == (2 ** 13):
                        event['accuracy'] = 'correct'
                    if events[ii+1,2] == (2 ** 14):
                        event['accuracy'] = 'incorrect'
        else:
            event['accuracy'] = 'none'

        # Block
        if event['type'] == 'stim':
            if (events[ii,2]%2==1):
                #odd numbers
                event['block']='invisible'
            if (events[ii,2]%2==0):
                #even numbers
                event['block']='visible'
        else:
            event['block'] = 'none'


        # Local context
        '''
        if event['type'] == 'stim':
            if  :
                event['local_context'] = 'UU'
            if  :
                event['local_context'] = 'SU'
            if  :
                event['local_context'] = 'US'
            if  :
                event['local_context'] = 'SS'
        else:
            event['local_context'] = 'none'
        '''

        # Motor side
        if event['type'] == 'motor':
            if events[ii,2] > 1000:
                event['motor_side'] = 'right'
                # 8192, 16384 - objective forced choice responses
            if events[ii,2] < 1000:
                event['motor_side'] = 'left'
                # 64, 128, 256, 512 - PAS responses
        else:
            event['motor_side'] = 'none'



        # Right hand response (index or middle finger)
        if event['type'] == 'motor':
            if events[ii, 2] == (2 ** 13):
                event['right_hand'] = 'right'
            if events[ii, 2] == (2 ** 14):
                event['right_hand'] = 'left'
        else:
            event['right_hand'] = 'none'

        # Stimulus side (top or bottom)
        ## uh oh, this isn't coded directly in the ttls, so we would need the
        ## behavioral structure if we want to make this contrast



        events_data_frame.append(event)

    return pd.DataFrame(events_data_frame)

def extract_events(raw):
    """This is a hand-made extraction of the events so as to look for all stim
    triggers, find the two following motor responses, and combined their trigger
    values to know that they go in triplets."""

    events = mne.find_events(raw, stim_channel='STI101', verbose=True,
                             consecutive='increasing', min_duration=0.003)

    # Define stim and motor triggers
    stim  = range(1, 64)
    motor = [64, 128, 256, 512, 8192, 16384]

    # Identify # of stim and motor events
    stim_inds = np.where([i in stim for i in events[:, 2]])[0]
    motor_inds = np.where([i in motor for i in events[:, 2]])[0]

    # For each stimulus trigger, find the corresponding motor responses, and
    # assign them combined values:
    new_events = list()
    for s in stim_inds:
        # find following triggers
        inds = np.where(events[:, 0] > events[s, 0])[0]
        # only keep those that are motor responses
        inds = np.intersect1d(inds, motor_inds)
        if len(inds) < 2:
            raise(RuntimeError('< 2 motor responses after stim %s !' % ind))

        # Combine stimulus and response trigger values so as to get both
        # stimulus and motor information in each event.
        combined = events[s, 2] + events[inds[0], 2] + events[inds[1], 2]

        # Add ttl value to explicitely differentiate the three events
        extra_ttl = 2 ** (np.log2(np.max(motor + stim)) + 1)
        # 1st: stim
        new_events.append([events[s, 0], 0, combined + 0 * extra_ttl])
        # 2nd: first motor response
        new_events.append([events[inds[0], 0], 0, combined + 1 * extra_ttl])
        # 3rd: second motor response
        new_events.append([events[inds[1], 0], 0, combined + 2 * extra_ttl])

    return np.array(new_events)
