
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
        selection = np.where((trigger > 0) & (trigger < 2 ** 15))[0]
    elif condition == 'motor':  # remove response not linked to stim
        selection = np.where(trigger >= 2 ** 15)[0]
    elif condition == 'motor1':  # remove response not linked to stim and 2nd motor response
        selection = np.where((trigger >= 2 ** 15) & (trigger < 2 ** 16))[0]
    elif condition == 'motor2':  # remove response not linked to stim
        selection = np.where(trigger >= 2 ** 16)[0]
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

        # Event type: stimulus, 1st or 2nd response
        if events[ii, 2] < (2 ** 15):
            event['type'] = 'stim'
        elif events[ii, 2] < (2 ** 16):
            event['type'] = 'motor1'
        else:
            event['type'] = 'motor2'

        # Extract trigger values of combined stim, first motor and second motor
        trigger = events[ii, 2] % (2 ** 15)
        trigger_stim = trigger % 64
        trigger_motor2 = trigger % 1024 - trigger_stim
        trigger_motor1 = trigger - trigger_stim - trigger_motor2


        # Stimulus category (absent/letter/digit)
        # XXX JRK: The stim definition should be simplifiable in base 2, to
        # avoid the possible introduction of errors
        if trigger_stim in (range(1,5) or range(13,17) or range(25,29) or
                            range(37,41) or range(49,53)):
            # 1-4 13-16 25-28 37-40 49-52
            event['target'] = 'absent'
        if trigger_stim in (range(5,9) or range(17,21) or range(29,33) or
                            range(41,45) or range(53,57)):
            # 5 7 17-20 29-32 41-44 54 56
            event['target'] = 'letter'
        if trigger_stim in (range(9,13) or range(21,25) or range(33,37) or
                            range(45,49) or range(57,60)):
            # 9 11 21-24 33-36 45-48 58 60
            event['target'] = 'number'
        else:
            # XXX JRK: Always add else conditions
            event['target'] = None  # is it an error? Else defined it

        # SOA for target-present trials
        if trigger_stim in range (5,13):
            # 5, 7, 9, 11 (can be 5-12)
            event['soa'] = 17
        if trigger_stim in range(17,25):
            # 17-24
            event['soa'] = 33
        if trigger_stim in range(29,37):
            # 29-36
            event['soa'] = 50
        if trigger_stim in range(41,49):
            # 41-48
            event['soa'] = 67
        if trigger_stim in range(53,61):
            # 54,56,58,60 (can be 53-60)
            event['soa'] = 83
        else:
            event['soa'] = None  # XXX JRK:

        # Forced_choice response (which right hand finger corresponds to letter
        # response)
        if trigger_stim in [1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26,
                            29, 30, 33, 34, 37, 38, 41, 42, 45, 46, 49, 50, 53,
                            54, 57, 58]:
            event['letter_resp'] = 'left'
        if trigger_stim in [3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23, 24, 27, 28,
                            31, 32, 35, 36, 39, 40, 43, 44, 47, 48, 51, 52, 55,
                            56, 59, 60]:
            event['letter_resp'] = 'right'
        else:
            event['letter_resp'] = None # XXX JRK : check

        # motor response
        if trigger_motor1 == (2 ** 13):
            event['motor1'] = 'right'
        elif trigger_motor1 == (2 ** 14):
            event['motor1'] = 'left'
        else:
            event['motor1'] = None

        # Accuracy
        # XXX JRK Check what I did, I try to simply but I'm not 100% sure
        if event['motor1'] is not None:
            event['missed_m1'] = False
            if event['target'] == 'letter':
                event['correct'] = event['letter_resp'] == event['motor1']
            elif event['target'] == 'number':
                event['correct'] = event['letter_resp'] != event['motor1']
        else:
            event['missed_m1'] = True
            event['correct'] = False

        # PAS
        event['missed_m2'] = False
        if trigger_motor2 == (2 ** 6):
            # 64
            event['pas'] = 0  # Let's take the python convention starting at 0
        if trigger_motor2 == (2 ** 7):
            # 128
            event['pas'] = 1
        if trigger_motor2 == (2 ** 8):
            # 256
            event['pas'] = 2
        if trigger_motor2 == (2 ** 9):
            # 512
            event['pas'] = 3
        else:
            event['missed_m2'] = True
            event['pas'] = None  # XXX JRK:


        # Block
        if ((trigger_stim % 2) == 1):
            #odd numbers
            event['block'] = 'invisible'
        if ((trigger_stim % 2) == 0):
            #even numbers
            event['block'] = 'visible'


        # Local context
        '''
        if  :
            event['local_context'] = 'UU'
        if  :
            event['local_context'] = 'SU'
        if  :
            event['local_context'] = 'US'
        if  :
            event['local_context'] = 'SS'
        '''

        # Stimulus side (top or bottom)
        ## uh oh, this isn't coded directly in the ttls, so we would need the
        ## behavioral structure if we want to make this contrast
        # XXX JRK : "Damned" oh well, not the end of the world.

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
