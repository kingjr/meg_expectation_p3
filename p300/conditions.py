
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
    elif condition == 'motor2':  # remove response not linked to stim and 2nd motor response
        selection = np.where((trigger >= 2 ** 15) & (trigger < 2 ** 16))[0]
    elif condition == 'motor1':  # remove response not linked to stim
        selection = np.where(trigger >= 2 ** 16)[0]
    return selection

def get_events(events):
    """ Define trigger condition correspondance here"""
    import pandas as pd

    n_events = events.shape[0]

    # initialize future data frame
    events_data_frame = list()

    # Loop across all trials
    for ii in range(n_events):
        event = dict()

        # Event type: stimulus, 1st (2AFC) or 2nd (PAS) response
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

        # Target present?
        if trigger_stim in [i * 12 + j for i in range(5) for j in range(1,5)]:
            # 1-4 13-16 25-28 37-40 49-52
            event['present'] = False
            event['target'] = None
            event['soa'] = None
            if trigger_stim in range(1, 5):
                event['soa_ttl'] = 17
            elif trigger_stim in range(13, 17):
                event['soa_ttl'] = 33
            elif trigger_stim in range(25, 29):
                event['soa_ttl'] = 50
            elif trigger_stim in range(37, 41):
                event['soa_ttl'] = 67
            elif trigger_stim in range(49, 53):
                event['soa_ttl'] = 83
            else:
                event['soa_ttl'] = None
        else:
            event['present'] = True

            # Target type
            if trigger_stim in [4 + i * 12 + j for i in range(5) for j in range(1,5)]:
                # 5 7 17-20 29-32 41-44 54 56
                event['target'] = 'letter'
            elif trigger_stim in [8 + i * 12 + j for i in range(5) for j in range(1,5)]:
                # 9 11 21-24 33-36 45-48 58 60
                event['target'] = 'number'
            else:
                event['target'] = None

            # SOA for target-present trials
            if trigger_stim in range (5,13):
                # 5, 7, 9, 11 (can be 5-12)
                event['soa'] = 17
            elif trigger_stim in range(17,25):
                # 17-24
                event['soa'] = 33
            elif trigger_stim in range(29,37):
                # 29-36
                event['soa'] = 50
            elif trigger_stim in range(41,49):
                # 41-48
                event['soa'] = 67
            elif trigger_stim in range(53,61):
                # 54,56,58,60 (can be 53-60)
                event['soa'] = 83
            else:
                event['soa'] = None

            event['soa_ttl'] = event['soa']

        # Forced_choice response (which right hand finger corresponds to letter
        # response)
        if (np.binary_repr(trigger_stim,width=2)[-2]!=np.binary_repr(trigger_stim,width=2)[-1]):
            event['letter_resp'] = 'left'
        elif (np.binary_repr(trigger_stim,width=2)[-2]==np.binary_repr(trigger_stim,width=2)[-1]):
            event['letter_resp'] = 'right'
        else:
            event['letter_resp'] = None

        # motor response
        if trigger_motor1 == (2 ** 13):
            event['motor1'] = 'right'
        elif trigger_motor1 == (2 ** 14):
            event['motor1'] = 'left'
        else:
            event['motor1'] = None

        # Accuracy
        if event['motor1'] is not None:
            event['missed_m1'] = False
            if event['target'] == 'letter':
                event['correct'] = event['letter_resp'] == event['motor1']
            elif event['target'] == 'number':
                event['correct'] = event['letter_resp'] != event['motor1']
        else:
            event['missed_m1'] = True
            event['correct'] = None # we want to differentiate between incorrect and no answer

        # PAS
        event['missed_m2'] = False
        if trigger_motor2 == (2 ** 6):
            # 64
            event['pas'] = 0
        elif trigger_motor2 == (2 ** 7):
            # 128
            event['pas'] = 1
        elif trigger_motor2 == (2 ** 8):
            # 256
            event['pas'] = 2
        elif trigger_motor2 == (2 ** 9):
            # 512
            event['pas'] = 3
        else:
            event['missed_m2'] = True
            event['pas'] = None

        # XXX Define this only when you'll analyze it 
        # # Seen/Unseen (0,1 vs. 2,3)
        # if event['pas'] < 2:
        #     event['seen'] = 0
        # elif event['pas'] > 1:
        #     event['seen'] = 1
        # else:
        #     event['seen'] = None

        # Seen/Unseen (0 vs. 1,2,3)
        if event['pas'] < 1:
            event['seen'] = False
        elif event['pas'] > 0:
            event['seen'] = True
        else:
            event['seen'] = None

        # Block
        if ((trigger_stim % 2) == 1):
            #odd numbers
            event['block'] = 'invisible'
        elif ((trigger_stim % 2) == 0):
            #even numbers
            event['block'] = 'visible'
        else:
            event['block'] = None


        # Local context
        N1_trigger = events[ii-1, 2] % (2 ** 15)
        N1_trigger_stim = N1_trigger % 64
        N1_trigger_motor2 = N1_trigger % 1024 - trigger_stim # PAS response for N-1

        N2_trigger = events[ii-2, 2] % (2 ** 15)
        N2_trigger_stim = N2_trigger % 64
        N2_trigger_motor2 = N2_trigger % 1024 - trigger_stim # PAS response for N-2

        if (N1_trigger_motor2 <= (2 ** 7)):
            if (N2_trigger_motor2 <= (2 ** 7)):
                event['local_context'] = 'UU'
            elif (N2_trigger_motor2 > (2 ** 7)):
                event['local_context'] = 'US'
        elif (N1_trigger_motor2 > (2 ** 7)):
            if (N2_trigger_motor2 <= (2 ** 7)):
                event['local_context'] = 'SU'
            elif (N2_trigger_motor2 > (2 ** 7)):
                event['local_context'] = 'SS'
        else:
            event['local_context'] = None


        events_data_frame.append(event)
    return pd.DataFrame(events_data_frame)

def extract_events(raw):
    """This is a hand-made extraction of the events so as to look for all stim
    triggers, find the two following motor responses, and combined their trigger
    values to know that they go in triplets."""

    events = mne.find_events(raw, stim_channel='STI101', verbose=True,
                             consecutive='increasing', min_duration=0.000,
                             shortest_event=1) # XXX ISSUE WITH MISSING TRIGGER
                                               # IN ABSENT TRIAL

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
