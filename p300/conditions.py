
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
        selection = np.where((trigger > 0) & (trigger < 8192))[0]
    elif condition == 'motor':  # remove response not linked to stim
        selection = np.where(trigger >= 8192)[0]
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

        # Event type: stimulus or response trigger
        if events[ii, 2] < (2 ** 13):
            event['type'] = 'stim'
        else:
            event['type'] = 'motor'

        # Motor side
        if event['type'] == 'motor':
            if events[ii, 2] == (2 ** 13):
                event['motor_side'] = 'left'
            if events[ii, 2] == (2 ** 14):
                event['motor_side'] = 'right'
        else:
            event['motor_side'] = 'none'

        # Stimulus side (top or bottom)

        # Stimulus category (letter or digit)

        # Stimulus Onset Asynchrony with mask

        # Reaction Time

        # ...



        events_data_frame.append(event)

    return pd.DataFrame(events_data_frame)
