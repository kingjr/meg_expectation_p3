from copy import deepcopy

def reverse_pas(events_orig):
    """This function reverses PAS rating triggers for the one subject who used
    the buttons in the wrong order and returns the corrected event
    structure."""
    events_rev = deepcopy(events_orig)

    for e in range(len(events_orig)):
        if events_orig[e][2] == 64:
            events_rev[e][2] = 512
        elif events_orig[e][2] == 128:
            events_rev[e][2] = 256
        elif events_orig[e][2] == 256:
            events_rev[e][2] = 128
        elif events_orig[e][2] == 512:
            events_rev[e][2] = 64
    return events_rev
