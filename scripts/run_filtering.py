# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import sys
import mkl
import os
import os.path as op
import mne
from config import(
    data_path,
    subjects,
    runs,
    raw_fname_tmp,
    raw_fname_filt_tmp,
    highpass,
    lowpass,
    filtersize
)

debug = False

if len(sys.argv) > 1:
    subjects = [sys.argv[1]]
    mkl.set_num_threads(1)

for subject in subjects:
    this_path = op.join(data_path, 'MEG', subject)
    os.chdir(this_path)
    for r in runs:
        raw_in = raw_fname_tmp.format(r)
        raw_out = raw_fname_filt_tmp.format(r)
        if op.exists(op.join(this_path, raw_out)) and debug:
            continue
        else:
            print op.join(this_path, raw_in)
        raw = mne.io.Raw(raw_in, preload=True)
        raw.filter(highpass, lowpass, filter_length=filtersize)
        raw.save(raw_out, overwrite=True)
