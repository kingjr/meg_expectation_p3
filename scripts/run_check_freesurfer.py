import os.path as op

from scripts.config import (
    data_path,
    subjects,
    results_dir,
    open_browser
)

for subject in subjects:
    fname = op.join(data_path, 'subjects', subject, 'scripts', 'recon-all.log')
    if op.isfile(fname):
        with open(fname, 'rb') as fh:
            fh.seek(-1024, 2)
            last = fh.readlines()[-1].decode()
        print('{}: "{}"'.format(subject, last))
    else:
        print('{}: missing'.format(subject))
