#!/usr/bin/env python3
'''
Script to clean the empty folders made when testing or doing incomplete run.
By default only deletes empty folders, can be set to also delete incomplete runs.

clean_output_dir.py
sentifmdetect17 
12/20/17
Copyright (c) Gilles Jacobs. All rights reserved.  
'''
import os
from sentifmdetect import settings
from sentifmdetect import util

opt_dir = os.path.abspath(os.path.join(settings.OPT_DIRP, os.pardir))
print(opt_dir)
pattern = [os.path.join(opt_dir, "{}_*".format(dataset)) for dataset in settings.ALL_DATA_DIR.keys()]
for dirp in util.multipattern_glob(*pattern):
    if not os.listdir(dirp): # check empty
        print("Delete empty dir {}".format(dirp))
        os.rmdir(dirp)

