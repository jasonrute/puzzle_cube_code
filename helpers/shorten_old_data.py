"""
Convert all data files (for a certain version) to another file labeled <original_version>.short
where we remove the 48 rotations.  (It is better to do the rotations during the training_phase.)
"""

import sys

import os
import numpy as np
import h5py

VERSIONS = ["v0.8.1"]

def str_between(s, start, end):
    return (s.split(start))[1].split(end)[0]

for version in VERSIONS:

    data_files = [(str_between(f, "_gen", ".h5"), f)
                        for f in os.listdir('../save/') 
                        if f.startswith("data_{}_gen".format(version))
                        and f.endswith(".h5")]
    
    for gen, f in data_files:
        path = "../save/" + f

        print("loading data:", "'" + path + "'")

        with h5py.File(path, 'r') as hf:
            old_inputs = hf['inputs'][:]
            old_policies = hf['outputs_policy'][:]
            old_values = hf['outputs_value'][:]

        # check that it is the augmented version
        l = len(old_inputs)
        assert l % 48 == 0
        assert l > 20000
        
        new_l = l // 48
        new_path = path.replace(version, version+".short")

        # check that file has not already been created
        assert path.replace('../save/', '') in os.listdir('../save/')
        assert new_path.replace('../save/', '') not in os.listdir('../save/')

        new_inputs = old_inputs[:new_l]
        new_policies = old_policies[:new_l]
        new_values = old_values[:new_l]

        with h5py.File(new_path, 'w') as hf:
            hf.create_dataset("inputs",  data=new_inputs)
            hf.create_dataset("outputs_policy",  data=new_policies)
            hf.create_dataset("outputs_value",  data=new_values)

        print("saved data:", "'" + new_path + "'")