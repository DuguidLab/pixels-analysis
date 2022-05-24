from pixels.experiment import Experiment
from pixels.behaviours.pushpull import PushPull, ActionLabels, Events
import pandas as pd
import numpy as np
#/home/s1612001/duguidlab/
pushpull=[Experiment(
    'C57_1335401',  # This can be a list
    PushPull, #Change to Pushpull?
    '/home/s1612001/duguidlab/duguidlab/Direction_Sensitivity/Data',
    '/home/s1612001/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
),
]
def gen_data_hitmiss(myexp):
    	#hits
	hits = myexp.align_trials(
	    ActionLabels.rewarded_push,  # This selects which trials we want
	    Events.tone_onset,  # This selects what event we want them aligned to 
 	    data='spike',
        raw=True,
        duration=2  # And this selects what kind of data we want
	)
	hits=hits[0][0]#change for thalamus
	hits.columns=hits.columns.swaplevel(0,1)
	hits.sort_index(axis=1, level=0, inplace=True)
	dim1=len(hits.groupby(level=0, axis=1))
	dim2=len(hits.groupby(level=1, axis=1))
	hits=hits.stack()
	hitsnpy=hits.values.T.reshape(dim1, -1, dim2)
	hits_L=np.ones(dim1)
	#Misses
	misses = myexp.align_trials(
	    ActionLabels.missed_tone,  # This selects which trials we want
	    Events.tone_onset,  # This selects what event we want them aligned to 
 	   'spike_rate',
		raw=True,
        duration=2  # And this selects what kind of data we want
	)
	misses=misses[0][0]#change for thalamus
	misses.columns=misses.columns.swaplevel(0,1)
	misses.sort_index(axis=1, level=0, inplace=True)
	dim1s=len(misses.groupby(level=0, axis=1))
	dim2s=len(misses.groupby(level=1, axis=1))
	misses=misses.stack()
	missesnpy=misses.values.T.reshape(dim1s, -1, dim2s)
	misses_L=np.zeros(dim1s)
	#Joining
	data_3D=np.concatenate((hitsnpy, missesnpy), axis=0)
	labels=np.concatenate((hits_L, misses_L), axis=0)
	data_2D=data_3D.reshape(dim1+dim1s, -1)
	idx=myexp.mouse_ids
	np.save("/home/s1612001/duguidlab/thalamus_paper/toni_NNdata/Numpy_arrays/M1_3D-{}_hitmiss".format(idx), data_3D)
	np.save("/home/s1612001/duguidlab/thalamus_paper/toni_NNdata/Numpy_arrays/M1_2D-{}_hitmiss".format(idx), data_2D)
	np.save("/home/s1612001/duguidlab/thalamus_paper/toni_NNdata/Numpy_arrays/M1_L-{}_hitmiss".format(idx), labels)
	return
def gen_data_pushpull(myexp):
    	#hits
	push = myexp.align_trials(
	    ActionLabels.rewarded_push,  # This selects which trials we want
	    Events.tone_onset,  # This selects what event we want them aligned to 
 	    data='spike',
        raw=True,
        duration=2  # And this selects what kind of data we want
	)
	push=push[0][0]#change for thalamus
	push.columns=push.columns.swaplevel(0,1)
	push.sort_index(axis=1, level=0, inplace=True)
	dim1=len(push.groupby(level=0, axis=1))
	dim2=len(push.groupby(level=1, axis=1))
	push=push.stack()
	pushnpy=push.values.T.reshape(dim1, -1, dim2)
	push_L=np.ones(dim1)
	#pull
	pull = myexp.align_trials(
	    ActionLabels.rewarded_pull,  # This selects which trials we want
	    Events.tone_onset,  # This selects what event we want them aligned to 
 	   'spike_rate',
		raw=True,
        duration=2  # And this selects what kind of data we want
	)
	pull=pull[0][0]#change for thalamus
	pull.columns=pull.columns.swaplevel(0,1)
	pull.sort_index(axis=1, level=0, inplace=True)
	dim1s=len(pull.groupby(level=0, axis=1))
	dim2s=len(pull.groupby(level=1, axis=1))
	pull=pull.stack()
	pullnpy=pull.values.T.reshape(dim1s, -1, dim2s)
	pull_L=np.zeros(dim1s)
	#Joining
	data_3D=np.concatenate((pushnpy, pullnpy), axis=0)
	labels=np.concatenate((push_L, pull_L), axis=0)
	#data_2D=data_3D.reshape(dim1+dim1s, -1)
	idx=myexp.mouse_ids
	np.save("/home/s1612001/duguidlab/thalamus_paper/toni_NNdata/Numpy_arrays/M1_3D-{}_pushpull".format(idx), data_3D)
	#np.save("/home/s1612001/duguidlab/thalamus_paper/toni_NNdata/Numpy_arrays/M1_2D-{}_pushpull".format(idx), data_2D)
	np.save("/home/s1612001/duguidlab/thalamus_paper/toni_NNdata/Numpy_arrays/M1_L-{}_pushpull".format(idx), labels)
	return
for exp in pushpull:
	gen_data_pushpull(exp)