from __future__ import print_function
from multiprocessing import sharedctypes
import ctypes
import numpy as np
import operator
import matplotlib.pyplot as plt
import json

def make_objective_indices_and_coeffs(temporal_coeffs, meas_coeffs):
    objective_coeffs = (np.reshape(temporal_coeffs, (1,-1)) * np.reshape(meas_coeffs, (-1,1))).flatten()
    objective_indices = np.where(np.abs(objective_coeffs) > 1e-8)[0]
    return objective_indices, objective_coeffs[objective_indices]

def make_array(shape=(1,), dtype=np.float32, shared=False, fill_val=None):  
    np_type_to_ctype = {np.float32: ctypes.c_float,
                        np.float64: ctypes.c_double,
                        np.bool: ctypes.c_bool,
                        np.uint8: ctypes.c_ubyte,
                        np.uint64: ctypes.c_ulonglong}
    
    if not shared:
        np_arr = np.empty(shape, dtype=dtype)
    else:
        numel = np.prod(shape)
        arr_ctypes = sharedctypes.RawArray(np_type_to_ctype[dtype], numel)
        np_arr = np.frombuffer(arr_ctypes, dtype=dtype, count=numel)
        np_arr.shape = shape
    
    if not fill_val is None:
        np_arr[...] = fill_val
    
    return np_arr

def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z

def binary_subset(n, indices):
    return int(''.join(operator.itemgetter(*indices)(bin(n)[2:])), 2)

def binary_superset(n, indices):
    superset = np.zeros(max(indices)+1, dtype=int)
    print(list(bin(n)[2:]))
    superset[indices] = list(bin(n)[2:])
    print(superset)
    return int(np.array_str(superset)[1:-1:2], 2)

class StackedBarPlot:
    def __init__(self, data, nfig=17, labels=[], ylim=[]):
        self.data = data
        self.fig = plt.figure(nfig)
        self.ax = plt.gca()
        self.colors = ['r','g','b','y','c','m','k', [0.5,0,0], [0,0.5,0], [0,0,0.5], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5], [0.5,0.5,0.5]]
        self.xs = np.arange(data.shape[1])
        self.width = 0.9
        
        if len(labels):
            assert(len(labels) == data.shape[1])
            self.labels = labels
        else:
            self.labels = range(data.shape[1])
            
        if len(ylim):
            self.ax.set_ylim(ylim)

        self.plots = []
        curr_bot = np.zeros(data.shape[1])
        for n in range(data.shape[0]):
            self.plots.append(plt.bar(self.xs, data[n], bottom=curr_bot, width=self.width, color=self.colors[n]))
            curr_bot += data[n]

    

        plt.ylabel('Objective value')
        plt.title('Action selection')
        plt.xticks(self.xs + self.width/2., self.labels)
        #plt.yticks(np.arange(0, 81, 10))
        #plt.legend((p1[0], p2[0], p3[0]), ('Men', 'Women', 'Undecided'))

    def show(self):
        self.fig.show()
        
    def draw(self):
        self.fig.canvas.draw()
        
    def set_data(self, data, labels=[]):
        assert(data.shape == self.data.shape)
        curr_bot = np.zeros(data.shape[1])
        for n in range(data.shape[0]):
            for nr in range(data.shape[1]):
                self.plots[n][nr].set_height(data[n,nr])
                self.plots[n][nr].set_y(curr_bot[nr])
            curr_bot += data[n]
        if len(labels):
            assert(len(labels) == data.shape[1])
            self.labels = labels
            self.ax.set_xticklabels(self.labels)
            
# modification which returns a string object instead of Unicode
# http://stackoverflow.com/a/33571117
def json_load_byteified(file_handle):
    return _byteify(
        json.load(file_handle, object_hook=_byteify),
        ignore_dicts=True
    )

def json_loads_byteified(json_text):
    return _byteify(
        json.loads(json_text, object_hook=_byteify),
        ignore_dicts=True
    )

def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data
