import random
import pandas as pd
import numpy as np

def next_batch(x, y, batch_size=50):
    '''
    returns a batch from x and y respectivally of size batch_size
    '''
    start = int(random.random()*x.shape[0])
    if start<x.shape[0]-1-batch_size:
        x_batch = x[start:start+batch_size]
        y_batch = y[start:start+batch_size]
    else:
        rollover = batch_size-(x.shape[0]-start-1)
        x_batch = pd.concat([x[start:], x[:rollover]])
        y_batch = np.concatenate((y[start:], y[:rollover]))
        
    return x_batch, y_batch