

Here's my take if you want to try using multiprocesses to process each row of numpy array,

from multiprocessing import Pool
import numpy as np

def my_function(x):
    pass     # do something and return something

if __name__ == '__main__':
    X = np.arange(6).reshape((3,2))
    pool = Pool(processes = 4)
    results = pool.map(my_function, map(lambda x: x, X))
    pool.close()
    pool.join()

pool.map take in a function and an iterable.
I used 'map' function to create an iterator over each rows of the array.
Maybe there's a better to create the iterable though.
