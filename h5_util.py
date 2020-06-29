import tensorflow as tf
import numpy as np

def write_list(f, name, direction):
    grp = f.create_group(name)
    for i, l in enumerate(direction):
        if isinstance(l, tf.Tensor):
            l_np = l.numpy()
        elif isinstance(l, np.ndarray):
            l_np = np.copy(l)
        grp.create_dataset(str(i), data=l_np)

def read_list(f, name):
    grp = f[name]
    return [grp[str(i)] for i in range(len(grp))] 
