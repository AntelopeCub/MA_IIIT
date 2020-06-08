import tensorflow as tf

def write_list(f, name, direction):
    grp = f.create_group(name)
    for i, l in enumerate(direction):
        if isinstance(l, tf.Tensor):
            l_np = l.numpy()
        grp.create_dataset(str(i), data=l_np)

def read_list(f, name):
    grp = f[name]
    return [grp[str(i)] for i in range(len(grp))] 
