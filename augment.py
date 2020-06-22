import PIL
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

def add_augment(img, policy):

    shape = img.shape

    if shape[2] == 3:
        fcol = tuple((np.mean(img, axis=(0,1))).astype('int'))
        fmod = 'RGB'
        fsha = (shape[1], shape[0], 3)

    for p in policy:
        
        if p['op'] == 'non':
            pass
        
        # flip(mirror)
        elif p['op'] == 'mrx':
            img = np.fliplr(img)
        elif p['op'] == 'mry':
            img = np.flipud(img)
        elif p['op'] == '180':
            img = np.fliplr(np.flipud(img))
        
        #crop
        elif p['op'] == 'crp':
            mag = p['mag'] / 10.0 * np.random.uniform(0.0, 1.0)
            zoom = np.random.uniform(0.25 + 0.75*(1-mag) -1e-6, 1+1e-6)
            h1 = np.random.uniform(0, np.random.uniform(1e-6, 1-zoom+1e-4))
            w1 = np.random.uniform(0, np.random.uniform(1e-6, 1-zoom+1e-4))
            h2 = h1 + zoom
            w2 = w1 + zoom
            img = np.expand_dims(img, axis=0)
            img = tf.image.crop_and_resize(img, [[h1, w1, h2, w2]], [0], (shape[1], shape[0]))
            img = img[0]

        #cutout
        elif p['op'] == 'ct1':
            rand = np.random.uniform(0, 1, 2)
            blob = (np.array([rand[1]*shape[1], rand[1]*(0.5+rand[0])*shape[0]]) * p['mag']/20.).astype('int')
            if blob[1]*blob[0] != 0:
                pxls = (np.random.randint(shape[0]-blob[1]-1), np.random.randint(shape[1]-blob[0]-1))
                if shape[2] == 3:
                    blob = np.append(blob, [3])
                noise = PIL.Image.fromarray(np.clip(np.random.randint(255, size=blob),0,255).astype('uint8'))
                img = PIL.Image.fromarray(np.array(img, dtype=np.uint8))
                img.paste(noise, pxls)
                img = np.array(img)

        elif p['op'] == 'ct2':
            rand = np.random.uniform(0, 1, 2)
            blob = (np.array([rand[1]*shape[1], rand[1]*(0.5+rand[0])*shape[0]]) * p['mag']/20.).astype('int')
            if blob[1]*blob[0] != 0:
                pxls = (np.random.randint(shape[0]-blob[1]-1), np.random.randint(shape[1]-blob[0]-1))
                img = np.expand_dims(img, axis=0)
                if blob[0] % 2 != 0:
                    blob[0] += 1
                if blob[1] % 2 != 0:
                    blob[1] += 1
                img = tfa.image.cutout(img, mask_size=(blob[0], blob[1]), offset=(pxls[1], pxls[0]), constant_values=128)
                img = img[0]
    
    return img


def get_policies(name):
    
    policies = []

    if name == 'reduced_mirror':
        policies.append({'op': 'mrx', 'prob': 0.5, 'mag': 0., 'const': True})
        policies.append({'op': 'non', 'prob': 0.5, 'mag': 0., 'const': True})

    elif name == 'cutout':
        policies.append({'op': 'ct1', 'prob': 0.5, 'mag': 7.0, 'const': False})
        policies.append({'op': 'ct2', 'prob': 0.5, 'mag': 7.0, 'const': False})

    elif name == 'crop':
        policies.append({'op': 'crp', 'prob': 0.7, 'mag': 9., 'const': False})
        policies.append({'op': 'crp', 'prob': 0.3, 'mag': 5., 'const': False})

    return policies