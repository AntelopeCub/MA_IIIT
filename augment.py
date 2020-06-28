import numpy as np
import PIL
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import ImageOps
from tensorflow.keras.preprocessing.image import random_rotation, random_shift


def add_augment(img, policy):

    shape = img.shape

    if shape[2] == 3:
        fcol = np.mean(img, axis=(0,1)).astype('uint8')
        fcol_int = np.mean(img).astype('uint8')
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
            img = np.asarray(img[0], dtype=np.uint8)

        #cutout
        elif p['op'] == 'ct1':
            rand = np.random.uniform(0, 1, 2)
            blob = (np.array([rand[1]*shape[1], rand[1]*(0.5+rand[0])*shape[0]]) * p['mag']/20.).astype('int')
            if blob[1]*blob[0] != 0:
                pxls = (np.random.randint(shape[0]-blob[1]-1), np.random.randint(shape[1]-blob[0]-1))
                if shape[2] == 3:
                    blob = np.append(blob, [3])
                noise = PIL.Image.fromarray(np.clip(np.random.randint(255, size=blob),0,255).astype('uint8'))
                img = PIL.Image.fromarray(np.asarray(img, dtype=np.uint8))
                img.paste(noise, pxls)
                img = np.asarray(img)

        elif p['op'] == 'ct2':
            rand = np.random.uniform(0, 1, 2)
            blob = (np.array([rand[1]*shape[1], rand[1]*(0.5+rand[0])*shape[0]]) * p['mag']/20.).astype('int')
            if blob[1]*blob[0] != 0:
                img = np.expand_dims(img, axis=0)
                if blob[0] % 2 != 0:
                    blob[0] += 1
                if blob[1] % 2 != 0:
                    blob[1] += 1
                pxls = (np.random.randint(blob[1] // 2, shape[0]-blob[1] // 2), np.random.randint(blob[0] // 2, shape[1]-blob[0] // 2))
                img = tfa.image.cutout(img, mask_size=(blob[0], blob[1]), offset=(pxls[1], pxls[0]), constant_values=fcol_int)
                img = img[0]

        #invert
        elif p['op'] == 'inv':
            img = np.invert(img)

        #rotation
        elif p['op'] == 'rot':
            angle = 45 * p['mag'] / 10.
            img = random_rotation(img, angle, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=fcol_int, interpolation_order=3)

        #sharpness
        elif p['op'] == 'sha':
            if not isinstance(img, tf.Tensor):
                img = tf.convert_to_tensor(img)
            enha  = 1 + np.random.uniform(-1, 1) * p['mag'] / 10.
            img = tfa.image.sharpness(img, enha)

        #shear
        elif p['op'] == 'srx':
            #img = np.asarray(img, dtype=np.uint8)
            shear = np.random.uniform(-0.5, 0.5) * p['mag'] / 10.
            img = tfa.image.shear_x(img, shear, fcol)
        elif p['op'] == 'sry':
            #img = np.asarray(img, dtype=np.uint8)
            shear = np.random.uniform(-0.5, 0.5) * p['mag'] / 10.
            img = tfa.image.shear_y(img, shear, fcol)

        #autocontrast
        elif p['op'] == 'auc':
            img = PIL.Image.fromarray(np.asarray(img, dtype=np.uint8))
            img = ImageOps.autocontrast(img, cutoff=2)
            img = np.asarray(img)
        
        #contrast
        elif p['op'] == 'con':
            enha = 1 + np.random.uniform(-1, 1) * p['mag'] / 10.
            img = tf.image.adjust_contrast(img, enha)

        #saturation
        elif p['op'] == 'clr':
            enha = 1 + np.random.uniform(-1, 1) * p['mag'] / 10.
            img = tf.image.adjust_saturation(img, enha)

        #brightness
        elif p['op'] == 'bri':
            enha = 1 + np.random.uniform(-1, 1) * p['mag'] / 10.
            img = tf.image.adjust_brightness(img, enha)
        
        #equaliz
        elif p['op'] == 'eqz':
            if not isinstance(img, tf.Tensor):
                img = tf.convert_to_tensor(img)
            img = tfa.image.equalize(img)

        #translation
        elif p['op'] == 'tlx':
            shift = np.random.uniform(-0.5, 0.5) * p['mag'] / 10.
            img = random_shift(img, shift, 0, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=fcol_int, interpolation_order=3)
        elif p['op'] == 'tly':
            shift = np.random.uniform(-0.5, 0.5) * p['mag'] / 10.
            img = random_shift(img, 0, shift, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=fcol_int, interpolation_order=3)

        #solarize
        elif p['op'] == 'sol':
            th = int(256 * (1 - np.random.uniform(0, 1) * p['mag'] / 10.))
            img = PIL.Image.fromarray(np.asarray(img, dtype=np.uint8))
            if np.random.rand() - 0.5 > 0:
                img = ImageOps.solarize(img, threshold=th)
            else:
                img = ImageOps.invert(img)
                img = ImageOps.solarize(img, threshold=th)
                img = ImageOps.invert(img)
            img = np.asarray(img)

        #posterize
        elif p['op'] == 'pos':
            bit = int(8.5 - np.random.uniform(0, 0.5) * p['mag'])
            img = PIL.Image.fromarray(np.asarray(img, dtype=np.uint8))
            img = ImageOps.posterize(img, bit)
            img = np.asarray(img)


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

    elif name == 'cifar_pol1':
        policies.append({'op': 'inv', 'prob': 0.004, 'mag': 0.0, 'const': False}) 
        policies.append({'op': 'rot', 'prob': 0.028, 'mag': 2.0, 'const': False})  
        policies.append({'op': 'sha', 'prob': 0.044, 'mag': 9.0, 'const': False}) 
        policies.append({'op': 'sry', 'prob': 0.028, 'mag': 4.8, 'const': False})  
        policies.append({'op': 'auc', 'prob': 0.088, 'mag': 0.0, 'const': False}) 
        policies.append({'op': 'clr', 'prob': 0.084, 'mag': 9.0, 'const': False})  
        policies.append({'op': 'eqz', 'prob': 0.084, 'mag': 0.0, 'const': False}) 
        policies.append({'op': 'con', 'prob': 0.024, 'mag': 7.0, 'const': False})
        policies.append({'op': 'tly', 'prob': 0.080, 'mag': 8.2, 'const': False}) 
        policies.append({'op': 'bri', 'prob': 0.040, 'mag': 6.0, 'const': False})  
        policies.append({'op': 'sol', 'prob': 0.036, 'mag': 5.0, 'const': False})  
        policies.append({'op': 'non', 'prob': 0.460, 'mag': 0.0, 'const': False})
    elif name == 'cifar_pol2':
        policies.append({'op': 'con', 'prob': 0.008, 'mag': 6.0, 'const': False}) 
        policies.append({'op': 'tlx', 'prob': 0.032, 'mag': 8.2, 'const': False})  
        policies.append({'op': 'sha', 'prob': 0.068, 'mag': 6.0, 'const': False}) 
        policies.append({'op': 'tly', 'prob': 0.056, 'mag': 8.2, 'const': False})  
        policies.append({'op': 'eqz', 'prob': 0.104, 'mag': 0.0, 'const': False}) 
        policies.append({'op': 'pos', 'prob': 0.012, 'mag': 5.6, 'const': False})  
        policies.append({'op': 'bri', 'prob': 0.052, 'mag': 7.0, 'const': False}) 
        policies.append({'op': 'auc', 'prob': 0.112, 'mag': 0.0, 'const': False})
        policies.append({'op': 'clr', 'prob': 0.036, 'mag': 8.0, 'const': False}) 
        policies.append({'op': 'inv', 'prob': 0.004, 'mag': 0.0, 'const': False})  
        policies.append({'op': 'sol', 'prob': 0.040, 'mag': 8.0, 'const': False})  
        policies.append({'op': 'non', 'prob': 0.476, 'mag': 0.0, 'const': False})

    return policies
