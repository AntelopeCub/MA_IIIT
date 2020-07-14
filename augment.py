import numpy as np
import PIL
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

#from tensorflow.keras.preprocessing.image import random_rotation, random_shift


def add_augment(img, policy):

    shape = img.shape #shape[0]:height, shape[1]: width, shape[2]: channel

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
            img = Image.fromarray(img)
            mag = p['mag'] / 10.0 * np.random.uniform(0.0, 1.0)
            zoom = np.random.uniform(0.25 + 0.75*(1-mag)-1e-6, 1+1e-6)
            left = np.random.uniform(0, shape[1]*np.random.uniform(1e-6, 1-zoom+1e-4)) #left
            upper = np.random.uniform(0, shape[0]*np.random.uniform(1e-6, 1-zoom+1e-4)) #upper
            right = left + zoom * shape[1] #right
            lower = upper + zoom * shape[0] #lower
            img = img.crop((left, upper, right, lower))
            img = img.resize((shape[1], shape[0]), resample=Image.BICUBIC)

        #cutout
        elif p['op'] == 'ct1':
            rand = np.random.uniform(0, 1, 2)
            blob = (np.array([rand[1]*shape[1], rand[1]*(0.5+rand[0])*shape[0]]) * p['mag']/20.).astype('int')
            if blob[1]*blob[0] != 0:
                pxls = (np.random.randint(shape[1]-blob[0]-1), np.random.randint(shape[0]-blob[1]-1))
                if shape[2] == 3:
                    blob = np.append(blob, [3])
                noise = Image.fromarray(np.clip(np.random.randint(255, size=blob),0,255).astype('uint8'))
                img = Image.fromarray(img)
                img.paste(noise, pxls)
                #img = np.asarray(img)

        elif p['op'] == 'ct2':
            img = Image.fromarray(img)
            rand = np.random.uniform(0, 1, 2)
            blob = (np.array([rand[1]*shape[1], rand[1]*(0.5+rand[0])*shape[0]]) * p['mag']/20.).astype('int')
            if blob[1]*blob[0] != 0:
                pxls = (np.random.randint(shape[1]-blob[0]-1), np.random.randint(shape[0]-blob[1]-1))
                if img.mode == 'RGB':
                    blob = np.append(blob, [3])
                draw = ImageDraw.Draw(img)
                draw.rectangle([pxls[0], pxls[1], pxls[0]+blob[1], pxls[1]+blob[0]], fill=tuple(fcol))

        #invert
        elif p['op'] == 'inv':
            img = np.invert(img)

        #rotation
        elif p['op'] == 'rot':
            img = Image.fromarray(img)
            angle = np.random.uniform(-45, 45) * p['mag'] / 10.
            img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=tuple(fcol))

        #sharpness
        elif p['op'] == 'sha':
            img = Image.fromarray(img)
            enha  = 1 + np.random.uniform(-1, 1) * p['mag'] / 10.
            img = ImageEnhance.Sharpness(img).enhance(enha)
        
        #shear
        elif p['op'] == 'srx':
            img = Image.fromarray(img)
            shear = np.random.uniform(-0.5, 0.5) * p['mag'] / 10.
            img = img.transform((shape[1], shape[0]), Image.AFFINE, (1, shear, 0, 0, 1, 0), resample=Image.BICUBIC, fillcolor=tuple(fcol))
        elif p['op'] == 'sry':
            img = Image.fromarray(img)
            shear = np.random.uniform(-0.5, 0.5) * p['mag'] / 10.
            img = img.transform((shape[1], shape[0]), Image.AFFINE, (1, 0, 0, shear, 1, 0), resample=Image.BICUBIC, fillcolor=tuple(fcol))
        
        #autocontrast
        elif p['op'] == 'auc':
            img = Image.fromarray(img)
            img = ImageOps.autocontrast(img, cutoff=2)

        #contrast
        elif p['op'] == 'con':
            img = Image.fromarray(img)
            enha = 1 + np.random.uniform(-1, 1) * p['mag'] / 10.
            img = ImageEnhance.Contrast(img).enhance(enha)
        
        #saturation
        elif p['op'] == 'clr':
            img = Image.fromarray(img)
            enha = 1 + np.random.uniform(-1, 1) * p['mag'] / 10.
            img = ImageEnhance.Color(img).enhance(enha)

        #brightness
        elif p['op'] == 'bri':
            img = Image.fromarray(img)
            enha = 1 + np.random.uniform(-1, 1) * p['mag'] / 10.
            img = ImageEnhance.Brightness(img).enhance(enha)
        
        #equalize
        elif p['op'] == 'eqz':
            img = Image.fromarray(img)
            img = ImageOps.equalize(img)

        #translation
        elif p['op'] == 'tlx':
            img = Image.fromarray(img)
            shift = np.random.uniform(-0.5, 0.5) * p['mag'] / 10.
            img = img.transform((shape[1], shape[0]), Image.AFFINE, (1, 0, shift, 0, 1, 0), resample=Image.BICUBIC, fillcolor=tuple(fcol))
        elif p['op'] == 'tly':
            img = Image.fromarray(img)
            shift = np.random.uniform(-0.5, 0.5) * p['mag'] / 10.
            img = img.transform((shape[1], shape[0]), Image.AFFINE, (1, 0, 0, 0, 1, shift), resample=Image.BICUBIC, fillcolor=tuple(fcol))

        #solarize
        elif p['op'] == 'sol':
            th = int(256 * (1 - np.random.uniform(0, 1) * p['mag'] / 10.))
            img = Image.fromarray(img)
            if np.random.rand() - 0.5 > 0:
                img = ImageOps.solarize(img, threshold=th)
            else:
                img = ImageOps.invert(img)
                img = ImageOps.solarize(img, threshold=th)
                img = ImageOps.invert(img)

        #posterize
        elif p['op'] == 'pos':
            bit = int(8.5 - np.random.uniform(0, 0.5) * p['mag'])
            img = Image.fromarray(img)
            img = ImageOps.posterize(img, bit)

        
        img = np.asarray(img, dtype=np.uint8)

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
