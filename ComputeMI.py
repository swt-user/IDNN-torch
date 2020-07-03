import os, pickle
from collections import defaultdict, OrderedDict

import numpy as np

import simplebinMI
from my_kde import *
from my_hsic import compute_relative
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_style('darkgrid')

import utils


def MI(cfg, infoplane_measure='bin', full_dataset = None):

    measures = {}
    # configuration
    PLOT_LAYERS    = None
    DO_SAVE = False

    if infoplane_measure == 'upper':
        DO_UPPER = True
    else:
        DO_UPPER = False
    if infoplane_measure == 'lower':
        DO_LOWER = True
    else:
        DO_LOWER = False
    if infoplane_measure == 'bin':
        DO_BINNED = True
    else:
        DO_BINNED = False
    if infoplane_measure =='HSIC':
        DO_HSIC = True
    else:
        DO_HSIC = False

    noise_variance = 1e-3

    labelprobs = np.mean(full_dataset.Y.numpy(), axis=0)

    saved_labelixs={}
    for i in range(cfg['NUM_CLASSES']):
        saved_labelixs[i] = full_dataset.y == i

    if not os.path.exists(cfg['SAVE_DIR']):
        print("Directory %s not found" % cfg['SAVE_DIR'])
        return None
        
    # Load files saved during each epoch, and compute MI measures of the activity in that epoch
    print('*** Doing %s ***' % cfg['SAVE_DIR'])
    for epochfile in sorted(os.listdir(cfg['SAVE_DIR'])):
        if not epochfile.startswith('epoch'):
            continue
            
        fname = cfg['SAVE_DIR'] + "/" + epochfile
        with open(fname, 'rb') as f:
            d = pickle.load(f)

        epoch = d['epoch']
        #if epoch in measures[activation]: # Skip this epoch if its already been processed
        #    continue                      # this is a trick to allow us to rerun this cell multiple times)
            
        if epoch > cfg['NUM_EPOCHS']:
            continue

        print("Doing", fname)
        
        num_layers = len(d['data']['activity_tst'])

        if PLOT_LAYERS == None:
            PLOT_LAYERS = []
            for lndx in range(num_layers):
                #if d['data']['activity_tst'][lndx].shape[1] < 200 and lndx != num_layers - 1:
                PLOT_LAYERS.append(lndx)
                
        cepochdata = defaultdict(list)
        for lndx in range(num_layers):
            activity = d['data']['activity_tst'][lndx]
            
            # Compute marginal entropies
            if DO_UPPER:
                h_upper = entropy_estimator_kl(activity, noise_variance)

                # Layer activity given input. This is simply the entropy of the Gaussian noise
                hM_given_X = kde_condentropy(activity, noise_variance)

                # Compute conditional entropies of layer activity given output
                hM_given_Y_upper=0.
                for i in range(cfg['NUM_CLASSES']):
                    hcond_upper = entropy_estimator_kl(activity[saved_labelixs[i],:], noise_variance)
                    hM_given_Y_upper += labelprobs[i] * hcond_upper

                cepochdata['MI_XM_upper'].append( (h_upper - hM_given_X) )
                cepochdata['MI_YM_upper'].append( (h_upper - hM_given_Y_upper) )
                cepochdata['H_M_upper'  ].append( h_upper )

                pstr = 'upper: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_upper'][-1], cepochdata['MI_YM_upper'][-1])

            if DO_LOWER:
                h_lower = entropy_estimator_bd(activity, noise_variance)

                # Layer activity given input. This is simply the entropy of the Gaussian noise
                hM_given_X = kde_condentropy(activity, noise_variance)
                
                hM_given_Y_lower=0.
                for i in range(cfg['NUM_CLASSES']):
                    hcond_lower = entropy_estimator_bd(activity[saved_labelixs[i],:], noise_variance)
                    hM_given_Y_lower += labelprobs[i] * hcond_lower
                
                cepochdata['MI_XM_lower'].append(  (h_lower - hM_given_X) )
                cepochdata['MI_YM_lower'].append(  (h_lower - hM_given_Y_lower) )
                cepochdata['H_M_lower'  ].append(  h_lower )
                pstr = ' | lower: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_lower'][-1], cepochdata['MI_YM_lower'][-1])

            if DO_BINNED: # Compute binned estimates
                if cfg['ACTIVATION'] == 'relu' or 'softsign':
                    binxm = simplebinMI.bin_calc_information(full_dataset.X.numpy(), activity.numpy(), cfg['NUM_BINS'], 0, cfg['MAX_ELEMENT'])
                    binym = simplebinMI.bin_calc_information(full_dataset.Y.numpy(), activity.numpy(), cfg['NUM_BINS'], 0 , cfg['MAX_ELEMENT'])
                else:
                    binxm = simplebinMI.bin_calc_information(full_dataset.X.numpy(), activity.numpy(), cfg['NUM_BINS'], -1, 1)
                    binym = simplebinMI.bin_calc_information(full_dataset.Y.numpy(), activity.numpy(), cfg['NUM_BINS'], -1, 1)
                
                cepochdata['MI_XM_bin'].append( binxm )
                cepochdata['MI_YM_bin'].append( binym )
                pstr = ' | bin: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_bin'][-1], cepochdata['MI_YM_bin'][-1])
            
            if DO_HSIC:
                HSIC_xm = compute_relative(full_dataset.X, activity)
                HSIC_ym = compute_relative(full_dataset.Y, activity)

                cepochdata['MI_XM_HSIC'].append( HSIC_xm*1000)
                cepochdata['MI_YM_HSIC'].append( HSIC_ym*100 )

                pstr = ' | HSIC: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_HSIC'][-1], cepochdata['MI_YM_HSIC'][-1])
            print('- Layer %d %s' % (lndx, pstr) )

        measures[epoch] = cepochdata

    return measures

def plot_infoplane(cfg, infoplane_measure, measures, PLOT_LAYERS):
    'plot the infoplane details'
    sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=cfg['NUM_EPOCHS']))
    sm._A = []

    fig=plt.figure(figsize=(10,5))
    
    epochs = sorted(measures.keys())
    if not len(epochs):
        return None
    plt.subplot(1,2,1)    
    for epoch in epochs:
        c = sm.to_rgba(epoch)
        xmvals = np.array(measures[epoch]['MI_XM_'+infoplane_measure])[PLOT_LAYERS]
        ymvals = np.array(measures[epoch]['MI_YM_'+infoplane_measure])[PLOT_LAYERS]

        plt.plot(xmvals, ymvals, c=c, alpha=0.1, zorder=1)
        plt.scatter(xmvals, ymvals, s=20, facecolors=[c for _ in PLOT_LAYERS], edgecolor='none', zorder=2)

    plt.ylim([0, 0.8])
    plt.xlim([0, 12])
    # plt.ylim([0, 3.5])
    # plt.xlim([0, 14])
    plt.xlabel('I(X;M)')
    plt.ylabel('I(Y;M)')
    plt.title(cfg['SAVE_DIR'])
        
    cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8]) 
    plt.colorbar(sm, label='Epoch', cax=cbaxes)
    plt.tight_layout()
    plt.show()

    
def plot_SNR(cfg, PLOT_LAYERS):

    plt.figure(figsize=(12,5))

    gs = gridspec.GridSpec(1, len(PLOT_LAYERS))
    saved_data = {}
    
    cur_dir = cfg['SAVE_DIR']
    if not os.path.exists(cur_dir):
        return
        
    epochs = []
    means = []
    stds = []
    wnorms = []
    
    for epochfile in sorted(os.listdir(cur_dir)):
        if not epochfile.startswith('epoch'):
            continue
            
        with open(cur_dir + "/"+epochfile, 'rb') as f:
            d = pickle.load(f)
            
        epoch = d['epoch']
        epochs.append(epoch)
        wnorms.append(d['data']['weights_norm'])
        means.append(d['data']['gradmean'])
        stds.append(d['data']['gradstd'])

    wnorms, means, stds = map(np.array, [wnorms, means, stds])
    saved_data[cfg['ACTIVATION']] = {'wnorms':wnorms, 'means': means, 'stds': stds}

    for lndx,layerid in enumerate(PLOT_LAYERS):
        plt.subplot(gs[0, lndx])
        plt.plot(epochs, means[:,layerid], 'b', label="Mean")
        plt.plot(epochs, stds[:,layerid], 'orange', label="Std")
        #plt.plot(epochs, means[:,layerid]/stds[:,layerid], 'red', label="SNR")
        plt.plot(epochs, wnorms[:,layerid], 'g', label="||W||")

        plt.title('%s - Layer %d'%(cfg['ACTIVATION'], layerid))
        plt.xlabel('Epoch')
        plt.gca().set_xscale("log", nonposx='clip')
        plt.gca().set_yscale("log", nonposy='clip')
    

    plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0.2))
    plt.tight_layout()
    plt.show()

            