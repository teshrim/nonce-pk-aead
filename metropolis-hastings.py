#! /usr/bin/env python

import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter
import pickle
import copy
import time
import sys
import mh_sampling as samp

plt.close('all')

#---------- Global values ----------#
fname_base = '/Users/teshrim/Dropbox/code/python/'

nbits=5                      #number of bits
N = 2**nbits                 #permutations will be over {0,1,...,N-1}
Nfact = factorial(N)
Ntrials =  1000              #number of trials before creating histogram
MAX_STEPS = int(2**16)       #number of steps to use in MH walk

BASE = 1000                     #weighting are all of the form BASE**{z} where z is some property of interest
LOG2_BASE = np.log2(BASE)    #compute this once, rather than every time it's needed

MAX_WEIGHT = float(BASE**N)
SWITCH_THRESHOLD = MAX_WEIGHT
#SWITCH_THRESHOLD = float(BASE**(N/2.0))

ALPHA_RANDOM=0.01

SYMMETRIC_P = False          #is the original transition matrix symmetric?
if SYMMETRIC_P == False:     #... if not, need to explicitly define P, or how [P]_ij is computed online
    pass #define P


SAVE_FIGS_TO_FILE = True
PRINT_AT_TRIAL = 10           #prints trial number and trial execution time at trials PRINT_AT_TRIAL*k for k>0

#---------- Function definitions ----------#



def MinNonOrthoIndex(s):
    #print 'In MinNonOrthoIndex with len(s)=%d',len(s)
    S = set()
    i = 0
    while ((i < len(s))):
        if ((s[i]^i) in S):
            break
        S.add(s[i]^i)
        i=i+1
        #print i
    return i
        
# w_fpcount(s_prop): weight function w: S -> R^+, where w(s)=BASE^{#fixed points in s}
def w_fpcount(s_prop):
    #print 'called w_fpcount'
    counts = Counter([s_prop[i]==(i+1) for i in range(len(s_prop))])
    return float(BASE**counts[True])

# w_ortho(s_prop): weight function w: S -> R^+, where w(s)=BASE^{# "orthmorphism respecting" points in s}
def w_ortho(s_prop):
    #    s_prop= [bv.BitVector(intVal=elt,size=nbits) for elt in s_prop]
    #   F pix_xor_x = [ bv.BitVector(intVal=i,size=nbits)^s_prop[i] for i in range(len(s_prop)) ]
    #print 'called w_ortho'
    pix_xor_x = [ s_prop[i]^i for i in range(len(s_prop)) ]
    counts = [0 for i in range(N)]
    for i in range(N):
        ix = int(pix_xor_x[i])
        counts[ix] = counts[ix]+1
    good = Counter([counts[j]==1 for j in range(N)]) #a value has count of 1 if it appears exactly once in pi(x)+x
                                                     #...making it a "good" value under pi(x)+x
    ngood = good[True]
    
    return float(BASE**ngood)


# myMHalg(wfunc, state_init, MAX_STEPS, samp_method): runs the M-H alg from initial state state_init
# using weight function wfunc, for a total of MAX_STEPS steps, using sampling method named by string samp_method
def myMHalg(wfunc, state_init, MAX_STEPS, samp_method):
    
    state_current = state_init;
    accept_pr = 0.0;
    max_weight = 0.0;
    weights = [0.0 for i in range(MAX_STEPS)];
    acceptprobs = [0.0 for i in range(MAX_STEPS)];

    for step in range(MAX_STEPS):

        w_current = wfunc(state_current)

        if samp_method == 'Init':
            if w_current > SWITCH_THRESHOLD:  #you're going to sample by swap
                state_prop = samp.sampleProposal('BySwap',state_current)
                eq_test = [state_prop[i]==state_current[i] for i in range(N)]
                if (False in eq_test):    #state_current and state_prop are different
                    p_forw = 2.0 / (N**2) #prob of moving from current to (different) proposed is 2/N^2 under swap
                    w_prop = wfunc(state_prop)
                    if w_prop > SWITCH_THRESHOLD: #if state_prop is "good", you'll sample by swap from there
                        p_back = p_forw           #...so prob of coming back to current is same as prob of leaving current
                    else:                         #if state_prop is "bad", you'll sample a uniform perm from there
                        p_back = 1.0 / Nfact      #...so prob of coming back to current is 1/N!
            
                else:  #state_prop = state_current
                    p_forw = 1.0 / N #under swaping, prob of getting same state back is 1/N
                    p_back = 1.0 / N
                    w_prop = w_current

            
            else: #you're going to sample uniformly
                  #state_prop = samp.sampleProposal('Init',num_points=N)
                state_prop = samp.sampleProposal('Init',num_points=N)
                w_prop = wfunc(state_prop)
                p_forw = 1    #it doesn't matter what constant these are set to
                p_back = 1    #just so long as they are the same

        elif samp_method == 'BySwap':
            state_prop = samp.sampleProposal('BySwap',state_current)
            eq_test = [state_prop[i]==state_current[i] for i in range(N)]
            if (False in eq_test):    #state_current and state_prop are different
                p_forw = 2.0 / (N**2) #prob of moving from current to (different) proposed is 2/N^2 under swap
                w_prop = wfunc(state_prop)
                p_back = p_forw           #prob of coming back to current is same as prob of leaving current
            
            else:  #state_prop = state_current
                p_forw = 1.0 / N #under swaping, prob of getting same state back is 1/N
                p_back = 1.0 / N
                w_prop = w_current

            
        elif samp_method == 'Fixed4':
            if w_current > SWITCH_THRESHOLD:  #you're going to sample by swap
                state_prop = samp.sampleProposal('BySwap',state_current)
                eq_test = [state_prop[i]==state_current[i] for i in range(N)]
                if (False in eq_test):    #state_current and state_prop are different
                    p_forw = 2.0 / (N**2) #prob of moving from current to (different) proposed is 2/N^2 under swap
                    w_prop = wfunc(state_prop)
                    if w_prop > SWITCH_THRESHOLD: #if state_prop is "good", you'll sample by swap from there
                        p_back = p_forw           #...so prob of coming back to current is same as prob of leaving current
                    else:                         #if state_prop is "bad", you'll sample a uniform perm from there
                        p_back = 1.0 / Nfact      #...so prob of coming back to current is 1/N!
            
                else:  #state_prop = state_current
                    p_forw = 1.0 / N #under swaping, prob of getting same state back is 1/N
                    p_back = 1.0 / N
                    w_prop = w_current

            
            else: #you're going to sample uniformly
                  #state_prop = samp.sampleProposal('Init',num_points=N)
                state_prop = samp.sampleProposal('Fixed4',num_points=N)
                w_prop = wfunc(state_prop)
                p_forw = 1    #it doesn't matter what constant these are set to
                p_back = 1    #just so long as they are the same


        elif samp_method == 'MinIndexSwap':
            state_prop = samp.sampleProposal('MinIndexSwap',state_current,N,ALPHA_RANDOM)
            w_prop = wfunc(state_prop)
            ix_c = MinNonOrthoIndex(state_current) #first non-ortho point under s_current
            ix_p = MinNonOrthoIndex(state_prop) #first non-ortho point under s_prop
            
            diff = [state_prop[i] == state_current[i] for i in range(len(state_current))]
            try:
                first_diff = diff.index(False)
            except ValueError: #if there's no 'False' in diff, diff.index() throws an exception
                first_diff = N+1
            
            if (first_diff == ix_c): #state_prop[0:ix_c-1]=state_current[0:ix_c-1], look for extensions
                p_forw = (ALPHA_RANDOM / factorial(N-4)) + (1-ALPHA_RANDOM)*(1.0 / (N - ix_c))
                if (ix_p > ix_c): #state_prop is an ortho-extension of state_current
                    p_back = (ALPHA_RANDOM / factorial(N-4))
                else: #ix_p=ix_c, no extention
                    p_back=p_forw

            elif (first_diff < ix_c):
                p_forw = (ALPHA_RANDOM / factorial(N-4)) #you only can get to state_prop by random perm choice
                if (first_diff == (ix_c-1)):    #state_prop can't be orthomorphic at ix_c-1, but it is up to that
                                                #because state_current is; in particular ix_p = ix_c-1
                    p_back = (ALPHA_RANDOM / factorial(N-4)) + (1-ALPHA_RANDOM)*(1.0 / (N - ix_p))
                else: #you can't get to state_current from state_prop with one swap, so only by random perm choice
                    p_back = p_forw
            else:
                if (ix_p == ix_c +1):
                    p_forw = (ALPHA_RANDOM / factorial(N-4)) + (1-ALPHA_RANDOM)*(1.0 / (N - ix_c))
                else:
                    p_forw = ALPHA_RANDOM / factorial(N-4)
                
                p_back = (ALPHA_RANDOM / factorial(N-4))

        else:
            print 'Unknown sampling method\n'
            
        #w_prop = wfunc(state_prop);
        max_weight = np.log2(max(max_weight,w_current)) / LOG2_BASE 
        
        weights[step]=w_current;

        if SYMMETRIC_P:
            accept_pr = min(1,float(w_prop)/float(w_current))
        else: 
            accept_pr = min(1,(float(w_prop)*p_back)/(float(w_current)*p_forw))
            #pass
            
        acceptprobs[step] = accept_pr;
        
        if accept_pr == 1:
            #print 'Step=' + repr(step) + ': ' + repr(accept_pr) + ' (ACCEPTING)'
            state_current = state_prop;
        else:
            flip = np.random.sample();
            if flip <= accept_pr:
                #print 'Step=' + repr(step) + ': ' + repr(flip) + '|' + repr(accept_pr) + ' (ACCEPTING)'
                state_current = state_prop;
                #myPrint(state_current,w(state_current));

    return(state_current,max_weight,weights,acceptprobs)

# main(wfunc,samp_method): main body of code, called with weight function name wfunc (as string) using samp_method sampling method
def main(wfunc,samp_method):
    state_final = [list() for i in range(Ntrials)];
    w_init = [-1 for i in range(Ntrials)];
    w_final = [-1 for i in range(Ntrials)];
    w_ensemble = [0 for i in range(MAX_STEPS)];
    ap_ensemble = [0 for i in range(MAX_STEPS)];
    print "Running w=" + str(wfunc.__name__) + ' with samp_method=' + samp_method+ " with N=" + str(N) + " Ntrials=" + str(Ntrials) + ", BASE=" + str(BASE) + ", MAX_STEPS=" + str(MAX_STEPS)

    #run MH a bunch of times from random initial points

    mytick = time.time()
    for i in range(Ntrials):

        s_init = samp.sampleProposal('Fixed4',num_points=N);
        ws=wfunc(s_init);
        w_init[i] = np.log2(ws) / LOG2_BASE
        if (i % PRINT_AT_TRIAL == 0):
            mytock = time.time()
            myticktock = mytock-mytick
            mytick = mytock
            print 'Trial ' + str(i)+ ', elapsed: ' + str(myticktock) + 'sec'
        
        returnval = myMHalg(wfunc,s_init, MAX_STEPS,samp_method);
        state_final[i] = returnval[0];
        w_final[i] = np.log2(wfunc(state_final[i])) / LOG2_BASE;
        w_ensemble = [w_ensemble[i]+(np.log2(returnval[2][i])/LOG2_BASE) for i in range(MAX_STEPS)]; 
        ap_ensemble = [ap_ensemble[i]+returnval[3][i] for i in range(MAX_STEPS)];
        
    w_ensemble = [(w_ensemble[i] / float(Ntrials)) for i in range(MAX_STEPS)];
    ap_ensemble = [(ap_ensemble[i] / float(Ntrials)) for i in range(MAX_STEPS)];
                   
    #write data to file
    run_name = str(wfunc.__name__) + 'samp_' + samp_method + '_N' + str(N) +'_Ntrials'+str(Ntrials)+'_base'+str(BASE)+'_maxsteps'+str(MAX_STEPS)+ '_switch' + str(int(np.log2(SWITCH_THRESHOLD)/LOG2_BASE))
    fname = open(fname_base+run_name,'w')
    pickle.dump(state_final,fname)
    pickle.dump(w_final,fname)
    pickle.dump(w_init,fname)
    pickle.dump(w_ensemble,fname)
    pickle.dump(ap_ensemble,fname)
    fname.close()

    print 'Wrote data, all done!'

    #Make plots
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.25,hspace=0.75)
    ax1 = fig.add_subplot(2,1,1)
    upper = int(max(w_init))+2;
    bins = [-0.5 + float(elt) for elt in range(upper)] 
    c,b,foo = ax1.hist(w_init,bins,normed=True,facecolor='blue')
    if wfunc.__name__ == w_fpcount.__name__:
        x_axis_str = '# fixed points (0 to ' + str(N) + ', but prob=0 above ' + str(upper-2) +')'
    elif wfunc.__name__== w_ortho.__name__:
        x_axis_str = '# good positions (0 to ' + str(N) + ', but prob=0 above ' + str(upper-2) +')'
    else:
        x_axis_str = ' '
    ax1.set_xlabel(x_axis_str)
    ax1.set_ylabel('estimated probability\n over ' + str(Ntrials) + ' runs')
    ax1.set_xlim(-0.5,upper)
    ax1.set_ylim(0,1)
    ax1.set_title('Initial samples')
    ax1.grid(True)
    
    upper = int(max(w_final))+2;
    bins = [-0.5 + float(elt) for elt in range(upper)] 
    ax2 = fig.add_subplot(2,1,2)
    ax2.hist(w_final,bins,normed=True,facecolor='green')
    if wfunc.__name__ == w_fpcount.__name__:
        x_axis_str = '# fixed points (0 to ' + str(N) + ', but prob=0 above ' + str(upper-2) +')'
    elif wfunc.__name__ == w_ortho.__name__:
        x_axis_str = '# good positions (0 to ' + str(N) + ', but prob=0 above ' + str(upper-2) +')'
    else:
        x_axis_str = ' '
    ax2.set_xlabel(x_axis_str)
    ax2.set_ylabel('estimated probability\n over ' + str(Ntrials) + ' runs')
    ax2.set_xlim(-0.5,upper)
    ax2.set_ylim(0,1)
    ax2.set_title('MH samples')
    ax2.grid(True)

    fig2 = plt.figure()
    fig2.subplots_adjust(bottom=0.25,hspace=.75)
    upper = int(max(w_ensemble))+2;
    lower = int(min(w_ensemble))-2;
    ax3 = fig2.add_subplot(2,1,1)
    x_axis_str = 'trial'
    ax3.set_xlabel(x_axis_str)
    ax3.set_ylabel('log weight')
    ax3.set_ylim(lower,upper)
    ax3.set_title('Weight trajectory,\n ensemble average (log, ' + str(Ntrials)+ ' runs)')
    ax3.grid(True)
    ax3.plot(w_ensemble)

    upper = (max(ap_ensemble))+.1;
    lower = (min(ap_ensemble))-.1;
    avg_ap = sum(ap_ensemble) / float(MAX_STEPS)
    ax4 = fig2.add_subplot(2,1,2)
    x_axis_str = 'trial'
    ax4.set_xlabel(x_axis_str)
    ax4.set_ylabel('accept prob')
    ax4.set_ylim(lower,upper)
    ax4.set_title('Acceptance probability,\n ensemble average (' +str(Ntrials)+ ' runs)')
    ax4.grid(True)
    ax4.plot(ap_ensemble,'b', (np.ones(MAX_STEPS) * avg_ap), 'r')

    if SAVE_FIGS_TO_FILE:
        if wfunc.__name__ == w_fpcount.__name__:
            fname = './FPHist_' + samp_method + '_N' + str(N) + '_Ntrials' + str(Ntrials) + '_MAXSTEPS' + str(MAX_STEPS) + '_BASE' + str(BASE) + '_switch' + str(int(np.log2(SWITCH_THRESHOLD)/LOG2_BASE))
        elif wfunc.__name__ == w_ortho.__name__:
            fname = './OrthoHist_' + samp_method +'_N' + str(N) + '_Ntrials' + str(Ntrials) + '_MAXSTEPS' + str(MAX_STEPS) + '_BASE' + str(BASE) + '_switch' + str(int(np.log2(SWITCH_THRESHOLD)/LOG2_BASE))
        else:
            fname = './UnkHist_' + samp_method +'_N' + str(N) + '_Ntrials' + str(Ntrials) + '_MAXSTEPS' + str(MAX_STEPS) + '_BASE' + str(BASE) + '_switch' + str(int(np.log2(SWITCH_THRESHOLD)/LOG2_BASE))
            #plt.savefig(fname)
        pp = PdfPages(fname + '.pdf')
        pp.savefig(fig)
        pp.savefig(fig2)
        pp.close()
        #plt.show()
    else:
        plt.show()

#---------- Protected main execution ----------#

if __name__ == '__main__':
    start_time = time.time()
    main(w_ortho,'MinIndexSwap')
    end_time = time.time()
    print 'Total elapsed time: ' + str(end_time-start_time)

#---------- Protected main execution ----------#


""" all below is old stuff

# myPrint(s,v): diagnostic tool, prints the state s and value v (e.g. the weight)
def myPrint(s,v):
    print repr([elt for elt in s]) + ':' + repr(v)

# myPrintFP(s): diagnostic tool, prints the fixed points of permutation s
def myPrintFP(s):
    temp = str()
    for i in range(len(s)):
        if s[i]==i: temp=temp + repr(i+1) + ' '
    print 'fp are: ' + temp

"""
