import numpy as np
import copy

# sampleProposal(n): samples a random permutation over {0,1,...,n-1}
def sampleInitProposal(n):
    return np.random.permutation(range(0,n))

# sampleProposalBySwap(n):pick two positions of the input permutation and swap them 
def sampleProposalBySwap(s_in):
    ix = np.random.random_integers(min(s_in),max(s_in),2) #pick two random positions in the perm, with replacement
    shat = copy.copy(s_in)                                #now do a shallow copy (silly python) and swap those two
    s_in[ix[0]]=shat[ix[1]]
    s_in[ix[1]]=shat[ix[0]]
    return s_in

# Nichole's restricted sample space sampler -- see emails for discussion
def sampleProposalFixed4(N):
    flip = np.random.randint(1,N-2)
    if flip == 1:
        second_flip = np.random.randint(1,N-2)
        if second_flip == 1:
            perm = np.append([0,2,3,1],\
                             np.random.permutation(filter(lambda x: x not in [0,2,3,1],\
                                                          range(0,N))))
        else:
            perm = np.append([0,2,3,4],\
                             np.random.permutation(filter(lambda x: x not in [0,2,3,4],\
                                                          range(0,N))))
            
    else:
        second_flip = np.random.randint(1,N-4)
        if second_flip == 1:
            perm = np.append([0,2,4,1],\
                             np.random.permutation(filter(lambda x: x not in [0,2,4,1],\
                                                          range(0,N))))
        elif second_flip == 2:
            perm = np.append([0,2,4,6],\
                             np.random.permutation(filter(lambda x: x not in [0,2,4,6],\
                                                          range(0,N))))
        elif second_flip == 3:
            perm = np.append([0,2,4,7],\
                             np.random.permutation(filter(lambda x: x not in [0,2,4,7],\
                                                          range(0,N))))
        else:
            perm = np.append([0,2,3,8],\
                             np.random.permutation(filter(lambda x: x not in [0,2,3,8],\
                                                          range(0,N))))
    return perm


def sampleProposalMinIndexSwap(s_in,n,alpha_random):
    #print 'calling with |s_in|=' + str(len(s_in)) + 'n=' + str(n) + 'alpha_random=' +str(alpha_random)
    flip = np.random.sample()
    if (flip > alpha_random):
        return sampleProposalFixed4(n)
    
    S = set()
    i = 0
    #print 'hit the while...\n'
    while ( (i<len(s_in)) ):
        if((s_in[i]^i) in S):
            break
        #print 'in the while...\n'
        S.add(s_in[i]^i)
        i=i+1
        #print 'i=%d',i
    if (i==len(s_in)):
        return s_in  #s_in is an orthomorphism
    
    #otherwise, i+1 is the index of the first non-orthomophic point under s_in
    ix = np.random.randint(i+1,len(s_in))
    shat = copy.copy(s_in)                                #now do a shallow copy (silly python) and swap those two
    s_in[ix]=shat[i+1]
    s_in[i+1]=shat[ix]
    return s_in

    
        

def sampleProposal(method,s_in=[],num_points=0,alpha_random=0):
    if method=='Init':
        return sampleInitProposal(num_points)
    elif method=='BySwap':
        return sampleProposalBySwap(s_in)
    elif method=='Fixed4':
        return sampleProposalFixed4(num_points)
    elif method=='MinIndexSwap':
        return sampleProposalMinIndexSwap(s_in,num_points,alpha_random)
    else:
        print 'Unrecognized sampling method, running Init by default'
        return sampleInitProposal(num_points)
