import random as ra
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from numpy import linalg as LA
from scipy.misc import logsumexp

#---------------------------------------------------------------------------------
# Utility Functions - There is no need to edit code in this section.
#---------------------------------------------------------------------------------

# Reads a data matrix from file.
# Output: X: data matrix.
def readData(file):
    X = []
    with open(file,"r") as f:
        for line in f:
            X.append(map(float,line.split(" ")))
    return np.array(X)
    

# plot 2D toy data
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
#        Label: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
#        title: a string represents the title for the plot
def plot2D(X,K,Mu,P,Var,Label,title):
    r=0.25
    color=["r","b","k","y","m","c"]
    n,d = np.shape(X)
    per= Label/(1.0*np.tile(np.reshape(np.sum(Label,axis=1),(n,1)),(1,K)))
    fig=plt.figure()
    plt.title(title)
    ax=plt.gcf().gca()
    ax.set_xlim((-20,20))
    ax.set_ylim((-20,20))
    for i in xrange(len(X)):
        angle=0
        for j in xrange(K):
            cir=pat.Arc((X[i,0],X[i,1]),r,r,0,angle,angle+per[i,j]*360,edgecolor=color[j])
            ax.add_patch(cir)
            angle+=per[i,j]*360
    for j in xrange(K):
        sigma = np.sqrt(Var[j])
        circle=plt.Circle((Mu[j,0],Mu[j,1]),sigma,color=color[j],fill=False)
        ax.add_artist(circle)
        text=plt.text(Mu[j,0],Mu[j,1],"mu=("+str("%.2f" %Mu[j,0])+","+str("%.2f" %Mu[j,1])+"),stdv="+str("%.2f" % np.sqrt(Var[j])))
        ax.add_artist(text)
    plt.axis('equal')
    plt.show()

#---------------------------------------------------------------------------------



#---------------------------------------------------------------------------------
# K-means methods - There is no need to edit code in this section.
#---------------------------------------------------------------------------------

# initialization for k means model for toy data
# input: X: n*d data matrix;
#        K: number of mixtures;
#        fixedmeans: is an optional variable which is
#        used to control whether Mu is generated from a deterministic way
#        or randomized way
# output: Mu: K*d matrix, each row corresponds to a mixture mean;
#         P: K*1 matrix, each entry corresponds to the weight for a mixture;
#         Var: K*1 matrix, each entry corresponds to the variance for a mixture;    
def init(X,K,fixedmeans=False):
    n, d = np.shape(X)
    P=np.ones((K,1))/float(K)

    if (fixedmeans):
        assert(d==2 and K==3)
        Mu = np.array([[4.33,-2.106],[3.75,2.651],[-1.765,2.648]])
    else:
        # select K random points as initial means
        rnd = np.random.rand(n,1)
        ind = sorted(range(n),key = lambda i: rnd[i])
        Mu = np.zeros((K,d))
        for i in range(K):
            Mu[i,:] = np.copy(X[ind[i],:])

    Var=np.mean( (X-np.tile(np.mean(X,axis=0),(n,1)))**2 )*np.ones((K,1))
    return (Mu,P,Var)


# K Means method
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# output: Mu: K*d matrix, each row corresponds to a mixture mean;
#         P: K*1 matrix, each entry corresponds to the weight for a mixture;
#         Var: K*1 matrix, each entry corresponds to the variance for a mixture;
#         post: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
def kMeans(X, K, Mu, P, Var):
    prevCost=-1.0; curCost=0.0
    n=len(X)
    d=len(X[0])
    while abs(prevCost-curCost)>1e-4:
        post=np.zeros((n,K))
        prevCost=curCost
        #E step
        for i in xrange(n):
            post[i,np.argmin(np.sum(np.square(np.tile(X[i,],(K,1))-Mu),axis=1))]=1
        #M step
        n_hat=np.sum(post,axis=0)
        P=n_hat/float(n)
        curCost = 0
        for i in xrange(K):
            Mu[i,:]= np.dot(post[:,i],X)/float(n_hat[i])
            # summed squared distance of points in the cluster from the mean
            sse = np.dot(post[:,i],np.sum((X-np.tile(Mu[i,:],(n,1)))**2,axis=1))
            curCost += sse
            Var[i]=sse/float(d*n_hat[i])
        print(curCost)
    # return a mixture model retrofitted from the K-means solution
    return (Mu,P,Var,post) 
#---------------------------------------------------------------------------------



#---------------------------------------------------------------------------------
# PART 1 - EM algorithm for a Gaussian mixture model
#---------------------------------------------------------------------------------
def differenceSquared(X,Mu):
    n,d = np.shape(X)
    K = np.shape(Mu)[0]

    # Subtract means from corresponding datapoints and store in a (n,d,K)-shaped array
    # Extend datapoint array into 3D array by repeating the X array K times on axis=2.
    normsSquared  =  X.reshape(n,d,1).repeat(K,2)
    # Same for Mu; repeat n times on axis=2. Then transpose to change shape from (K,d,n) to (n,d,K),
    #     and subtract from the X 3D array
    normsSquared -= Mu.reshape(K,d,1).repeat(n,2).transpose((2,1,0))
    # Take squared norm along axis=1, along the feature dimension, which has dimensionality d.
    #     normsSquared.shape now equals (n,K)
    normsSquared = (normsSquared**2.0).sum(1)

    return normsSquared

# E step of EM algorithm
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# output:post: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
#        LL: a Loglikelihood value

def Estep(X,K,Mu,P,Var):
    n,d = np.shape(X) # n data points of dimension d
    posterior = np.zeros((n,K)) # posterior probabilities to compute
    LL = 0.0    # the LogLikelihood
    # Write your code here

    normsSquared = differenceSquared(X,Mu)

    # Debugging to see if normsSquared is correct.
    # The squared norm of x0-m0 should equal
    
    # Calculate the Spherical Normal/Gaussian Distribution of the datapoints given Mu and Var.
    # np.diag requires a 1D array; Var is given as 2D array with shape (K,1)
    # Also, all operations with variances here are division, so we take the inverse of Var.
    VarInv = np.diag(Var.reshape(K)**-1.0)
    exparg = np.dot(normsSquared, -VarInv/2.0)
    sphericalNormalDist = np.dot(np.exp(exparg), (VarInv/(2*np.pi))**(d/2.0))

    # P(x|theta) = P(datapoint|features) = the denominator in Bayes' theorem.
    # Expressed here on the diagonal of the nxn obvervationsInv matrix.
    # As this is the denominator, we take the inverse of each element to simply multiply later. 
    observations = np.dot(sphericalNormalDist, P).reshape(n)
    observationsInv = np.diag(observations**-1.0)

    # Use baye's theorem to calculate the posterior probability for each point to be in a cluster
    posterior = np.dot(np.dot(observationsInv, sphericalNormalDist), np.diag(P.reshape(K)))

    # The log-likelihood is just the sum of the log of the observations.
    LL = np.sum(np.log(observations))
    
    return (posterior,LL)


# M step of EM algorithm
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
#        post: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
# output:Mu: updated Mu, K*d matrix, each row corresponds to a mixture mean;
#        P: updated P, K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: updated Var, K*1 matrix, each entry corresponds to the variance for a mixture;
def Mstep(X,K,Mu,P,Var,post):
    n,d = np.shape(X) # n data points of dimension d

    # Write your code here
    # Effective number of points assigned to cluster corresponding
    # to index along cluster axis=1 in post
    n_hat = np.sum(post, axis=0)
    P  = 1.0*n_hat.reshape(K,1)/n

    Mu = np.dot(np.diag(n_hat**-1.0), np.dot(post.T,X))

    # Var requires squered norms of the datapoints offset by means.
    # It might be possible to extract the norms from post. Copy/paste from Estep is easier.

    normsSquared = differenceSquared(X,Mu)
    # The shape is now (n,K).

    # I am still confused as to why Var is an input. Keith doesn't use it, so I won't either.
    # the sum should be along the datapoint axis=0
    Var = np.sum(normsSquared * post, axis=0) * (n_hat**-1.0) / d

    return (Mu,P,Var)


# Mixture of Guassians
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# output: Mu: updated Mu, K*d matrix, each row corresponds to a mixture mean;
#         P: updated P, K*1 matrix, each entry corresponds to the weight for a mixture;
#         Var: updated Var, K*1 matrix, each entry corresponds to the variance for a mixture;
#         post: updated post, n*K matrix, each row corresponds to the soft counts for all mixtures for an example
#         LL: Numpy array for Loglikelihood values
def mixGauss(X,K,Mu,P,Var,estp=Estep,mstp=Mstep):
    n,d = np.shape(X) # n data points of dimension d
    post = np.zeros((n,K)) # posterior probabilities
    
    #Write your code here
    #Use function Estep and Mstep as two subroutines
    
    ll_new, ll_old, LL = np.inf, 0, []
    # Has ll_new increased by more than this moving threshold? If so, loop.
    while abs(ll_new - ll_old) >= 10**-6 * np.abs(ll_new) or  np.isinf(ll_new):
        ll_old = ll_new
        (post, ll_new) = estp(X,K,Mu,P,Var)
        (Mu,P,Var) = mstp(X,K,Mu,P,Var, post)

        LL.append(ll_new)      
        
    LL = np.asarray(LL)

    return (Mu,P,Var,post,LL)


# Bayesian Information Criterion (BIC) for selecting the number of mixture components
# input:  n*d data matrix X, a list of K's to try, times to try each K
# output: the highest scoring choice of K
def BICmix(X,Kset,returnBIC=False,tries=25):
    n,d = np.shape(X)
    #Write your code here

    BIC_best = -np.inf
    K_best = (None, BIC_best)
    for K in Kset:
        # A model with K gausians tries tries tries.
        # Its best log-likelihood from these tries is recorded.
        for i in range(tries):
            # Using randomized means for toy data X
            (Mu,P,Var) = init(X,K)
            # mixGauss returns a tuple. Its last element is a list of ordered LL's. 
            LL = mixGauss(X,K,Mu,P,Var)[-1][-1]

            # Using BIC definition from problem set 5,
            # where a higher BIC implies a better model.
            parameter_count = K*(d+2)-1 # From piazza @571
            BIC_try = LL - 0.5*parameter_count*np.log(n)
            if BIC_try > BIC_best:
                BIC_best = BIC_try
        if K_best[1] < BIC_best:
            K_best = (K, BIC_best)
    if returnBIC:
        return K_best
    else:
        return K_best[0]
#---------------------------------------------------------------------------------



#---------------------------------------------------------------------------------
# PART 2 - Mixture models for matrix completion
#---------------------------------------------------------------------------------

# RMSE criteria
# input: X: n*d data matrix;
#        Y: n*d data matrix;
# output: RMSE
def rmse(X,Y):
    return np.sqrt(np.mean((X-Y)**2))


# E step of EM algorithm with missing data
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# output:post: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
#        LL: a Loglikelihood value
def Estep_part2(X,K,Mu,P,Var):
    n,d = np.shape(X) # n data points of dimension d
    post = np.zeros((n,K)) # posterior probabilities to compute
    LL = 0.0    # the LogLikelihood
    
    # By masking x, we keep from worrying about data we don't have.
    X_ma = np.ma.masked_equal(X, 0)
    VarInv = np.diag(Var.reshape(K)**-1.0)

    # The log of probabilities must be extended in the datapoint dimension to be summed
    P_log = np.log(P).reshape(K,1).repeat(n, 1).T
    
    # Calculate the log of the spherical Normal/gaussian distribution
    normsSquared = differenceSquared(X_ma,Mu)
    nonzeroCounts = -(X != 0).sum(1).reshape(n,1)/2.0

    sphericalNormalDist_log  = np.dot(nonzeroCounts,np.log(2*np.pi*Var.reshape(1,K)))
    sphericalNormalDist_log += np.dot(normsSquared, -VarInv/2.0)

    # Calculate the log of the observations using advice in project3.pdf
    numerator = P_log + sphericalNormalDist_log
    observation_log = logsumexp(numerator, axis=1).reshape(n,1)

    LL = np.sum(observation_log)
    
    observation_log = observation_log.repeat(K,1)
    post = np.subtract(numerator, observation_log)
    return (np.exp(post),LL)

	
# M step of EM algorithm
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
#        post: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
# output:Mu: updated Mu, K*d matrix, each row corresponds to a mixture mean;
#        P: updated P, K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: updated Var, K*1 matrix, each entry corresponds to the variance for a mixture;
def Mstep_part2(X,K,Mu,P,Var,post, minVariance=0.25):
    n,d = np.shape(X) # n data points of dimension d
    
    #Write your code here
    X_ma = np.ma.masked_equal(X, 0)
    indicator = (X != 0)
    nonzeroCounts = indicator.sum(1).reshape(1,n)

    # find the means. This took time, and could be better, but it works. 
    Mu_denom = np.dot(post.T, indicator)
    Mu_numer = np.dot(post.T, X_ma)
    Mu_update = Mu_numer / Mu_denom
    for j in range(K):
        for i in range(d):
            if Mu_denom[j,i] < 1:
                Mu_update[j,i] = Mu[j,i]
    Mu = Mu_update
    
    # Cluster Probabilies
    P = post.sum(0)/float(n)

    # find the variances
    Var_numer = (post * differenceSquared(X_ma, Mu)).sum(0).reshape(K,1)
    Var_denom = np.dot(nonzeroCounts,post).T
    Var = Var_numer / Var_denom
    for variance in np.nditer(Var, op_flags=['readwrite']):
        if variance < minVariance:
            variance[...] = minVariance
    
    return (Mu,P,Var)

	
# mixture of Guassians
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# output: Mu: updated Mu, K*d matrix, each row corresponds to a mixture mean;
#         P: updated P, K*1 matrix, each entry corresponds to the weight for a mixture;
#         Var: updated Var, K*1 matrix, each entry corresponds to the variance for a mixture;
#         post: updated post, n*K matrix, each row corresponds to the soft counts for all mixtures for an example
#         LL: Numpy array for Loglikelihood values
def mixGauss_part2(X,K,Mu,P,Var):
    n,d = np.shape(X) # n data points of dimension d
    post = np.zeros((n,K)) # posterior probs tbd
    
    #Write your code here
    #Use function Estep and Mstep as two subroutines
    return mixGauss(X,K,Mu,P,Var,Estep_part2,Mstep_part2)


# fill incomplete Matrix
# input: X: n*d incomplete data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# output: Xnew: n*d data matrix with unrevealed entries filled
def fillMatrix(X,K,Mu,P,Var):
    n,d = np.shape(X)
    Xnew = np.copy(X)
    print '(n,K,d)',  (n,K,d)
    #Write your code here
    for t in range(n):
        for i in range(d):
            if Xnew[t,i] == 0:
                Xnew[t,i] = np.dot(P.reshape(K), Mu.T[i])        
    return Xnew
#---------------------------------------------------------------------------------
