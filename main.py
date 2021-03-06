import project3 as p3
import random as ra
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from numpy import linalg as LA


#-------------------------------------------------------------------------------
# Section 1.a
#-------------------------------------------------------------------------------
# Test the provided kMeans method on the toy data for K=[1,2,3,4] with
# several different random initializations. Provide plots of the solution
# for each K that minimizes the total distortion cost.

# Write your code here
X = p3.readData('toy_data.txt')
for K in [1,2,3,4]:
    (Mu,P,Var) = p3.init(X,K)
    (Mu,P,Var,post) =  p3.kMeans(X,K,Mu,P,Var)
    p3.plot2D(X,K,Mu,P,Var,post,'K = %i clusters' % K)
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Section 1.b
#-------------------------------------------------------------------------------
# Fill in the Estep, Mstep, and mixGauss methods.
# Test your Estep using K=3, after initializing using
# (Mu,P,Var) = p3.init(X,K,fixedmeans=True).  The returned log-likelihood
# should be -1331.67489.

# Write your code here
K = 3
(Mu,P,Var) = p3.init(X,K,fixedmeans=True)
LL = p3.Estep(X,K,Mu,P,Var)[1]
print 'LL should be -1331.67489 after one Estep() call, and equals ', LL
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Section 1.c and 1.d
#-------------------------------------------------------------------------------
# 1) Test your implementation of EM on the toy data set by checking that the
#	   LogLikelihoods at each iteration are indeed monotonically increasing; and,
# 2) Compare your algorithm's Log-likelihood to the one provided in the project.
#
# Once convinced that your EM implementation is working, generate plot like you
# did for kMeans above.  Compare these plots to those achieved using kMeans and
# explain when, how, and why they differ.

# Each run of mixGauss shouldn't take more than a few seconds.

# Write your code here
print 'SECTION 1.C TEST 1: LL from mixGauss is monotonically increasing:', 
LL = p3.mixGauss(X,K,Mu,P,Var)[-1]
ll_old = LL[0]
passed = True
for ll_current in LL:
    if ll_old > ll_current:
        passed = False
        break
    ll_old = ll_current
print passed,'\n'

provided_ll = -1138.89248
print 'SECTION 1.C TEST 2: My algorithm\'s Log-likelihood equals', provided_ll, ':', provided_ll == np.round(LL[-2],5)
print '    My code has np.round(LL[-2],5) = ', np.round(LL[-2],5)

# Section 1.d plots
for K in [1,2,3,4]:
    (Mu,P,Var) = p3.init(X,K)
    (Mu,P,Var,post,LL) = p3.mixGauss(X,K,Mu,P,Var)
    p3.plot2D(X,K,Mu,P,Var,post,'EM: K = %i clusters' % K)

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Section 1.e
#-------------------------------------------------------------------------------
# Fill in the BICmix method, then find the best K in [1,2,3,4] for the toy
# dataset.  Report the best K and the corresponding BIC score.

# Write your code here
(best_K, score) = p3.BICmix(X,[1,2,3,4],returnBIC=True)
print 'The best K is K = ',best_K, ' with BIC score = ', score

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Section 2.d
#-------------------------------------------------------------------------------
# Fill in the Estep_part2, Mstep_part2, and mixGaus_part2 methods.
# Run the file matrixCompletionTest.py (with irrelevant sections commented out)
# to test the E step, M step, mixGauss, and fillMatrix functions you wrote in part 2.
# (Note that you may need to comment out the fillMatrix portion of 
# matrixCompletionTest.py if you run it at this point before writing you fillMatrix
# function.) The expected results can be found in the file
# matrixCompletionTest_Solutions.txt. You do not need to include your code for
# this debugging / validation.

# Tip: The terminal command used to write the script output to
# matrixCompletionTest_Solutions.txt was:

# python matrixCompletionTest.py >> matrixCompletionTest_Solutions.txt

# So you can use the same line to write your own output to a file with a 
# different name (e.g. 'validation.txt', if there is not already a file 
# with that name in the same folder).

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Section 2.e
#-------------------------------------------------------------------------------
# Run the mixture model with K = 12 on 'netflix_incomplete.txt', and verify that
# the log-likelihood increases monotonically. Also run the model with K = 1,
# and verify that the log-likelihood is -1521060.95399.

# Write your code here
K=1
X = p3.readData('netflix_incomplete.txt')
(Mu,P,Var) = p3.init(X,K)
LL = p3.mixGauss_part2(X,K,Mu,P,Var)[-1]
provided_ll = -1521060.95399
print 'SECTION 2.E TEST 1: My algorithm\'s Log-likelihood equals', provided_ll, ':', provided_ll == np.round(LL[-2],5)
print '    My code has np.round(LL[-2],5) = ', np.round(LL[-2],5)

K=12
print 'SECTION 2.E TEST 2: LL from mixGauss is monotonically increasing:',
(Mu,P,Var) = p3.init(X,K)
LL = p3.mixGauss_part2(X,K,Mu,P,Var)[-1]

ll_old = LL[0]
passed = True
for ll_current in LL:
    if ll_old > ll_current:
        passed = False
        break
    ll_old = ll_current
print passed,'\n'
 

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Section 2.h
#-------------------------------------------------------------------------------
# Run fillMatrix on 'netflix_incomplete.txt' with K=12 and mixture model on
# 'netflix_complete.txt' and report the root mean squared error between the two
# matrices using rmse.
Xc = p3.readData('netflix_complete.txt')
Xpred = p3.fillMatrix(X,K,Mu,P,Var)
print 'RMSE = ', p3.rmse(Xpred,Xc)


# Write your code here

#-------------------------------------------------------------------------------
