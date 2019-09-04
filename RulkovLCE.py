"""
Created on Tue Sep 03 , 2019

@author: Abhirup Banerjee
contact : abhirup.banerjee@pik-potsdam.de
"""


#Background
'====================================='
#paper : Regularization of Synchronized Chaotic Bursts 

#working formula : 
#	X(n+1)= f(x_n,y_n) 
#	Y(n+1)= g(x_n,y_n) 
#where
#        f(x_n,y_n)= alpha / (1+ X(i)^2) + Y(i) 
#        g(x_n,y_n)= Y(i) -sigma*X(i) - beta 
#
#It has a tangent
# 
#    df/dx= -2*x*alpha/(1+x^2)^2
#	 df/dy= 1
#	 dg/dx= - sigma 
#	 dg/dy= 1

#Initial conditions (xi,yi)= (0.1,0.3) 
#alpha is the bursting parameter ~ (4.1 - 4.5) ; beta=0.0009 , sigma= 0.0011 


import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm

alpharange=np.arange(3.5,5.0,0.01)
MaxLce=[]
MinLce=[]
#f=open("Rulkov_Lyapunov.txt","w")

for alpha in tqdm(alpharange):

    # Return is a list : next (x,y)
    def RulkovMap(alpha, beta ,sigma,x,y):
    	return alpha/(1+x**2) +y , y-sigma*x-beta 
    
    # Return is a list: the next tangent vector
    def RulkovMapTangent(alpha,beta,sigma,x,y,dx,dy):
    	return dy-2*alpha*x*dx/(1+x**2)**2 , dy-sigma*dx 
    
    # Simulation parameters
    #
    # Control parameters:
    #alpha = 
    beta = 0.0009
    sigma= 0.0011
    # The number of iterations to throw away
    nTransients = 4000
    # The number of iterations to generate
    nIterates = 1000
    
    # Initial condition for iterates to plot
    xtemp = 0.1
    ytemp = 0.3
    # Let's through away some number of transients, so that we're on an attractor
    for n in range(0,nTransients):
    	xtemp, ytemp = RulkovMap(alpha,beta,sigma,xtemp,ytemp)
    # Set up arrays of iterates (x_n,y_n) and set the initial condition
    x = [xtemp]
    y = [ytemp]
    
    # The main loop that generates iterates and stores them for plotting
    
    for n in range(0,nIterates):
    	# at each iteration calculate (x_n+1,y_n+1)
    	xtemp, ytemp = RulkovMap(alpha,beta,sigma,x[n],y[n])
    	# and append to lists x and y
    	x.append( xtemp )
    	y.append( ytemp )
    
    # Estimate the LCEs
    # The number of iterations to throw away
    nTransients = 200
    # The number of iterations to over which to estimate
    nIterates = 10000
    # Initial condition
    xState = 0.0
    yState = 0.0
    # Initial tangent vectors
    e1x = 1.0
    e1y = 0.0
    e2x = 0.0
    e2y = 1.0
    # Iterate away transients and let the tangent vectors align
    #    with the global stable and unstable manifolds
    for n in range(0,nTransients):
    	xState, yState = RulkovMap(alpha,beta,sigma,xState,yState)
      # Evolve tangent vector for maxLCE
    	e1x, e1y =RulkovMapTangent(alpha,beta,sigma,xState,yState,e1x,e1y)
      # Normalize the tangent vector's length
    	d = np.sqrt(e1x*e1x + e1y*e1y)
    	e1x = e1x / d
    	e1y = e1y / d
      # Evolve tangent vector for minLCE
    	e2x, e2y = RulkovMapTangent(alpha,beta,sigma,xState,yState,e2x,e2y)
      # Pull-back: Remove any e1 component from e2
    	dote1e2 = e1x * e2x + e1y * e2y
    	e2x = e2x - dote1e2 * e1x
    	e2y = e2y - dote1e2 * e1y
      # Normalize second tangent vector
    	d =np.sqrt(e2x*e2x + e2y*e2y)
    	e2x = e2x / d
    	e2y = e2y / d
    # Okay, now we're ready to begin the estimation
    # This is essentially the same as above, except we accumulate estimates
    # We have to set the min,max LCE estimates to zero, since they are sums
    maxLCE = 0.0
    minLCE = 0.0
    for n in range(0,nIterates):
      # Get next state
    	xState, yState = RulkovMap(alpha,beta,sigma,xState,yState)
      # Evolve tangent vector for maxLCE
    	e1x, e1y = RulkovMapTangent(alpha,beta,sigma,xState,yState,e1x,e1y)
      # Normalize the tangent vector's length
    	d = np.sqrt(e1x*e1x + e1y*e1y)
    	e1x = e1x / d
    	e1y = e1y / d
      # Accumulate the stretching factor (tangent vector's length)
    	maxLCE = maxLCE + np.log(d)
      # Evolve tangent vector for minLCE
    	e2x, e2y =  RulkovMapTangent(alpha,beta,sigma,xState,yState,e2x,e2y)
      # Pull-back: Remove any e1 component from e2
    	dote1e2 = e1x * e2x + e1y * e2y
    	e2x = e2x - dote1e2 * e1x
    	e2y = e2y - dote1e2 * e1y
      # Normalize second tangent vector
    	d = np.sqrt(e2x*e2x + e2y*e2y)
    	e2x = e2x / d
    	e2y = e2y / d
      # Accumulate the shrinking factor (tangent vector's length)
    	minLCE = minLCE + np.log(d)
    
    # Convert to per-iterate LCEs and to base-2 logs
    maxLCE = maxLCE / float(nIterates) / np.log(2.)
    minLCE = minLCE / float(nIterates) / np.log(2.)

    MaxLce.append(maxLCE)
    MinLce.append(minLCE)
    #f.write('{:.5f} {:10.5f} {:10.5f}\n'.format(alpha, maxLCE,minLCE))
#f.close()


fig,(ax1,ax2)=plt.subplots(2,1,sharex=True)
ax1.plot(alpharange,MaxLce,label="Maximum Lyapunov")
ax1.set_ylabel(r'$\lambda_1$')
ax1.legend()
ax1.grid(axis='x',alpha=0.5,linestyle='--')
ax2.plot(alpharange,MinLce,'r',label="Minimum Lyapunov")
ax2.set_ylabel(r'$\lambda_2$')
ax2.set_xlabel(r'$\alpha$')
ax2.legend(loc='lower right')
ax2.grid(axis='x',alpha=0.5,linestyle='--')

fig.suptitle("Lyapunov exponent for Rulkov System")
#fig.savefig("Rulkov_Lce.jpeg",dpi=250)











