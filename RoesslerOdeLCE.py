
"""
Created on Wed Aug 14 16:01:28 2019

@author: banerjee
"""
#from __future__import division
from __future__ import division
from pylab import *
from numpy import sqrt,log
import numpy as np
from tqdm import tqdm



LCEa = []
LCEb = []
LCEc = []

Rrange=np.arange(0,1.4,0.01)
for R in tqdm(Rrange):    
    # The Lorenz 3D ODEs
    #	Original parameter values: (a,b,c) = (0.1,0.1,14)
    def RossXDot(a,b,c,x,y,z):
    	return -y-z
    
    def RossYDot(a,b,c,x,y,z):
    	return x+a*y
    
    def RossZDot(a,b,c,x,y,z):
    	return b+z*(x-c)
    
    # The tangent space (linearized) flow (aka co-tangent flow)
    def RossDXDot(a,b,c,x,y,z,dx,dy,dz):
    	return  -dy-dz
    
    def RossDYDot(a,b,c,x,y,z,dx,dy,dz):
    	return dx+a*dy
    
    def RossDZDot(a,b,c,x,y,z,dx,dy,dz):
    	return x*dz+z*dx-c*dz
    
    
    # 3D fourth-order Runge-Kutta integrator
    def RKThreeD(a,b,c,x,y,z,f,g,h,dt):
    	k1x = dt * f(a,b,c,x,y,z)
    	k1y = dt * g(a,b,c,x,y,z)
    	k1z = dt * h(a,b,c,x,y,z)
    	k2x = dt * f(a,b,c,x + k1x / 2.0,y + k1y / 2.0,z + k1z / 2.0)
    	k2y = dt * g(a,b,c,x + k1x / 2.0,y + k1y / 2.0,z + k1z / 2.0)
    	k2z = dt * h(a,b,c,x + k1x / 2.0,y + k1y / 2.0,z + k1z / 2.0)
    	k3x = dt * f(a,b,c,x + k2x / 2.0,y + k2y / 2.0,z + k2z / 2.0)
    	k3y = dt * g(a,b,c,x + k2x / 2.0,y + k2y / 2.0,z + k2z / 2.0)
    	k3z = dt * h(a,b,c,x + k2x / 2.0,y + k2y / 2.0,z + k2z / 2.0)
    	k4x = dt * f(a,b,c,x + k3x,y + k3y,z + k3z)
    	k4y = dt * g(a,b,c,x + k3x,y + k3y,z + k3z)
    	k4z = dt * h(a,b,c,x + k3x,y + k3y,z + k3z)
    	x += ( k1x + 2.0 * k2x + 2.0 * k3x + k4x ) / 6.0
    	y += ( k1y + 2.0 * k2y + 2.0 * k3y + k4y ) / 6.0
    	z += ( k1z + 2.0 * k2z + 2.0 * k3z + k4z ) / 6.0
    	return x,y,z
    
    # Tanget space flow (using fourth-order Runge-Kutta integrator)
    def TangentFlowRKThreeD(a,b,c,x,y,z,df,dg,dh,dx,dy,dz,dt):
    	k1x = dt * df(a,b,c,x,y,z,dx,dy,dz)
    	k1y = dt * dg(a,b,c,x,y,z,dx,dy,dz)
    	k1z = dt * dh(a,b,c,x,y,z,dx,dy,dz)
    	k2x = dt * df(a,b,c,x,y,z,dx+k1x/2.0,dy+k1y/2.0,dz+k1z/2.0)
    	k2y = dt * dg(a,b,c,x,y,z,dx+k1x/2.0,dy+k1y/2.0,dz+k1z/2.0)
    	k2z = dt * dh(a,b,c,x,y,z,dx+k1x/2.0,dy+k1y/2.0,dz+k1z/2.0)
    	k3x = dt * df(a,b,c,x,y,z,dx+k2x/2.0,dy+k2y/2.0,dz+k2z/2.0)
    	k3y = dt * dg(a,b,c,x,y,z,dx+k2x/2.0,dy+k2y/2.0,dz+k2z/2.0)
    	k3z = dt * dh(a,b,c,x,y,z,dx+k2x/2.0,dy+k2y/2.0,dz+k2z/2.0)
    	k4x = dt * df(a,b,c,x,y,z,dx+k3x,dy+k3y,dz+k3z)
    	k4y = dt * dg(a,b,c,x,y,z,dx+k3x,dy+k3y,dz+k3z)
    	k4z = dt * dh(a,b,c,x,y,z,dx+k3x,dy+k3y,dz+k3z)
    	dx += ( k1x + 2.0 * k2x + 2.0 * k3x + k4x ) / 6.0
    	dy += ( k1y + 2.0 * k2y + 2.0 * k3y + k4y ) / 6.0
    	dz += ( k1z + 2.0 * k2z + 2.0 * k3z + k4z ) / 6.0
    	return dx,dy,dz
    
    # Simulation parameters
    # Integration time step
    dt = 0.01
    #
    # Control parameters for the Lorenz ODEs:
    sigma = 0.1  #a
    #R	  = 0.1  #b
    b	  = 14   # c
    # The number of iterations to throw away
    nTransients = 10
    # The number of time steps to integrate over
    nIterates = 1000
    
    # The main loop that generates the orbit, storing the states
    xState = 5.0
    yState = 5.0
    zState = 5.0
    # Iterate for some number of transients, but don't use these states
    for n in range(0,nTransients):
    	xState,yState,zState = RKThreeD(sigma,R,b,xState,yState,zState,RossXDot,RossYDot,RossZDot,dt)
    # Set up array of iterates and store the current state
    x = [xState]
    y = [yState]
    z = [zState]
    for n in range(0,nIterates):
    	# at each time step calculate new (x,y,z)(t)
    	xt,yt,zt = RKThreeD(sigma,R,b,x[n],y[n],z[n],RossXDot,RossYDot,RossZDot,dt)
    	# and append to lists
    	x.append(xt)
    	y.append(yt)
    	z.append(zt)
    
    # Estimate the LCEs
    # The number of iterations to throw away
    nTransients = 100
    # The number of iterations to over which to estimate
    #  This is really the number of pull-backs
    nIterates = 1000
    # The number of iterations per pull-back
    nItsPerPB = 10
    # Initial condition
    xState = 5.0
    yState = 5.0 
    zState = 5.0 
    # Initial tangent vectors
    e1x = 1.0
    e1y = 0.0
    e1z = 0.0
    e2x = 0.0
    e2y = 1.0
    e2z = 0.0
    e3x = 0.0
    e3y = 0.0
    e3z = 1.0
    # Iterate away transients and let the tangent vectors align
    #	with the global stable and unstable manifolds
    for n in range(0,nTransients):
    	for i in range(nItsPerPB):
    		xState,yState,zState = RKThreeD(sigma,R,b,xState,yState,zState,\
    			RossXDot,RossYDot,RossZDot,dt)
    		# Evolve tangent vector for maximum LCE (LCE1)
    		e1x,e1y,e1z = TangentFlowRKThreeD(sigma,R,b,xState,yState,zState,\
    			RossDXDot,RossDYDot,RossDZDot,e1x,e1y,e1z,dt)
    		# Evolve tangent vector for next LCE (LCE2)
    		e2x,e2y,e2z = TangentFlowRKThreeD(sigma,R,b,xState,yState,zState,\
    			RossDXDot,RossDYDot,RossDZDot,e2x,e2y,e2z,dt)
    		# Evolve tangent vector for last LCE
    		e3x,e3y,e3z = TangentFlowRKThreeD(sigma,R,b,xState,yState,zState,\
    			RossDXDot,RossDYDot,RossDZDot,e3x,e3y,e3z,dt)
    	# Normalize the tangent vector
    	d = sqrt(e1x*e1x + e1y*e1y + e1z*e1z)
    	e1x /= d
    	e1y /= d
    	e1z /= d
    	# Pull-back: Remove any e1 component from e2
    	dote1e2 = e1x * e2x + e1y * e2y + e1z * e2z
    	e2x -= dote1e2 * e1x
    	e2y -= dote1e2 * e1y
    	e2z -= dote1e2 * e1z
    	# Normalize second tangent vector
    	d = sqrt(e2x*e2x + e2y*e2y + e2z*e2z)
    	e2x /= d
    	e2y /= d
    	e2z /= d
    	# Pull-back: Remove any e1 and e2 components from e3
    	dote1e3 = e1x * e3x + e1y * e3y + e1z * e3z
    	dote2e3 = e2x * e3x + e2y * e3y + e2z * e3z
    	e3x -= dote1e3 * e1x + dote2e3 * e2x
    	e3y -= dote1e3 * e1y + dote2e3 * e2y
    	e3z -= dote1e3 * e1z + dote2e3 * e2z
    	# Normalize third tangent vector
    	d = sqrt(e3x*e3x + e3y*e3y + e3z*e3z)
    	e3x /= d
    	e3y /= d
    	e3z /= d
    
    # Okay, now we're ready to begin the estimation
    
    LCE1=0.0
    LCE2=0.0
    LCE3=0.0
    
    for n in range(0,nIterates):
    	for i in range(nItsPerPB):
    		xState,yState,zState = RKThreeD(sigma,R,b,xState,yState,zState,\
    			RossXDot,RossYDot,RossZDot,dt)
    		# Evolve tangent vector for maximum LCE (LCE1)
    		e1x,e1y,e1z = TangentFlowRKThreeD(sigma,R,b,xState,yState,zState,\
    			RossDXDot,RossDYDot,RossDZDot,e1x,e1y,e1z,dt)
    		# Evolve tangent vector for next LCE (LCE2)
    		e2x,e2y,e2z = TangentFlowRKThreeD(sigma,R,b,xState,yState,zState,\
    			RossDXDot,RossDYDot,RossDZDot,e2x,e2y,e2z,dt)
    		# Evolve tangent vector for last LCE
    		e3x,e3y,e3z = TangentFlowRKThreeD(sigma,R,b,xState,yState,zState,\
    			RossDXDot,RossDYDot,RossDZDot,e3x,e3y,e3z,dt)
    	# Normalize the tangent vector
    	d = sqrt(e1x*e1x + e1y*e1y + e1z*e1z)
    	e1x /= d
    	e1y /= d
    	e1z /= d
    	# Accumulate the first tangent vector's length change factor
    	LCE1+=log(d)
    
    	# Pull-back: Remove any e1 component from e2
    	dote1e2 = e1x * e2x + e1y * e2y + e1z * e2z
    	e2x -= dote1e2 * e1x
    	e2y -= dote1e2 * e1y
    	e2z -= dote1e2 * e1z
    	# Normalize second tangent vector
    	d = sqrt(e2x*e2x + e2y*e2y + e2z*e2z)
    	e2x /= d
    	e2y /= d
    	e2z /= d
    	# Accumulate the second tangent vector's length change factor
    	LCE2 += log(d)
    	# Pull-back: Remove any e1 and e2 components from e3
    	dote1e3 = e1x * e3x + e1y * e3y + e1z * e3z
    	dote2e3 = e2x * e3x + e2y * e3y + e2z * e3z
    	e3x -= dote1e3 * e1x + dote2e3 * e2x
    	e3y -= dote1e3 * e1y + dote2e3 * e2y
    	e3z -= dote1e3 * e1z + dote2e3 * e2z
    	# Normalize third tangent vector
    	d = sqrt(e3x*e3x + e3y*e3y + e3z*e3z)
    	e3x /= d
    	e3y /= d
    	e3z /= d
    	# Accumulate the third tangent vector's length change factor
    
    	LCE3 += log(d)
    
    
    # Convert to per-iterate, per-second LCEs and to base-2 logs
    IntegrationTime = dt * float(nItsPerPB) * float(nIterates)
    LCE1 =LCE1 / IntegrationTime
    LCE2 =(LCE2 / IntegrationTime).tolist()
    LCE3 =LCE3 / IntegrationTime
    
    #LCEa+=LCE1
    LCEb.append(LCE2)
    #LCEc+=LCE3
    
    
plot(Rrange,LCEb)
