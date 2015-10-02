import numpy
import math


def pdfGaussian(x,mu,sigma):
  (d,nx)=x.shape #assumes that x is 2d array
  tmp1= numpy.tile(mu,[1,nx])
  tmp2= numpy.tile(sigma,[1,nx])
  sub1=numpy.subtract(x,tmp1)
  tmp3=numpy.divide(sub1,tmp2)/math.sqrt(2)
  px1=(2*math.pi)**(-d/2.0)/numpy.prod(sigma) #sigma value to use??
  px=px1*numpy.exp(-numpy.sum(tmp3**2,axis=0)) #why add ones?
  px=numpy.reshape(px,(1,len(px)))
  return px
