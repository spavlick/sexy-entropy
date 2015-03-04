import numpy
import math

def pdfGaussian(x,mu,sigma):
  (d,nx)=x.shape #assumes that x is 2d array
  tmp=numpy.divide(numpy.subtract(x,numpy.tile(mu,[1,nx])),numpy.tile(sigma,[1,nx]))/math.sqrt(2)
  px=(2*math.pi)**(-d/2.0)/numpy.prod(sigma)*numpy.power(math.e,-numpy.sum(tmp**2,1))
  return px
