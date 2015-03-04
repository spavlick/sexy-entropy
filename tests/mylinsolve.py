import numpy
import numpy.linalg as linalg
def mylinsolve(A,b):
  R=linalg.cholesky(A).transpose()
  y=linalg.lstsq(R.conj().transpose(),b) #indices of lstsq??
  x=linalg.lstsq(R,y)
  return x
