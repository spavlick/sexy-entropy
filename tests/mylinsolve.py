from scipy import linalg
def mylinsolve():
  R=linalg.cholesky(A).T
  x=R.lstsqr(R.conj().transpose().lstsqr(b))
