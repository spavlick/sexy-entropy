import numpy
import scipy.linalg as linalg

function uLSIF(x_de,x_nu,x_re=[],sigma_list=[],lambda_list=[],b=100,fold=0):
  #casting arrays to numpy ndarrays
  x_de = numpy.ndarray(x_de)
  x_nu=numpy.ndarray(x_nu)
  x_re=numpy.ndarray(x_re)
  sigma_list=numpy.ndarray(sigma_list)
  lambda_list=numpy.ndarray(lambda_list)

  #finding size of matrices
  (d,n_de)=x_de.shape()
  (d_nu,n_nu)=x_nu.shape()
  
  #checking for errors
  if d!=d_nu:
    error

  #choose Gaussian kernel center
  rand_index=numpy.random.permutation(n_nu)
  b=min(b,n_nu)
  x_ce=x_nu[:,rand_index[1:b]]
  n_min=min(n_de,n_nu)

  x_de2=sum(x_de**2,1)
  x_nu2=sum(x_nu**2,1)
  x_ce2=sum(x_ce**2,1)
  dist2_x_de=numpy.tile(x_ce2.conj().transpose(),[1 n_de])+numpy.tile(x_de2,[b 1])-2*x_ce.conj().transpose().dot(x_de)
  dist2_x_nu=numpy.tile(x_ce.conj().transpose(),[1 n_nu])+numpy.tile(x_nu2,[b 1])-2*x_ce.conj().transpose().dot(x_nu)

  score_cv=numpy.zeros(len(sigma_list),len(lambda_list))

  if len(sigma_list)==1 && len(lambda_list)==1:
    sigma_chosen=sigma_list
    lambda_chosen=lambda_list

  if fold!=0:
    cv_index_nu=numpy.random.permutation(n_nu)
    cv_split_nu=numpy.floor([0:n_nu-1]*fold/n_nu)+1
    cv_index_de=numpy.random.permutation(n_de)
    cv_split_de=numpy.floor([0:n_de-1]*fold/n_de)+1

  for sigma in sigma_list:
    K_de=linalg.expm(-dist2_x_de/(2*sigma**2))
    K_nu=linalg.expm(-dist2_x_nu/(2*sigma**2))
    if fold==0:
      K_de2=K_de[:,1:n_min]
      K_nu2=K_nu[:,1:n_min]
      H=K_de*K_de.conj().transpose()/shape(K_de,2)
      h=numpy.mean(K_nu,2)

    for lamb in lambda_list:
      if fold==0:
        C=H+lamb*(n_de-1)/n_de*numpy.eye(b)
        invC=linalg.inv(C)
        beta=invC*h
        invCK_de=invC*K_de2
        tmp=n_de*numpy.ones(1,n_min)-sum(K_de*invCK_de,1)
        B0=beta*numpy.ones(1,n_min)+invCK_de*numpy.diag((beta.conj().transpose()*K_de2)/tmp)
        B1=invC*K_nu2+invCK_de*numpy.diag(sum(K_nu2*invCK_de,1))
        A=max(0,(n_de-1)/(n_de*(n_nu-1))*(n_nu*B0-B1))
        wh_x_de2=sum(K_de2*A,1).conj().transpose()
