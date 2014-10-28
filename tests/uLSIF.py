import numpy
import scipy.linalg as linalg
from mylinsolve import mylinsolve

def uLSIF(x_de,x_nu,x_re=[],sigma_list=[],lambda_list=[],b=100,fold=0):
  #casting arrays to numpy ndarrays
  x_de = numpy.ndarray(x_de)
  x_nu=numpy.ndarray(x_nu)
  x_re=numpy.ndarray(x_re)
  sigma_list=numpy.ndarray(sigma_list)
  lambda_list=numpy.ndarray(lambda_list)

  #finding size of matrices
  (d,n_de)=x_de.shape
  (d_nu,n_nu)=x_nu.shape
  
  #checking for errors
  if d!=d_nu:
    error

  #choose Gaussian kernel center
  rand_index=numpy.random.permutation(n_nu) #list of numbers from 0 to n_nu-1 in random order
  b=min(b,n_nu) #number of Gaussian kernels
  x_ce=x_nu[:,rand_index[1:b]] #finds Gaussian kernel centers
  n_min=min(n_de,n_nu) #finds smaller or two distribution sample sizes

  #computing distances
  x_de2=numpy.sum(x_de**2,1) 
  x_nu2=numpy.sum(x_nu**2,1)
  x_ce2=numpy.sum(x_ce**2,1)
  dist2_x_de=numpy.tile(x_ce2.conj().transpose(),[1 n_de])+numpy.tile(x_de2,[b 1])-2*x_ce.conj().transpose().dot(x_de)
  dist2_x_nu=numpy.tile(x_ce.conj().transpose(),[1 n_nu])+numpy.tile(x_nu2,[b 1])-2*x_ce.conj().transpose().dot(x_nu)

  score_cv=numpy.zeros(len(sigma_list),len(lambda_list)) #cross validation scores

  #choosing lambda and sigma
  if len(sigma_list)==1 && len(lambda_list)==1:
    sigma_chosen=sigma_list[0]
    lambda_chosen=lambda_list[0]

    #fold tells how many times to run cross validation
    if fold!=0:
      cv_index_nu=numpy.random.permutation(n_nu)
      cv_split_nu=numpy.floor([0:n_nu-1]*fold/n_nu)+1
      cv_index_de=numpy.random.permutation(n_de)
      cv_split_de=numpy.floor([0:n_de-1]*fold/n_de)+1

    for sigma,sigma_index in enumerate(sigma_list):
      K_de=linalg.expm(-dist2_x_de/(2*sigma**2)) #creating kernels for tr
      K_nu=linalg.expm(-dist2_x_nu/(2*sigma**2)) #kernels for te
      if fold==0:
        K_de2=K_de[:,1:n_min]
        K_nu2=K_nu[:,1:n_min]
        H=K_de*K_de.conj().transpose()/shape(K_de,2)
        h=numpy.mean(K_nu,2)

      #LOOCV
      for lamb,lambda_index in enumerate(lambda_list):
        if fold==0: #if fold==0 run LOOCV
          #look in Appendix I and formula 33
          C=H+lamb*(n_de-1)/n_de*numpy.eye(b) #b hat
          invC=linalg.inv(C) #inverting C matrix
          beta=invC*h #finding beta values
          invCK_de=invC*K_de2
          tmp=n_de*numpy.ones(1,n_min)-numpy.sum(K_de*invCK_de,1) #denom in B0
          B0=beta*numpy.ones(1,n_min)+invCK_de*numpy.diag((beta.conj().transpose()*K_de2)/tmp)
          B1=invC*K_nu2+invCK_de*numpy.diag(numpy.sum(K_nu2*invCK_de,1))
          A=numpy.max(0,(n_de-1)/(n_de*(n_nu-1))*(n_nu*B0-B1)) #retrieve positive vals
          wh_x_de2=numpy.sum(K_de2*A,1).conj().transpose()
          wh_x_nu2=numpy.sum(K_nu2*A,1).conj().transpose()
          score_cv[sigma_index,lambda_index]=numpy.mean(wh_x_de2**2)/2-numpy.mean(wh_x_nu2)
        else: #if fold!=0 run k validation
          score_tmp=numpy.zeros(1,fold)

          for k in range(1,fold)
            Ktmp=K_de[:,cv_index_de[cv_split_de!=k]]
            alphat_cv=mylinsolve(Ktmp*Ktmp.conj().transpose()/shape(Ktmp,2)+lamb*numpy.eye(b),numpy.mean(K_nu[:,cv_index_nu[cv_split_nu!=k]],2))
       
    [score_cv_temp,lambda_chosen_index]=numpy.min(score_cv,[],2)
    [score,sigma_chosen_index]=numpy.min(score_cv_tmp)
    lambda_chosen=sigma_list[sigma_chosen_index]

  #solving for alpha parameter matrix
  K_de=linalg.expm(-dist2_x_de/(2*sigma_chosen**2))
  K_nu=linalg.expm(-dist2_x_nu/(2*sigma_chosen**2))
  alphat=mylinsolve(K_de.dot(K_de.conj().tranpose())/n_de+lambda_chosen*numpy.eye(b),numpy.mean(K_nu,2))
  alphah=numpy.max(0,alphat)
  wh_x_de=alphah.conj().transpose()*K_de


  #significance of x_re?? compare to wh_x_de?
  if x_re.isempty():
    wh_x_re=None
  else:
    [d,n_re]=x_re.shape()
    x_re2=numpy.sum(x_re**2,1)
    dist2_x_re=numpy.tile(x_ce.conj().transpose(),[1 n_re])+numpy.tile(x_re2,[b 1])-2*x_ce.conj.transpose.dot(x_re)
    wh_x_re=alphah.conj().transpose().dot(linalg.expm(-dist2_x_re/(2*sigma_chosen**2)))

  return wh_x_de,wh_x_re
