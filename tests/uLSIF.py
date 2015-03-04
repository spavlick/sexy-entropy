import numpy as np
import scipy.linalg as linalg
from mylinsolve import mylinsolve

def uLSIF(x_de,x_nu,x_re=[],sigma_list=[],lambda_list=[],b=100,fold=0):

  #finding size of matrices
  (d,n_de)=x_de.shape
  (d_nu,n_nu)=x_nu.shape
  
  #checking for errors
  if sigma_list==[]:
    sigma_list=np.logspace(-3,1,num=9)
  
  if lambda_list==[]:
    lambda_list=np.logspace(-3,1,num=9)

  #choose Gaussian kernel center
  rand_index=np.random.permutation(n_nu) #list of numbers from 0 to n_nu-1 in random order
  b=min(b,n_nu) #number of Gaussian kernels
  x_ce=x_nu[:,rand_index[0:b]] #finds Gaussian kernel centers
  n_min=min(n_de,n_nu) #finds smaller or two distribution sample sizes

  #computing distances
  x_de2=np.add(np.power(x_de,2),np.ones(x_de.shape)) 
  x_nu2=np.add(np.power(x_nu,2),np.ones(x_nu.shape))
  x_ce2=np.add(np.power(x_ce,2),np.ones(x_ce.shape))
  
  dist2_x_de1=np.tile(x_ce2.conj().transpose(),[1,n_de])
  dist2_x_de2=np.tile(x_de2,[b,1])
  dist2_x_de3=2*np.dot(x_ce.conj().transpose(),x_de)
  dist2_x_de=np.subtract(np.add(dist2_x_de1,dist2_x_de2),dist2_x_de3)

  dist2_x_nu1=np.tile(x_ce.conj().transpose(),[1,n_nu])
  dist2_x_nu2=np.tile(x_nu2,[b,1])
  dist2_x_nu3=2*np.dot(x_ce.conj().transpose(),x_nu)
  dist2_x_nu=np.subtract(np.add(dist2_x_nu1,dist2_x_nu2),dist2_x_nu3)

  score_cv=np.zeros((len(sigma_list),len(lambda_list))) #cross validation scores
  lambda_chosen=None
  sigma_chosen=None

  #choosing lambda and sigma
  if len(sigma_list)==1 and len(lambda_list)==1:
    sigma_chosen=sigma_list[0]
    lambda_chosen=lambda_list[0]

  else:
    #fold tells how many times to run cross validation
    if fold!=0:
      cv_index_nu=np.random.permutation(n_nu)
      cv_split_nu=np.add(np.floor(np.arange(n_nu)*fold/n_nu),1)
      cv_index_de=np.random.permutation(n_de)
      cv_split_de=np.add(np.floor(np.arange(n_de)*fold/n_de),1)

    for sigma_index,sigma in enumerate(sigma_list):
      K_de=linalg.expm(-dist2_x_de/(2*sigma**2)) #creating kernels for tr
      K_nu=linalg.expm(-dist2_x_nu/(2*sigma**2)) #kernels for te
      if fold==0:
        K_de2=K_de[:,0:n_min-1]
        K_nu2=K_nu[:,0:n_min-1]
        H=K_de.dot(K_de.conj().transpose()/shape(K_de,1))
        h=np.mean(K_nu,1) #axis??

      #LOOCV
      for lambda_index,lamb in enumerate(lambda_list):
        if fold==0: #if fold==0 run LOOCV
          #look in Appendix I and formula 33
          C=np.add(H,lamb*np.dot((n_de-1)/n_de,np.eye(b))) #b hat
          invC=linalg.inv(C) #inverting C matrix
          beta=invC*h #finding beta values
          invCK_de=np.dot(invC,K_de2)
          tmp=np.subtract(n_de*np.ones(1,n_min),np.sum(K_de.dot(invCK_de),1)) #denom in B0
          B0=np.add(np.dot(beta,np.ones(1,n_min)),np.divide(np.dot(np.dot(invCK_de,np.diag((beta.conj().transpose()),K_de2)),tmp)))
          B1=np.add(np.dot(invC,K_nu2),np.dot(invCK_de,np.diag(np.sum(K_nu2*invCK_de,1))))
          A=np.max(0,(n_de-1)/(n_de*(n_nu-1))*(n_nu*np.subtract(B0,B1))) #retrieve positive vals
          wh_x_de2=np.sum(K_de2*A,1).conj().transpose()
          wh_x_nu2=np.sum(K_nu2*A,1).conj().transpose()
          score_cv[sigma_index,lambda_index]=np.mean(wh_x_de2**2)/2-np.mean(wh_x_nu2)
        else: #if fold!=0 run k validation
          score_tmp=np.zeros((1,fold))

          for k in range(1,fold):
            Ktmp=K_de[:,cv_index_de[cv_split_de!=k]]
            alphat_cv=mylinsolve(np.add(np.dot(Ktmp,Ktmp.conj().transpose())/Ktmp.shape[1],lamb*np.eye(b)),np.mean(K_nu[:,cv_index_nu[cv_split_nu!=k]],1))

          score_cv[sigma_index,lambda_index]=np.mean(score_tmp)
       
    lambda_chosen_index=np.amin(score_cv,axis=1)
    sigma_chosen_index=np.amin(score_cv[lambda_chosen_index,:])
    score=score_cv[sigma_chosen_index,lambda_chosen_index]
    sigma_chosen=sigma_list[sigma_chosen_index]
    lambda_chosen=lambda_list[lambda_chosen_index]

  #solving for alpha parameter matrix
  K_de=linalg.expm(-dist2_x_de/(2*sigma_chosen**2))
  K_nu=linalg.expm(-dist2_x_nu/(2*sigma_chosen**2))
  alphat=mylinsolve(np.add(np.dot(K_de,K_de.conj().T)/n_de,lambda_chosen*np.eye(b)),np.mean(K_nu,axis=1))
  alphah=np.maximum(np.zeros(alphat.shape),alphat) #check maximum function
  wh_x_de=np.dot(alphah.conj().transpose(),K_de)


  #finish syntax
  if x_re.size==0:
    wh_x_re=None
  else:
    [d,n_re]=x_re.shape
    x_re2=np.sum(x_re**2,1)
    dist2_x_re=np.tile(x_ce.conj().transpose(),[1,n_re])+np.tile(x_re2,[b,1])-2*x_ce.conj().transpose().dot(x_re)
    wh_x_re=alphah.conj().transpose().dot(linalg.expm(-dist2_x_re/(2*sigma_chosen**2)))

  return wh_x_de,wh_x_re
