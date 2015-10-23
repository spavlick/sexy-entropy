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
  #x_ce=np.array([x_nu[:,rand_index[0:b/2]],x_de[:,rand_index[0:b/2]]])
  #x_ce=np.reshape(x_ce,(2,b))
  n_min=min(n_de,n_nu) #finds smaller or two distribution sample sizes

  #computing distances
  x_de2=np.sum(np.power(x_de,2),axis=0)
  x_nu2=np.sum(np.power(x_nu,2),axis=0)
  x_ce2=np.sum(np.power(x_ce,2),axis=0)

  #reshaping arrays
  x_de2=np.reshape(x_de2,(1,len(x_de2)))
  x_nu2=np.reshape(x_nu2,(1,len(x_nu2)))
  x_ce2=np.reshape(x_ce2,(1,len(x_ce2)))

  dist2_x_de1=np.tile(x_ce2.conj().transpose(),[1,n_de])
  dist2_x_de2=np.tile(x_de2,[b,1])
  dist2_x_de3=2*np.dot(x_ce.conj().transpose(),x_de)
  dist2_x_de=np.subtract(np.add(dist2_x_de1,dist2_x_de2),dist2_x_de3)

  dist2_x_nu1=np.tile(x_ce2.conj().transpose(),[1,n_nu])
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
      cv_split_nu=np.floor(np.arange(n_nu)*fold/n_nu)
      cv_index_de=np.random.permutation(n_de)
      cv_split_de=np.floor(np.arange(n_de)*fold/n_de)

      cv_index_nu=np.reshape(cv_index_nu,(1,len(cv_index_nu)))
      cv_split_nu=np.reshape(cv_split_nu,(1,len(cv_split_nu)))
      cv_index_de=np.reshape(cv_index_de,(1,len(cv_index_de)))
      cv_split_de=np.reshape(cv_split_de,(1,len(cv_split_de)))

    for sigma_index,sigma in enumerate(sigma_list):
      K_de=np.exp(-dist2_x_de/(2.*sigma**2)) #creating kernels for tr
      K_nu=np.exp(-dist2_x_nu/(2.*sigma**2)) #kernels for te
      if fold==0:
        K_de2=K_de[:,0:n_min-1]
        K_nu2=K_nu[:,0:n_min-1]
        H=np.dot(K_de,K_de.conj().transpose()/shape(K_de,1))
        h=np.mean(K_nu,1) #axis??

      #LOOCV
      for lambda_index,lamb in enumerate(lambda_list):
        if fold==0: #if fold==0 run LOOCV
          #look in Appendix I and formula 33
          C=np.add(H,lamb*np.dot((n_de-1)/n_de,np.eye(b))) #b hat
          invC=linalg.inv(C) #inverting C matrix
          beta=invC*h #finding beta values
          invCK_de=np.dot(invC,K_de2)
          tmp=np.subtract(n_de*np.ones((1,n_min)),np.add(np.dot(K_de,invCK_de),np.ones(np.dot(K_de,invCK_de).shape))) #denom in B0
          B0=np.add(np.dot(beta,np.ones(1,n_min)),np.divide(np.dot(np.dot(invCK_de,np.diag((beta.conj().transpose()),K_de2)),tmp)))
          B1=np.add(np.dot(invC,K_nu2),np.dot(invCK_de,np.diag(np.add(np.dot(K_nu2,invCK_de),np.ones(np.dot(K_nu2,invCK_de).shape)))))
          A=np.max(0,(n_de-1)/(n_de*(n_nu-1))*(n_nu*np.subtract(B0,B1))) #retrieve positive vals
          wh_x_de2=np.add(np.dot(K_de2,A),np.ones(np.dot(K_de2,A)).shape).conj().transpose()
          wh_x_nu2=np.add(np.dot(K_nu2,A),np.ones(np.dot(K_nu2,A)).shape).conj().transpose()
          score_cv[sigma_index,lambda_index]=np.mean(np.power(wh_x_de2,2))/2.-np.mean(wh_x_nu2)
        else: #if fold!=0 run k validation
          score_tmp=np.zeros((1,fold))

          for k in range(0,fold):
            Ktmp=K_de[:,cv_index_de[cv_split_de!=k]]
            al1=np.add(np.dot(Ktmp,Ktmp.conj().transpose())/Ktmp.shape[1],lamb*np.eye(b))
            al2=np.mean(K_nu[:,cv_index_nu[cv_split_nu!=k]],1)
            al2=np.reshape(al2,(len(al2),1))
            alphat_cv=mylinsolve(al1,al2)
            alphah_cv=np.maximum(alphat_cv,np.zeros(alphat_cv.shape))
            tmp1=np.dot(K_de[:,cv_index_de[cv_split_de==k]].conj().transpose(),alphah_cv)
            tmp2=np.mean(np.power(tmp1,2))/2.
            tmp3=np.dot(K_nu[:,cv_index_nu[cv_split_nu==k]].conj().transpose(),alphah_cv)
            tmp4=np.mean(tmp3)
            score_tmp[0,k]=tmp2-tmp4
            #score_tmp(k)=mean((K_de(:,cv_index_de(cv_split_de==k))'*alphah_cv).^2)/2 ...
                #-mean(K_nu(:,cv_index_nu(cv_split_nu==k))'*alphah_cv)

          score_cv[sigma_index,lambda_index]=np.mean(score_tmp)
       
    score_cv_tmp=np.amin(score_cv,axis=1)
    lambda_chosen_index=np.argmin(score_cv,axis=1) #this is a list
    score=np.amin(score_cv_tmp)
    sigma_chosen_index=np.argmin(score_cv_tmp)
    lambda_chosen=lambda_list[lambda_chosen_index[sigma_chosen_index]]
    sigma_chosen=sigma_list[sigma_chosen_index]

  #solving for alpha parameter matrix
  K_de=np.exp(-dist2_x_de/(2.*sigma_chosen**2))
  K_nu=np.exp(-dist2_x_nu/(2.*sigma_chosen**2))
  al1=np.add(np.dot(K_de,K_de.conj().transpose())/n_de,lambda_chosen*np.eye(b))
  al2=np.mean(K_nu,axis=1)
  al2=np.reshape(al2,(len(al2),1))
  alphat=mylinsolve(al1,al2)
  alphah=np.maximum(np.zeros(alphat.shape),alphat) #check maximum function
  wh_x_de=np.dot(alphah.conj().transpose(),K_de)
  wh_x_nu=np.dot(alphah.conj().transpose(),K_nu)


  if len(x_re)==0:
    wh_x_re=None
  else:
    d=x_re.shape[0]
    n_re=x_re.shape[1]
    x_re2=np.sum(np.power(x_re,2),axis=0)
    dist21=np.tile(x_ce2.conj().transpose(),[1,n_re])
    dist22=np.tile(x_re2,[b,1])
    dist23=2.*np.dot(x_ce.conj().transpose(),x_re)
    dist2_x_re=np.subtract(np.add(dist21,dist22),dist23)
    wh_x_re=np.dot(alphah.conj().transpose(),np.exp(-dist2_x_re/(2*sigma_chosen**2)))

  return wh_x_de, wh_x_nu,wh_x_re



