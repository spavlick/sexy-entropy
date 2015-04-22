#distance calculations
#LOOCV
#k validation
#lambda and sigma choosing
#solving for alpha
#wh_x_re calculation
#array of different x_re, write for an array, more dimensions for 2d or 3d

#fit_model - x_de,x_nu, sigma, lambda, b
#cross validation method - input fold - call LOOCV or k val
#'under the hood' method - input sigma and lambda lists??
#setter methods

#cross validation using b?
#allow user to set x_ce, default to random

import numpy as np

class uLSIF_API():
    def __init__(self,x_de,x_nu):
        self.x_de=x_de
        self.x_nu=x_nu

        #make default values
        self.sigma_chosen=None
        self.lambda_chosen=None
        #do we need to keep b?

        #finding size of matrices
        (d,n_de)=x_de.shape
        (d_nu,n_nu)=x_nu.shape
        self.d=d
        self.n_de=n_de
        self.d_nu=d_nu
        self.n_nu=n_nu
        self.n_min=min(n_de,n_nu)

        rand_index=np.random.permutation(n_nu)
        #default b value?
        self.x_ce=x_ce=x_nu[:,rand_index[0:b]]

        #variables for kernels
        self.K_de=None
        self.K_nu=None

        self.score_cv=None

    def calculate_distances(self,x): #x_ce as input??, what to return?
        x2=np.sum(np.power(x,2),axis=0)
        x_ce2=np.sum(np.power(self.x_ce,2),axis=0)

        #reshaping arrays
        x2=np.reshape(x2,(1,len(x2)))
        x_ce2=np.reshape(x_ce2,(1,len(x_ce2)))

        (d,n)=x.shape

        dist1=np.tile(x_ce2.conj().transpose(),[1,n])
        dist2=np.tile(x2,[b,1])
        dist3=2*np.dot(self.x_ce.conj().transpose(),x)
        dist=np.subtract(np.add(dist1,dist2),dist3)
        return dist

    def k_validation(self, fold): #cv_index, cv_split...attributes? run 1 loop or full?
        for sigma_index,sigma in enumerate(sigma_list):
            for lambda_index,lamb in enumerate(lambda_list):
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

    def LOOCV(self):
        for sigma_index,sigma in enumerate(sigma_list):
            K_de2=K_de[:,0:n_min-1]
            K_nu2=K_nu[:,0:n_min-1]
            H=np.dot(K_de,K_de.conj().transpose()/shape(K_de,1))
            h=np.mean(K_nu,1) #axis??
            for lambda_index,lamb in enumerate(lambda_list):
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

    def mylinsolve(self, A,b):
        R=linalg.cholesky(A).transpose()
        y=linalg.lstsq(R.conj().transpose(),b)[0] #indices of lstsq??
        x=linalg.lstsq(R,y)[0]
        return x

    def parameter_fit(self):
        score_cv_tmp=np.amin(self.score_cv,axis=1)
        lambda_chosen_index=np.argmin(self.score_cv,axis=1) #this is a list
        score=np.amin(score_cv_tmp)
        sigma_chosen_index=np.argmin(score_cv_tmp)
        self.lambda_chosen=lambda_list[lambda_chosen_index[sigma_chosen_index]]
        self.sigma_chosen=sigma_list[sigma_chosen_index]

    def fit_model(self):

    def alpha_solve(self):
        K_de,K_nu=self.kernel_create()
        al1=np.add(np.dot(K_de,K_de.conj().transpose())/n_de,lambda_chosen*np.eye(b))
        al2=np.mean(K_nu,axis=1)
        al2=np.reshape(al2,(len(al2),1))
        alphat=mylinsolve(al1,al2)
        alphah=np.maximum(np.zeros(alphat.shape),alphat) #check maximum function

    def dist_solve(self,x_re,x_ce):
        d=x_re.shape[0]
        n_re=x_re.shape[1]
        x_re2=np.sum(np.power(x_re,2),axis=0)
        dist21=np.tile(x_ce2.conj().transpose(),[1,n_re])
        dist22=np.tile(x_re2,[b,1])
        dist23=2.*np.dot(x_ce.conj().transpose(),x_re)
        dist2_x_re=np.subtract(np.add(dist21,dist22),dist23)
        wh_x_re=np.dot(alphah.conj().transpose(),np.exp(-dist2_x_re/(2*sigma_chosen**2)))
        return wh_x_re

    def cross_validation(self,fold,sigma_list,lambda_list):
        self.score_cv=np.zeros((len(sigma_list),len(lambda_list)))
        if fold==0:
            self.LOOCV()
        else:
            self.k_validation()

    def kernel_create(self,sigma):

        K_de=np.exp(-dist2_x_de/(2.*sigma**2)) #creating kernels for tr
        K_nu=np.exp(-dist2_x_nu/(2.*sigma**2)) #kernels for te

    def set_lambda(self,lamb):
        self.lambda_chosen=lamb

    def set_sigma(self,sigma):
        self.sigma_chosen=sigma

    def set_x_ce(self,x_ce):
        self.x_ce=x_ce

    def set_K_nu(self,K_nu):
        self.K_nu=K_nu

    def set_K_de(self,K_de):
        self.K_de=K_de

    def get_score_cv(self):
        return self.score_cv

    def get_lambda(self):
        return lambda_chosen

    def get_sigma(self):
        return sigma_chosen


