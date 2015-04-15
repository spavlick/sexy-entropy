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

#allow user to set x_ce, default to random


class uLSIF_API():
    def __init__(self,x_de,x_nu):
        self.x_de=x_de
        self.x_nu=x_nu
        self.x_re=x_re #make input to separate method to make wh_x_re
        #make default values
        self.sigma_list=sigma_list
        self.lambda_list=lambda_list
        #do we need to keep b?
        #store best lambda and sigma

        #finding size of matrices
        (d,n_de)=x_de.shape
        (d_nu,n_nu)=x_nu.shape
        self.d=d
        self.n_de=n_de
        self.d_nu=d_nu
        self.n_nu=n_nu

        #variables for kernels
        self.K_de=None
        self.K_nu=None

        #how to have optional inputs?
