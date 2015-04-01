import numpy as np
import matplotlib.pyplot as plt
from pdfGaussian import pdfGaussian
from uLSIF import uLSIF

np.random.seed(0)

d=1

if d==1:
  n_de=100
  n_nu=100
  mu_de=1
  mu_nu=1
  sigma_de=.5
  sigma_nu=1.0/8.0
  legend_position=1

else:
  n_de=200
  n_nu=1000
  mu_de=1
  mu_nu=2
  sigma_de=.5
  sigma_nu=1.0/4.0
  legend_position=2

#creating plot for functions
fig=plt.figure()

x_de=np.add(mu_de*np.ones((d,n_de)),sigma_de*np.random.randn(d,n_de))
x_nu=np.add(mu_nu*np.ones((d,n_nu)),sigma_nu*np.random.randn(d,n_nu))

x_disp=np.linspace(-.5,3,100)
x_disp.shape=(1,x_disp.size)
p_de_x_disp=pdfGaussian(x_disp,mu_de,sigma_de)
p_nu_x_disp=pdfGaussian(x_disp,mu_nu,sigma_nu)
w_x_disp=np.divide(p_nu_x_disp,p_de_x_disp)

p_de_x_de=pdfGaussian(x_de,mu_de,sigma_de)
p_nu_x_de=pdfGaussian(x_de,mu_nu,sigma_nu)
w_x_de=np.divide(p_nu_x_de,p_de_x_de)

wh_x_de,wh_x_disp=uLSIF(x_de,x_nu,x_disp,fold=5)


plt.plot(x_de,wh_x_de)
plt.plot(wh_x_disp)
plt.plot(w_x_de)

plt.title('Probability distributions')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
fig.savefig('uLSIF demo')
