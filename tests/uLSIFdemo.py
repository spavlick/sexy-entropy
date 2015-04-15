import numpy as np
import matplotlib.pyplot as plt
from pdfGaussian import pdfGaussian
from uLSIF import uLSIF

np.random.seed(0)

a=2

if a==1:
  n_de=2000
  n_nu=2000
  mu_de=1
  mu_nu=1
  sigma_de=.5
  sigma_nu=3
  legend_position=1

else:
  n_nu=200
  n_de=1000
  mu_nu=1
  mu_de=2
  sigma_nu=.5
  sigma_de=1.0/4.0
  legend_position=2

d=1
x_de=np.add(mu_de*np.ones((d,n_de)),sigma_de*np.random.randn(d,n_de))
x_nu=np.add(mu_nu*np.ones((d,n_nu)),sigma_nu*np.random.randn(d,n_nu))

fig=plt.figure()
plt.hist(x_de.flatten(),20)
plt.show()
fig=plt.figure()
plt.hist(x_nu.flatten(),20)
plt.show()

#x_de=np.loadtxt('x_de.csv',delimiter=',')
#x_de=np.reshape(x_de,(1,len(x_de)))
#x_nu=np.loadtxt('x_nu.csv',delimiter=',')
#x_nu=np.reshape(x_nu,(1,len(x_nu)))

x_disp=np.linspace(-.5,3,100)
x_disp=np.reshape(x_disp,(1,len(x_disp)))
p_de_x_disp=pdfGaussian(x_disp,mu_de,sigma_de)
p_nu_x_disp=pdfGaussian(x_disp,mu_nu,sigma_nu)
w_x_disp=np.divide(p_nu_x_disp,p_de_x_disp)

p_de_x_de=pdfGaussian(x_de,mu_de,sigma_de)
p_nu_x_de=pdfGaussian(x_de,mu_nu,sigma_nu)
w_x_de=np.divide(p_nu_x_de,p_de_x_de)

wh_x_de,wh_x_nu,wh_x_disp=uLSIF(x_de,x_nu,x_disp,fold=5,b=200)

fig=plt.figure()


plt.plot(x_disp.flatten(),p_de_x_disp.flatten(),'b-',linewidth=2)
plt.plot(x_disp.flatten(),p_nu_x_disp.flatten(),'k-',linewidth=2)
plt.legend(['p_{de}(x)','p_{nu}(x)'])
plt.xlabel('x')

plt.grid()
plt.show()

fig=plt.figure()

#plt.plot(x_disp.flatten(),w_x_disp.flatten(),'r-',linewidth=2)
plt.plot(x_disp.flatten(),wh_x_disp.flatten(),'g-',linewidth=2)
plt.plot(x_de.flatten(),wh_x_de.flatten(),'bo',linewidth=1,markersize=8)
plt.plot(x_nu.flatten(),wh_x_nu.flatten(),'mo',linewidth=1,markersize=8)


plt.legend(['w(x)','w-hat(x)','w-hat(x^{de})'])
plt.xlabel('x')

plt.grid()
plt.show()
