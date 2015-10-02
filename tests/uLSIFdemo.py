import numpy as np
import matplotlib.pyplot as plt
from pdfGaussian import pdfGaussian
from uLSIF import uLSIF
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)

a=1

if a==1:
  n_de=2000
  n_nu=2000
  mu_de=np.array([[1.5],[1.5]])
  mu_nu=np.array([[1],[1]])
  sigma_de=np.array([[1],[1]])
  sigma_nu=np.array([[.5],[.5]])
  legend_position=1

else:
  n_nu=200
  n_de=1000
  mu_nu=1
  mu_de=2
  sigma_nu=.5
  sigma_de=1.0/4.0
  legend_position=2

d=2
x_de=np.add(mu_de*np.ones((d,n_de)),sigma_de*np.random.randn(d,n_de))
x_nu=np.add(mu_nu*np.ones((d,n_nu)),sigma_nu*np.random.randn(d,n_nu))

#x_de=np.loadtxt('x_de.csv',delimiter=',')
#x_de=np.reshape(x_de,(1,len(x_de)))
#x_nu=np.loadtxt('x_nu.csv',delimiter=',')
#x_nu=np.reshape(x_nu,(1,len(x_nu)))

#mesh grid
x_disp=y_disp=np.linspace(-.5,3,50)
#X,Y=np.meshgrid(x_disp,y_disp)
x_disp2=np.repeat(x_disp,len(y_disp))
y_disp2=np.tile(y_disp,len(y_disp))
xy_disp=np.vstack((x_disp2,y_disp2))
p_de_x_disp=pdfGaussian(xy_disp,mu_de,sigma_de)
p_nu_x_disp=pdfGaussian(xy_disp,mu_nu,sigma_nu)
w_x_disp=np.divide(p_nu_x_disp,p_de_x_disp)

p_de_x_de=pdfGaussian(x_de,mu_de,sigma_de)
p_nu_x_de=pdfGaussian(x_de,mu_nu,sigma_nu)
w_x_de=np.divide(p_nu_x_de,p_de_x_de)

wh_x_de,wh_x_nu,wh_x_disp=uLSIF(x_de,x_nu,xy_disp,fold=5,b=100)


fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')

#ax = Axes3D(fig)
x1=np.tile(x_disp[np.newaxis,:],[50, 1])
x2=np.tile(x_disp,[50, 1])

wh_x_disp2=np.reshape(wh_x_disp,x1.shape)
ax.plot_surface(x1,x2,wh_x_disp2,color='b')
plt.show()

#plt.plot(x_disp.flatten(),p_de_x_disp.flatten(),'b-',linewidth=2)
#plt.plot(x_disp.flatten(),p_nu_x_disp.flatten(),'k-',linewidth=2)
#plt.legend(['p_{de}(x)','p_{nu}(x)'])
#plt.xlabel('x')

#plt.grid()
#plt.show()

#fig=plt.figure()

#plt.plot(x_disp.flatten(),w_x_disp.flatten(),'r-',linewidth=2)
#plt.plot(x_disp.flatten(),wh_x_disp.flatten(),'g-',linewidth=2)
#plt.plot(x_de.flatten(),wh_x_de.flatten(),'bo',linewidth=1,markersize=8)
#plt.plot(x_nu.flatten(),wh_x_nu.flatten(),'mo',linewidth=1,markersize=8)


#plt.legend(['w(x)','w-hat(x)','w-hat(x^{de})'])
#plt.xlabel('x')

#plt.grid()
#plt.show()
