import numpy as np
import matplotlib.pyplot as plt
from pdfGaussian import pdfGaussian
from uLSIF import uLSIF
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

np.random.seed(0)

a=1

if a==1:
  n_de=200
  n_nu=200
  mu_de=np.array([[-1],[-2]])
  mu_nu=np.array([[-2],[1]]) #backwards when plotted
  sigma_de=np.array([[.5],[.5]])
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
#x_de=np.add(mu_de*np.ones((d,n_de)),sigma_de*np.random.randn(d,n_de))
#x_nu=np.add(mu_nu*np.ones((d,n_nu)),sigma_nu*np.random.randn(d,n_nu))
x1=np.add(mu_de*np.ones((d,n_de)),sigma_de*np.random.randn(d,n_de))

x_de_rep=np.repeat(x1[0],len(x1[0]))
x_de_tile=np.tile(x1[1],len(x1[1]))
x_de=x_nu=np.vstack((x_de_rep,x_de_tile))

#x_de=np.loadtxt('x_de.csv',delimiter=',')
#x_de=np.reshape(x_de,(1,len(x_de)))
#x_nu=np.loadtxt('x_nu.csv',delimiter=',')
#x_nu=np.reshape(x_nu,(1,len(x_nu)))

#mesh grid
x_disp=y_disp=np.linspace(-2,2,100)
x_disp2=np.repeat(x_disp,len(y_disp))
y_disp2=np.tile(y_disp,len(y_disp))
xy_disp=np.vstack((x_disp2,y_disp2))
p_de_x_disp=pdfGaussian(xy_disp,mu_de,sigma_de)
p_nu_x_disp=pdfGaussian(xy_disp,mu_nu,sigma_nu)
w_x_disp=np.divide(p_nu_x_disp,p_de_x_disp)

p_de_x_de=pdfGaussian(x_de,mu_de,sigma_de)
p_nu_x_de=pdfGaussian(x_de,mu_nu,sigma_nu)
w_x_de=np.divide(p_nu_x_de,p_de_x_de)

wh_x_de,wh_x_nu,wh_x_disp=uLSIF(x_de,x_nu,xy_disp,fold=5,b=50)


fig=plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

X1,Y1=np.meshgrid(x_disp,y_disp)

wh_x_disp2=np.reshape(wh_x_disp,X1.shape)
ax.plot_surface(X1,Y1,wh_x_disp2,cmap=cm.coolwarm,)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

fig=plt.figure(2)
ax = fig.add_subplot(111, projection='3d')

X2,Y2=np.meshgrid(x_de[0,:],x_de[1,:])

wh_x_de2=np.reshape(wh_x_de,X2.shape)
ax.plot_surface(X2,Y2,wh_x_de2,cmap=cm.coolwarm,)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

fig=plt.figure(3)
ax = fig.add_subplot(111, projection='3d')

X3,Y3=np.meshgrid(x_nu[0,:],x_nu[1,:])

wh_x_nu2=np.reshape(wh_x_nu,X3.shape)
ax.plot_surface(X3,Y3,wh_x_nu2,cmap=cm.coolwarm,)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
