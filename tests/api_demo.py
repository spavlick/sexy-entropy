import numpy as np
from uLSIF_api import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

n_de=250
n_nu=250
mu_de=1
mu_nu=1
sigma_de=.5
sigma_nu=3
d=2

x_de=np.add(mu_de*np.ones((d,n_de)),sigma_de*np.random.randn(d,n_de))
x_nu=np.add(mu_nu*np.ones((d,n_nu)),sigma_nu*np.random.randn(d,n_nu))

obj=uLSIF_API(x_de,x_nu)
obj.full_model(np.logspace(-3,1,num=9),np.logspace(-3,1,num=9),5)

fig=plt.figure()
ax = fig.gca(projection='3d')

X = np.arange(min(x_de[0,:]), max(x_de[0,:]), 0.25)
Y = np.arange(min(x_de[1,:]), max(x_de[1,:]), 0.25)
X, Y = np.meshgrid(X,Y)

wh_x_de=np.dot(obj.alphah.conj().transpose(),obj.K_de)
ax.plot_surface(X,Y,wh_x_de.flatten(),rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
#plt.plot(x_nu[0,:].flatten(),obj.wh_x_nu.flatten(),'mo',linewidth=1,markersize=8)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

#plt.legend(['w(x)','w-hat(x)','w-hat(x^{de})'])
#plt.xlabel('x')

#plt.grid()
plt.show()

#add plot function to api
#add wh variables to api
#change getters and setters


