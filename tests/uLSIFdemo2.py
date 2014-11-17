import numpy
import matplotlib.pyplot as plt
from pdfGaussian import pdfGaussian
from uLSIF import uLSIF

numpy.random.seed(0)

d=1

n_de=numpy.array([[100],[200]])
n_nu=numpy.array([[100],[1000]])
mu_de=numpy.array([[1],[1]])
mu_nu=numpy.array([[1],[2]])
sigma_de=([[.5],[.5]])
sigma_nu=([[1.0/8.0],[1.0/4.0]])
legend_position=([[1],[2]])

#creating plot for functions
fig=plt.figure()

x_de=mu_de+sigma_de*numpy.random.randn(n_de)
x_nu=mu_nu+sigma_nu*numpy.random.randn(n_nu)

x_disp=numpy.linspace(-.5,3,100)
p_de_x_disp=pdfGaussian(x_disp,mu_de,sigma_de)
p_nu_x_disp=pdfGaussian(x_disp,mu_nu,sigma_nu)
w_x_disp=numpy.divide(p_nu_x_disp,p_de_x_disp)

p_de_x_de=pdfGaussian(x_de,mu_de,sigma_de)
p_nu_x_de=pdfGaussian(x_de,mu_nu,sigma_nu)
w_x_de=numpy.divide(p_nu_x_de,p_de_x_de)

wh_x_de,wh_x_disp=uLSIF(x_de,x_nu,x_disp,fold=5)

plt.plot(wh_x_de)
plt.plot(wh_x_disp)
plt.plot(w_x_de)

plt.title('Probability distributions')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
fig.savefig('uLSIF demo')

