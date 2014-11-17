import numpy
import matplotlib.pyplot as plt
from pdfGaussian import pdfGaussian
from uLSIF import uLSIF

numpy.random.seed(0)

d=1

case1={}
case1['n_de']=100
case1['n_nu']=100
case1['mu_de']=1
case1['mu_nu']=1
case1['sigma_de']=.5
case1['sigma_nu']=1.0/8.0
case1['legend_position']=1

case2={}
case2['n_de']=200
case2['n_nu']=1000
case2['mu_de']=1
case2['mu_nu']=2
case2['sigma_de']=.5
case2['sigma_nu']=1.0/4.0
case2['legend_position']=2

#creating plot for functions
fig=plt.figure()

cases=[case1,case2]
for i in range(1):
  x_de=cases[i]['mu_de']+cases[i]['sigma_de']*numpy.random.randn(d, cases[i]['n_de'])
  x_nu=cases[i]['mu_nu']+cases[i]['sigma_nu']*numpy.random.randn(d,cases[i]['n_nu'])

  x_disp=numpy.linspace(-.5,3,100)
  x_disp.shape=(1,x_disp.size)
  p_de_x_disp=pdfGaussian(x_disp,cases[i]['mu_de'],cases[i]['sigma_de'])
  p_nu_x_disp=pdfGaussian(x_disp,cases[i]['mu_nu'],cases[i]['sigma_nu'])
  w_x_disp=numpy.divide(p_nu_x_disp,p_de_x_disp)

  p_de_x_de=pdfGaussian(x_de,cases[i]['mu_de'],cases[i]['sigma_de'])
  p_nu_x_de=pdfGaussian(x_de,cases[i]['mu_nu'],cases[i]['sigma_nu'])
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
