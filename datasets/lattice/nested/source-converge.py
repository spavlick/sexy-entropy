import os
import h5py
import numpy
import math
import matplotlib.pyplot as plt
from ast import literal_eval

particle_nums = [1e3,1e4,1e5,1e6]
colors = ['b','m','g','r']

kl_list=[]
entropy_list=[]

for part_index,part_num in enumerate(particle_nums):
  #get batch numbers from file names
  directory = '/'+str(int(part_num))+'/'
  filenames = [f for f in os.listdir(directory) if '.h5' in f]
  batch_nums = [int(f[11:-3]) for f in filenames]

  #make empty lists to store entropies and kl divergences
  entropies = numpy.zeros(len(batch_nums))
  kl_divs = numpy.zeros(len(batch_nums)-1)


  mesh_cells = (4,4,1)
  mesh_size = (4.0,4.0,100000000000.0)
  mesh_center = (0,0,0)


  #calculate dimensions for a single cell
  (num_x,num_y,num_z) = mesh_cells
  (whole_x_width,whole_y_width,whole_z_width) = mesh_size
  x_width = whole_x_width/float(num_x)
  y_width = whole_y_width/float(num_y)
  z_width = whole_z_width/float(num_z)

  #find lower left of mesh
  (x_center,y_center,z_center)=mesh_center
  x_left = x_center-((num_x/2.0)*x_width)
  y_left = y_center-((num_y/2.0)*y_width)
  z_left = z_center-((num_z/2.0)*z_width)
  lower_left = (x_left,y_left,z_left)

  #create variables for mesh probabilities
  prev_probs = None
  cur_probs = None

  #loops through all files, calculating mesh probabilities, 
  #entropy, and KL divergence
  for index, filename in enumerate(filenames):
    print part_num, batch_nums[index]
    f = h5py.File(directory + filename,'r')
    positions = f['source_bank']['xyz']
    num_neutrons = int(f['n_particles'][0])

    '''
    #plotting the particle positions
    xvals = [e[0] for e in positions]
    yvals = [e[1] for e in positions]
    fig = plt.figure()
    plt.plot(xvals,yvals,'o')
    plt.title('Particle Positions in the XY Plane')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    fig.savefig('batch'+str(batch_nums[index])+'.png')
    '''

    #calculating mesh probabilities
    cur_probs=numpy.histogramdd(positions,bins=mesh_cells)[0]
    cur_probs[:,:,:] /= float(num_neutrons)

    #calculating the shannon entropies
    entropy=cur_probs[:,:,:]*numpy.log(cur_probs[:,:,:])
    entropy=numpy.nan_to_num(entropy)
    entropies[index] = -entropy.sum()

    #calculating KL Divergence
    if prev_probs != None:
      kl_div=cur_probs[:,:,:]*numpy.log(cur_probs[:,:,:]/prev_probs[:,:,:])
      kl_div=numpy.nan_to_num(kl_div)
      kl_divs[index-1]=kl_div.sum()

    prev_probs = cur_probs
    f.close()

  entropy_list.append(entropies)
  kl_list.append(kl_divs)

#plotting Shannon entropies
fig = plt.figure()
legend=[]
for i,part_num in enumerate(particle_nums):
  plt.plot(batch_nums,entropy_list[i],'o-',color=colors[i])
  legend.append(str(part_num)+'particles per batch')
plt.title('Shannon Entropy v Batch Number')
plt.xlabel('Batch Number')
plt.ylabel('Shannon Entropy')
plt.legend(legend)
plt.grid()
fig.savefig('shannon-entropies.png')

#plotting KL Divergence
fig = plt.figure()
kl_batches = numpy.copy(batch_nums)
kl_batches = numpy.delete(kl_batches,numpy.array([0]))
legend=[]
for i,part_num in enumerate(particle_nums):
  plt.plot(kl_batches,kl_list[i],'o-',color=colors[i])
  legend.append(str(part_num)+'particles per batch')
plt.title('KL Divergence v Batch Number')
plt.xlabel('Batch Number')
plt.ylabel('KL Divergence')
plt.legend(legend)
plt.grid()
fig.savefig('kl-divergences.png')
