#!/usr/bin/env python
# coding: utf-8

# In[43]:


import nengo
import nengo_spa as spa
import matplotlib.pyplot as plt
# with and without seed
import numpy as np
import random
import pickle

d = 128  # the dimensionality of the vectors

numbers_array_0 = pickle.load(open("Numbers_fb02", "rb"))
numbers_array_1 = pickle.load(open("Numbers_fb05", "rb"))
numbers_array_2 = pickle.load(open("Numbers_fb99", "rb"))
x_range = np.linspace(0,8,8000)

plt.figure(figsize = (6, 6))
xmin, xmax = 1, 1.2
ymin, ymax = -0.27, 1.2

plt.plot(x_range,numbers_array_0[:,0],'b-', label = 'blue fb = 0.2' )
plt.plot(x_range,numbers_array_1[:,0],'b--',  label = 'blue fb = 0.5' )
plt.plot(x_range,numbers_array_2[:,0],'b:', label = 'blue fb = 0.99'  )
plt.plot(x_range,numbers_array_0[:,1],'r-', label =  'red fb = 0.2'  )
plt.plot(x_range,numbers_array_1[:,1],'r--', label =  'red fb = 0.5'   )
plt.plot(x_range,numbers_array_2[:,1],'r:', label =  'red fb = 0.99'   )
plt.plot(x_range,numbers_array_0[:,2],'g-',  label =  'green fb = 0.2'   )
plt.plot(x_range,numbers_array_1[:,2],'g--', label =  'green fb = 0.5'   )
plt.plot(x_range,numbers_array_2[:,2],'g:', label =  'green fb = 0.99'   )


plt.gca().xaxis.set_major_locator(plt.NullLocator())    # distable tick
plt.gca().yaxis.set_major_locator(plt.NullLocator())

#plt.legend(loc = 'best') #['blue fb = 0.2' , 'blue fb = 0.5', 'blue fb = 0.99', 'red fb = 0.2', 'red fb = 0.5', 'red fb = 0.99'],  loc='best')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.errorbar( 1.145, -0.2, xerr=0.05, color='k', capsize=5)
plt.text( 1.15, -0.14, '0.1 sec',  horizontalalignment='center', verticalalignment='top')

plt.title("B: Effect of Feedback on RT")
plt.xlabel("Time [s]")
plt.ylabel("Similarity to inputs")
plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)
plt.savefig('horiz_Fig5a_JL_30'+ '.svg')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




