#!/usr/bin/env python
# coding: utf-8

# In[1]:


# @gyleodhis=====gyleodhis@outlook.com=====
import numpy as np
# Generate an array of 10 random integers less than 100 
rand = np.random.RandomState(42)
x = rand.randint(100, size=10)
x


# Accesiing four different elemets in a array

# In[2]:


k=[x[2], x[5], x[9], x[0]]
k
#We can also do it this way
z=[2,5,9,1]
x[z]


# Fancy indexing also works in multiple dimensions. Consider the following array:

# In[3]:


x = np.arange(12).reshape((3,4))
x


# Like with standard indexing, the first index refers to the row, and the second to the
# column:

# In[4]:


row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
x[row, col]


# Selecting Random Points.

# In[5]:


mean = [0,0]
cov = [[1,2], [2,5]]
x = rand.multivariate_normal(mean, cov, 100)
x.shape


# we can visualize these points as a scatter plot

# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
#import seaborn; seaborn.set() #for plot styling
plt.scatter(x[:,0], x[:,1]);


# Let’s use fancy indexing to select 20 random points. We’ll do this by first choosing 20
# random indices with no repeats, and use these indices to select a portion of the origi‐
# nal array:

# In[7]:


indices = np.random.choice(x.shape[0], 20, replace=False)
indices


# In[8]:


selection = x[indices] # Fancy indexing here
selection.shape


# Now to see which points were selected, let’s over-plot large circles at the locations of
# the selected points 

# In[9]:


plt.scatter(x[:, 0], x[:, 1], alpha=0.3)
plt.scatter(selection[:, 0], selection[:, 1],
            facecolor='none', s=200);


# This sort of strategy is often used to quickly partition datasets, as is often needed in
# train/test splitting for validation of statistical models

# Modifying Values with Fancy Indexing. Just as fancy indexing can be used to access parts of an array, it can also be used to
# modify parts of an array

# In[10]:


x = np.arange(10)
i = np.array([2, 1, 8, 4])
x[i] = 99
x


# In[11]:


i


# We can use any assignment-type operator for this. For example:

# In[12]:


x[i] +=20
x


# imagine we have 1,000 values and would like to quickly find where they fall within an array of bins. We could compute it using ufunc.at like this:

# In[13]:


np.random.seed(42)
x=np.random.randn(100)
#compute a histogram by hand
bins = np.linspace(-5,5,20)
counts= np.zeros_like(bins)
# find the appropriate bin for each x
i=np.searchsorted(bins,x)
# add 1 to each of these bins
np.add.at(counts, i, 1)
#The counts now reflect the number of points within each bin—in other words, a histogram 
#plt.plot(bins, counts, linestyle='steps');
plt.hist(x, bins, histtype='step');


# Fast sorting pf arrays with np.sort and np.argsort. To return a sorted version of the array without modifying the input, you can use
# np.sort:

# In[14]:


x = np.array([2, 1, 4, 3, 5])
np.sort(x) #x.sort() produces the same result.


#  argsort,returns the indices of the sorted elements:

# In[15]:


x = np.array([2, 1, 4, 3, 5])
x.argsort()


# Sorting along rows or columns

# In[16]:


rand = np.random.RandomState(42)
x = rand.randint(0, 10, (4, 6))
x


# In[17]:


# sort each column of X. Replace axix=0 with 1 to sort rows
np.sort(x, axis=0)


# Sometimes we’re not interested in sorting the entire array, but simply want to find the
# K smallest values in the array.

# In[18]:


x = np.array([7, 2, 3, 1, 6, 5, 4])
np.partition(x,4)


# Similarly to sorting, we can partition along an arbitrary axis of a multidimensional
# array:

# In[20]:


np.partition(x, 2, axis=0)


# The result is an array where the first two slots in each row contain the smallest values
# from that row, with the remaining values filling the remaining slots.

# Example: k-Nearest Neighbors. We will start by creating random set of 10 points on a 2 dimentional plane.

# In[22]:


x= rand.rand(10, 2)
x


# In[23]:


plt.scatter(x[:,0], x[:,1], s=100)


# Now we will compute the distance between each pair of points

# In[24]:


dist_sq = np.sum((x[:,np.newaxis,:] - x[np.newaxis,:,:]) ** 2, axis=-1)
dist_sq


# In[25]:


plt.scatter(x[:, 0], x[:, 1], s=100)
# draw lines from each point to its two nearest neighbors
K = 2
nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)
for i in range (x.shape[0]):
    for j in nearest_partition[i, :k+1]:
        # plot a line from X[i] to X[j]
        # use some zip magic to make it happen:
        plt.plot(*zip(X[j], X[i]), color='black')


# The above cell is supposed to draw connections between the dots. Some how it has refused to do this.

# Structured Arrays.Imagine that we have several categories of data on a number of people (say, name,
# age, and weight). We can create a structured array using a compound data type specification:
# 

# In[32]:


data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
                          'formats':('U10', 'i4', 'f8')}) #formats':((np.str_, 10), int, np.float32) does the same.
print(data.dtype)


# Here 'U10' translates to “Unicode string of maximum length 10,” 'i4' translates to “4-byte (i.e., 32 bit) integer,” and 'f8' translates to “8-byte (i.e., 64 bit) float.” 

# In[36]:


name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]
data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)


# In[40]:


data[1 #remember array indexes are not quoted.


# Using Boolean masking, this even allows you to do some more sophisticated opera‐
# tions such as filtering on age:

# In[41]:


# Get names where age is under 30
data[data['age'] < 30]['name']


# In[ ]:




