#!/usr/bin/env python
# coding: utf-8

# In[1]:


#@gyleodhis =====gyleodhis@outlook.com===
import pandas as pd
# pd.__version__ this checks the pandas version in your machine.
#Pandas Series Object
# A Pandas Series is a one-dimensional array of indexed data.
data = pd.Series([0.25,0.5,0.75,1])
data
#data.values 
#data.index


# In[2]:


###This explicit index definition gives the series object additional cappabilities.
###For example the index need not be an interger but can consist of values of any desired type.
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                  index=['a', 'b', 'c', 'd'])
data


# In[3]:


###And the item access works as expected:
data['c']


# In[4]:


#We can even use noncontiguous or nonsequential indices:
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=[2, 5, 3, 7])
data


# ## The Pandas DataFrame Object as a Series is an analog of a one-dimensional array with flexible indices, a DataFrame is an analog of a two-dimensional array with both flexible row indices and flexible column names. Consider the  following Series constructed from a python dictionary:

# In[5]:


population_dict = {'Nairobi': 38332521,
                   'Mombasa': 26448193,
                   'Kisumu': 19651127,
                   'Kakamega': 19552860,
                   'Machakos': 12882135}
population = pd.Series(population_dict)
population


# Let as now construct a new series listing the area of each of the five counties listed above

# In[6]:


area_dict = {'Nairobi': 423967, 'Mombasa': 695662, 'Kisumu': 141297,
'Kakamega': 170312, 'Machakos': 149995}
area = pd.Series(area_dict)
area


# Now with this along with the population Series from before, we can use a dictionary to construct a single two-dimensional object containing the below information:

# In[7]:


counties = pd.DataFrame({'population': population,
'area': area})
counties


# In[8]:


#the DataFrame has an index attribute that gives access to the index labels:
counties.index


# In[9]:


#Additionally, the DataFrame has a columns attribute, which is an Index object holding the column labels:
counties.columns


# In[10]:


#asking for the 'area' attribute returns the Series object containing the areas we saw earlier this is why the DataFrame behaves like a specialized array:
counties['area']


# # Data Selection in Series

# In[11]:


#Consider the Series below:
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
data


# In[12]:


data.keys() # Returns the keys of the Series.


# In[13]:


#You can extend a series by assigning to a new index value.
data['e'] = 1.25
data['f'] = 1.50
data['g'] = 1.75
data


# In[14]:


#Slicing a Series by explicit indexing
data['a':'c']


# In[15]:


#Slicing a Series by implicit index
data[0:2]


# Among these, slicing may be the source of the most confusion. Notice that when you
# are slicing with an explicit index (i.e., data['a':'c']), the final index is included in
# the slice, while when you’re slicing with an implicit index (i.e., data[0:2]), the final
# index is excluded from the slice.

# In[16]:


#Masking a Series
data[(data>0.3) & (data<0.8)]


# In[17]:


#Fancy Indexing
data[['a','g']]


# # Data Selection in Dataframe
# Recall that a DataFrame acts in many ways like a two-dimensional or structured array,
# and in other ways like a dictionary of Series structures sharing the same index

# ## DataFrame as a Dictionary
# Reconsider the county dictionary below:

# In[18]:


counties


# Like with the Series objects discussed earlier, this dictionary-style syntax can also be
# used to modify the object, in this case to add a new column:

# In[19]:


counties['density'] = counties['population']/counties['area']
counties


# ## DataFrames as a two-dimensional array
# We can examine the raw underlying data array using the values attribute:

# In[20]:


counties.values


# With this picture in mind, we can do many familiar array-like observations on the
# DataFrame itself. For example, we can transpose the full DataFrame to swap rows and
# columns:

# In[21]:


counties.T


# In[22]:


#passing a single index to an array accesses a row:
counties.values[1]


# In[23]:


# While passing a single “index” to a DataFrame accesses a column:
counties['area']


# In[24]:


counties.iloc[:3, :2] #iloc uses implicit indexing (iloc for positional indexing)


# In[25]:


counties.loc[:'Mombasa', :'population'] #returns upto and including the specified index(es) in this case returns rows upto Mombasa and columns upto population.
#.loc for label based indexing


# ## Other usefull indexing conventions.

# In[26]:


#First, while indexing refers to columns, slicing refers to rows:
counties['Machakos':'Mombasa'] # Example of slicing


# In[27]:


# Such slices can also refer to rows by number rather than by index:
counties[2:3] # Remember that slicing by indexing does not include the final index.


# In[28]:


#Similarly, direct masking operations are also interpreted row-wise rather than column-wise:
counties[counties.density<100]


# # Data Manupilation
# Since Pandas is build on top of Numpy; any numpy ufunc will work on Pandas Sereies and DataFrame objects.
# Lets first define a simple Series and DataFrame on which to demonstrate this:

# In[29]:


import numpy as np
rng=np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4))
ser


# In[30]:


df = pd.DataFrame(rng.randint(0,10,(3,4)),
                 columns=['A','B','C','D'])
df


# In[31]:


#If we apply a NumPy ufunc on either of these objects, the result will be another Pandas object with the indices preserved:
np.exp(ser)


# In[32]:


# Or a slightly more complex calculation:
np.sin(df * np.pi / 4)


# ## working with incomplete data
# Suppose we are working with population and area of major towns in Kenya; assume the ppopulataion data is from a defferent source while the area data is from a different source:

# In[33]:


city_area = pd.Series({'Nairobi': 423967, 'Mombasa':695662, 'Kisumu': 141297}, name = "city_area")
city_population = pd.Series({'Kakamega': 19552860, 'Machakos': 12882135, 'Nairobi': 38332521, 'Kisumu': 19651127}, name = 'city_population')

city_population


# In[34]:


# Let’s see what happens when we divide these to compute the population density:
city_density = city_population/city_area
city_density


# ## Any item for which one or the other does not have an entry is marked with NaN, or
# ## “Not a Number,” which is how Pandas marks missing data .

# In[35]:


# More example:
A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
A + B


# If using NaN values is not the desired behavior, we can modify the fill value using
# appropriate object methods in place of the operators. For example, calling A.add(B)
# is equivalent to calling A + B, but allows optional explicit specification of the fill value
# for any elements in A or B that might be missing:

# In[36]:


A.add(B, fill_value=0) # so addition occurs with 0 instead of NaN.


# ## Ufuncs: Operations Between DataFrame and Series
# Operations between a DataFrame and a Series are similar to operations between a two-dimensional and one dimensional NumPy array.

# In[37]:


A = rng.randint(10, size=(3, 4))
A


# In[38]:


A - A[1] #According to NumPy’s broadcasting rules; subtraction/addition between a two-dimensional array and one of its rows is applied row-wise.


# In[39]:


# In Pandas, the convention similarly operates row-wise by default:
df = pd.DataFrame(A, columns=list('QRST'))
df - df.iloc[0]


# In[40]:


# To perform a columnwise operation we use the axis key word
df.subtract(df['R'], axis=0)


# # Handling Missing Data
# ## Trade-offs in missing data conventions
# ### NaN and None in Pandas
# NaN and None both have their place, and Pandas is built to handle the two of them
# nearly interchangeably, converting between them where appropriate:

# In[41]:


pd.Series([1,np.nan,2, None])


# For types that don’t have an available sentinel value, Pandas automatically type-casts
# when NA values are present. For example, if we set a value in an integer array to
# np.nan, it will automatically be upcast to a floating-point type to accommodate the
# NA:

# In[42]:


x = pd.Series(range(4), dtype=int)
x


# In[43]:


x[2] = None
x


# From the above two examples we notice that in addition to casting the integer array to floating point, Pandas
# automatically converts the None to a NaN value.

# ## Operating on Null Values
# there are several useful methods for detecting, removing, and replacing null values in Pandas data structures. They are:
#     1. isnull()
#         Generate a Boolean mask indicating missing values
#     2. notnull()
#         Opposite of isnull()
#     3. dropna()
#         Return a filtered version of the data
#     4. fillna()
#         Return a copy of the data with missing values filled or imputed

# In[44]:


counties.isnull()


# In[45]:


# For a DataFrame, there are more options. Consider the following DataFrame:
df = pd.DataFrame([[1, np.nan, 2],
                   [2, 3, 5],
                   [np.nan, 4, 6]])
df


# ## We cannot drop single values from a DataFrame; we can only drop full rows or full columns
# 

# In[46]:


# By default, dropna() will drop all rows in which any null value is present:
df. dropna()


# We can however drop NA values along a different axis; axis=1 drops all col‐
# umns containing a null value:

# In[47]:


df.dropna(axis='columns')


# In[48]:


# above methods drop some good data and can lead to huge loses of data. To gain more control on what to drop we use the how or thresh parameters.
df[3] = np.nan # added a third column containing all NaN.
df


# In[49]:


df.dropna(axis = 'columns', how = 'all') # Drops a column that contains all values NaN. (Replce 'all' with 'any' for droping any column with NaN value)


# ## For finer-grained control, the thresh parameter lets you specify a minimum number of non-null values for the row/column to be kept:

# In[50]:


df.dropna(axis='rows', thresh=3) #This drops the first and last rows, because they contain only two nonnull values.


# # Filling Null Values

# In[51]:


# Consider the following Series:
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data


# In[52]:


data.fillna(5) # Fills all the NaN values with the passed value in this case 5.


# In[53]:


# We can specify a forward-fill to propagate the previous value forward (the previous value before a NaN is placed in the predeeding NaN value.):
data.fillna(method='ffill')


# In[54]:


# Or we can specify a back-fill to propagate the next values backward:
data.fillna(method='bfill')


# In[55]:


# For DataFrames, the options are similar, but we can also specify an axis along which the fills take place:
df


# In[56]:


df.fillna(method='ffill', axis=1) # "axis = 1 works along columns while axis - 0 works n rows."


# ## Notice that during ffill and bfill, if a previous or the next value is not available the NA value remains.

# Representing a twdimentional data within a one dimentional Series. The Pandas MultiIndex enables us to perform this.Suppose you want to track the population of cities across two different years. Consider the following:

# In[57]:


index = [('Nairobi', 2000), ('Mombasa', 2000), ('Kisumu', 2000), ('Kakamega', 2000),('Nairobi', 2010), ('Mombasa', 2010), ('Kisumu', 2010), ('Kakamega', 2010)]
city_population = [10555760, 10782135, 12332451, 10651103, 19552860, 12882135, 38332521, 19651127]
pop = pd.Series(city_population, index=index)
pop


# In[58]:


# We can create a multiindex from the above tuples as follows:
index = pd.MultiIndex.from_tuples(index)
index


# In[59]:


# We ca see a hierarchical representation of the data when we reindex our series with MultiIndex
pop = pop.reindex(index)
pop
#Here the first two columns of the Series representation show the multiple index values, while the third column shows the data


# In[60]:


# Sometimes it is convenient to name the levels of the MultiIndex. You can accomplish this by passing the names argument to any of the above MultiIndex constructors, or by setting the names attribute of the index after the fact:
pop.index.names = ['state', 'year']
pop


# In[61]:


# To access all data for which the second index is 2010, we can simply use the Pandas slicing notation:
pop[:,2010]


# ## MultiIndex as extra dimension
# You might notice something else here: we could easily have stored the same data
# using a simple DataFrame with index and column labels. In fact, Pandas is built with
# this equivalence in mind. The unstack() method will quickly convert a multiplyindexed Series into a conventionally indexed DataFrame:

# In[62]:


pop_df = pop.unstack()
pop_df


# In[63]:


# The .stack() does the opposite. Remember it does not work on a Series but a DataFrame.
pop_df.stack()


# ## we can also use it to represent data of three or more dimensions in a Series or DataFrame. Each extra level in a multi-index represents an extra dimension of data; taking advantage of this property gives us much more flexibility in the types of data we can represent. For example let us add age demographic data about minors under the age of 18.

# In[64]:


pop_df = pd.DataFrame({'total': pop,
                      'Under18': [9267089, 9284094,4687374, 4318033,5906301, 6879014,5786301, 9519014]})
pop_df


# In[65]:


# Here we can now compute the fraction of people under 18 by year, given the above data:
f_u18 = pop_df['Under18'] / pop_df['total']
f_u18.unstack()


# ## Just as rows can have multiple levels of indices, the columns can have multiple levels as well. Consider the following.

# In[66]:



index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])
# mock some data
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37
# create the DataFrame
health_data = pd.DataFrame(data, index=index, columns=columns)
health_data


# This is fundamentally four-dimensional data, where the dimensions are the
# subject, the measurement type, the year, and the visit number. With this in place we
# can, for example, index the top-level column by the person’s name and get a full Data
# Frame containing just that person’s information:

# In[67]:


health_data['Sue']


# ## For complicated records containing multiple labeled measurements across multipletimes for many subjects (people, countries, cities, etc.), use of hierarchical rows and columns can be extremely convenient!

# In[ ]:





# # Indexing and Slicing a MultiIndex
# ## Multiply indexed Series

# In[68]:


#Consider the multiply indexed Series of state populations we saw earlier:
pop


# In[69]:


# We can access single elements by indexing with multiple terms:
pop['Kakamega', 2010] # or just pop['Kakamega']


# In[70]:


#With sorted indices, we can perform partial indexing on lower levels by passing an empty slice in the first index:
pop[:,2000]


# In[71]:


# Other types of indexing and selection (discussed in “Data Indexing and Selection” on page 107) work as well; for example, selection based on Boolean masks:
pop[pop > 22000000]


# In[72]:


# Selection based on fancy indexing also works:
pop [['Nairobi', 'Mombasa']]


# # Multiply indexed DataFrames

# In[73]:


# Let us reconsider our health dataframe
health_data


# In[74]:


# we can recover Sue’s heart rate data with a simple operation:
health_data['Sue','HR']


# ## Sorted and Unsorted Indices

# In[75]:


# Lets us create a multiIndexed data where the indices are not sorted
index = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['char', 'int']
data


# In[76]:


# To sort this data we can use the following funcs:
data = data.sort_index()
data


# In[77]:


# With the index sorted in this way, partial slicing will work as expected:
data['a':'b']


# ## Stacking and unstacking indices

# In[78]:


#it is possible to convert a dataset from a stacked multi-index to a simple two-dimensional representation, optionally specifying the level to use:
pop.unstack(level=0)


# In[79]:


# The opposite of unstack() is stack(), which here can be used to recover the original series:
pop.unstack().stack()


# ## Index setting and resetting
# Another way to rearrange hierarchical data is to turn the index labels into columns;
# this can be accomplished with the reset_index method.

# In[80]:


pop_flat = pop.reset_index(name='population')
pop_flat


# # Data Aggregations on Multi-Indices
# We’ve previously seen that Pandas has built-in data aggregation methods, such as
# mean(), sum(), and max().

# In[81]:


# Finding the avarage in the two visits in each year.
data_mean = health_data.mean(level='year')
data_mean


# In[82]:


#By further making use of the axis keyword, we can take the mean among levels on the columns as well:
data_mean.mean(axis=1, level='type')


# In[ ]:





# # Combining Datasets: Concat and Append.

# In[84]:


# lets look at a simple concatination of Series and DataFrame using Pandas .concat function
# We define a function that creates a dataframe of a particular form:
def make_df(cols, ind):
    """Lets quickly make a dataframe"""
    data = {c: [str(c) + str(i) for i in ind]
           for c in cols}
    return pd.DataFrame(data, ind)
#Example of a dataframe
make_df('ABCDE', range(3))


# In[88]:


# Concatenating a Series
ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
pd.concat([ser1, ser2])


# In[93]:


# Concatinating a datafram
df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
print('The first dataframe''\n',df1); print('The second dataframe''\n',df2); print('Final Dataframe''\n',pd.concat([df1, df2]))


# In[105]:


# By default, the concatenation takes place row-wise within the DataFrame (i.e.,axis=0). To perform concatination along columns use, the axis=1
print('Final Dataframe''\n',pd.concat([df1, df2], axis=1))

By default, the entries for which no data is available are filled with NA values. To
change this, we can specify one of several options for the join and join_axes param‐
eters of the concatenate function. By default, the join is a union of the input columns
(join='outer'), but we can change this to an intersection of the columns using
join='inner'
# In[106]:


print('Final Dataframe''\n',pd.concat([df1, df2], axis=1, join='inner'))


# Catching the repeats as an error. If you’d like to simply verify that the indices in the
# result of pd.concat() do not overlap, you can specify the verify_integrity flag.
# With this set to True, the concatenation will raise an exception if there are duplicate
# indices. Here is an example, where for clarity we’ll catch and print the error message: 
# ## pd.concat([x, y], verify_integrity=True)

# Ignoring the index. Sometimes the index itself does not matter, and you would prefer
# it to simply be ignored. You can specify this option using the ignore_index flag. With
# this set to True, the concatenation will create a new integer index for the resulting
# Series:
# ## pd.concat([x, y], ignore_index=True)

# ## Adding Multi-Index Keys
# Another alternative is to use the keys option to specify a label
# for the data sources; the result will be a hierarchically indexed series containing the
# data:

# In[102]:


x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])
print(pd.concat([x, y], keys=['Table1', 'Table2']))


# # The Append Method.
# ## Because direct array concatenation is so common, Series and DataFrame objects have an append method that can accomplish the same thing in fewer keystrokes. 

# In[107]:


# Forexample, rather than calling pd.concat([df1, df2]), you can simply call df1.append(df2):
df1.append(df2)


# Keep in mind that unlike the append() and extend() methods of Python lists, the
# append() method in Pandas does not modify the original object—instead, it creates a
# new object with the combined data

# # Combining Datasets: Merge and 
# ## one-to-one joins

# In[108]:


# Consider the following dataframes
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
print(df1); print(df2)


# In[113]:


# To merge the information into a single dataframe
df3 = pd.merge(df1,df2)
df3


# ## Many-to-one joins
# These are joins in which one of the two key columns contains duplicate entries:
# Consider the following:

# In[111]:


df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
pd.merge(df3,df4)


# ## Many-to-Many joins
# If the key column in both the left and right array contains duplicates, then the result is a many-to-many merge.
# By performing a many-to-many join, we can recover the skills associated with any
# individual person:

# In[121]:


df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'coding', 'linux',
                               'spreadsheets', 'organization']})
#print(df5)
print(pd.merge(df1, df5))


# In[ ]:




