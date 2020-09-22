#!/usr/bin/env python
# coding: utf-8

# In[14]:


# refer to MTH319 Tutorial -No 2 (Part I) - Python Basics

#string data
first = 'black'
second = 'jack'
third = '''You can create
long strings'''
print(first +' ' + second)
print(third)


# In[13]:


x=12.7
print('the answer is '+repr(x))


# In[35]:


what = 'This parrot is dead' #index starts from 0

print('Type of what is:')
print(type(what)) 

print('Type of what[3] is:')
print(type(what[3])) 

print(what[3])
print(what[0:3])

start =3
finish = 8
print(what[3:8])

#what.split()
what.split(maxsplit=2) #with maxsplit+1 elements

what2 = 'Note that:'
print(what2+what)


# In[40]:


#numeric data
a =2
b= 4

print(a+b)

print('%d+%d=%d'%(a,b,a+b))

print(a-b)
print(a*b)
print(a**b)
print(a/b)
print(a%b)


# In[122]:


# List, Tuples and Dictionaries

newlist =[7, 9, 12, 15, 17, 19 , 103]

print(type(newlist))
print(newlist)

newlist3 = [i*i for i in newlist]
print(newlist3)

newlist2 =[7, 9, 12, 15,         17, 19 , 103]
print(newlist2)

print('length of %s=%d'%('newlist2', len(newlist2)))

newlist3 = [1, 2, [10,20, 30,[10,50,100, [0,1,2]]], 5, [1,7, 8]]
print(newlist3[0:2])
print(newlist3[2])
print(newlist3[2][3])
print(newlist3[2][3][3])
print(newlist3[2][3][3][0])
print(newlist3[2][3][3][1])
print(newlist3[2][3][3][2])


# In[76]:


thelist =[0,5,10,15,20]

print(thelist[1:4])

thelist[1:4]=[6,7,8]
print(thelist[0:4])

first = [7,9,'dog']
second = ['cat', 13, 14,12]
print(first+second)

# 'in' operator
squarepairs =[[0,0], [1,1], [2,4], [3,9]]
print([2,4] in squarepairs)
print([0,4] in squarepairs)
print(4 in squarepairs)

# append method
funiture = ['couch', 'chair', 'table']
funiture.append('footstool')
print(funiture)

a = [1, 5, 7, 9]
b = [10, 20, 30, 40]
a.extend(b)
print(a)
b.extend(a)
print(b)

oldlist = [7, 9, -3, 5, -8, 19]
print('the length of %s = %d'%('oldlist', len(oldlist)))

newlist = []
for i in oldlist:
    if i<0: 
        newlist.append(i)

print(newlist)

newlist2 = oldlist[:]  # or oldlist
print(newlist2)


# In[78]:


#Tuple object
values = (3, 4, 5, 4, 5, 5,5)
values + (7,)
print(values)

newvalue = 7,
print(values+newvalue)

print(list(values).count(5))


# In[38]:


# Dictionaries
phonelist=[('Fred','555-1231'),('Andy','555-1195'), ('Sue', '555-2194')]  #list

phonedict = {}  # empty one
phonedict ={'Fred':'555-1231','Andy':'555-1195','Sue': '555-2194'}  # key:values pairs
print(type(phonedict))
print(phonedict.keys())
print(phonedict.values())
print(phonedict['Sue'])

tupledict={(7,3):21, (13,4):52, (18,5):90}
print(tupledict[(4+3,2+1)])
print(tupledict[(13,4)])
print(tupledict.keys())
print(tupledict.values())


# In[120]:


# Logical operators and if statement
x = 2
if x == 1: 
    z = 1
    print ('Setting z to 1')
elif x == 2:
    y = 2
    print ('Setting y to 2')
elif x == 3:
    w = 3
    print('Setting w to 3')
else:
    print('Other choice...')
    
    
# for loops
names =[('Smith','John'),('John','Fred'),('Williams', 'Sue')]
for i in names:
    print('%s %s'%(i[1],i[0]))

    
# range
x = [1,2,3,4,5]
for i in range(len(x)):
    x[i]= i + 2   # try x[i]= i + 1
    print(i)

print(x)

prices = [12.00, 14.00, 17.00]
taxes = [0.48, 0.56, 0.68]
total = [] # empty list
for i in range(len(prices)):
    total.append(prices[i]+taxes[i])
    #print(total[i])

print(total)

# while loop
num = 7.0
oldguess = 0.
guess = 3.
while abs(num - guess**3) > 1.e-8:
    oldguess = guess
    guess = oldguess - (oldguess**3-num)/(3*oldguess**2)
    
print('guess=%f,guess^3=%d'%(guess, guess**3))


# break / continue
x  = [7, 12, 92, 19, 18, 44, 31]
xmax = x[0]
imax = 0
for i in range(1,len(x)):
    if x[i] <= xmax: 
        print(i)
        continue       
    xmax = x[i]
    imax = i
    #else:
    #    print('the max iteration of %d is reached.'%(i))
    #    break

print('Maximum value  is %d at position %d.'%(xmax, imax))

#2
x  = [7, 12, 92, 19, 18, 44, 31]
x_find = x[5]
imax = 0
for i in range(1,len(x)):
    if x[i] != x_find:
        continue
    else:
        print('the value of x_find is found with %d at %d.'%(x[i],i))
        break

# new way to construct list
newlist =[7, 9, 12, 15, 17, 19 , 103]
print(newlist)
newlist3 = [i*i for i in newlist]
print(newlist3)


# In[15]:


# functions
scale = 10.0

def doscale(list):
    newlist = []
    for i in list:
        newlist.append(i/scale)
    return newlist    

mylist = [1,2,3,4,5]
otherlist = doscale(mylist)
print(otherlist)

#2
def merge(list1, list2):
    newlist = list1[:]
    for i in list2:
        newlist.append(i)
    return newlist

one = [7,12, 19, 44, 32]
two = [8,12, 19, 31, 44, 66]
print(merge(one,two))

#3
def change_letter(string, frm=' ', to= ' '):
    newstring = ' '
    for i in string:
        if i== frm:
            newstring = newstring + to
        else:
            newstring = newstring + i
    return newstring

print(change_letter('string with blanks', to= '+'))

#4: anonymous functions
results = lambda x: x+1
print(results(3))


# In[32]:


# refer to MTH319 Tutorial -No 2 (Part II) - NumPy and SciPy

import numpy as np   # a new module

x = np.array([[1,2,3],[3,4,6]]) #2 by 3 matrix
print(np.size(x,0))
print(np.size(x,1))
print(np.size(x))

print(np.std(x,0))
print(np.std(x,1))
print(np.std(x))

total = x.sum()
print(total)

z = np.random.rand(50)  #50 random obs from U[0,1] 
print(z)
y = np.random.normal(size=50)  # from N(0,1)
print(y)
r = np.array(range(0,10), dtype=float)/10
print(r)


# In[54]:


import numpy as np  # a new module
import scipy as sp

cashflows =[50, 40, 20,10, 50]
npv = sp.npv(0.1,cashflows) # NPV
round(npv, 2)
print(npv)

ret = sp.array([0.1, 0.05, -0.02, 0.05])
print(sp.mean(ret))

geometric_ret = pow(sp.prod(ret+1),1./len(ret))-1
print(geometric_ret)

print(sp.unique(ret))

print(sp.median(ret))

# arrays of ones, zeros and identity matrix
a = np.zeros(10) # arry with 10 zeros
print(a)
b = np.zeros((3,2),dtype = float)  #3 by 2 matrix
print(b)
c = np.ones((4,3),int)  #float 4 by 3 matrix
print(c)
d = np.array(range(10),float)
print(d)
e1 = np.identity(4)
print(e1)
e2 = np.eye(4)
print(e2)
e3 = np.eye(4,k=1)
print(e3)
f = np.arange(1,20,3, float)
print(f)
g = np.array([[2,2,2], [3,3,3]])  # 2 by 3
print(g)

# operation
pv = np.array([[100,10,10.2],[34,22,34]])  # 2 by 3
x = pv.flatten()
print(x) # a vector

vp2 = np.reshape(x, [3,2])
print(vp2)

# matrix multiplication
x = np.array([[1,2,3],[4,5,6]],float)
y = np.array([[1,2],[3,3],[4,5]], float)
print(np.dot(x,y))

# dot production
x = np.array([[1,2,3],[4,5,6]],float)
y = np.array([[2,1,2],[4,0,5]],float)
print(x*y)


# In[82]:


# refer to MTH319 Tutorial -No 2 (Part III) - Data Visualization

numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

np.random.seed(1000)
y = np.random.standard_normal(20)

#plt.plot(y.cumsum())

x = range(len(y))

plt.figure(figsize=(7,4))
#plt.plot(x,y)
plt.plot(y.cumsum(), 'b', lw=1.5)
plt.plot(y.cumsum(), 'ro')
plt.grid(True)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')

# 2
np.random.seed(2000)
y = np.random.standard_normal((20,2)).cumsum(axis=0)
print(y)

y[:,0] = y[:,0]*10

plt.figure(figsize=(7,5))
plt.title('A Simple Plot')
plt.subplot(211)
plt.plot(y[:,0], 'b', lw=1.5, label='1st')
plt.plot(y[:,0], 'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
#plt.xlabel('index')
plt.ylabel('value')

plt.subplot(212)
plt.plot(y[:,1], 'g', lw=1.5, label='2nd')
plt.plot(y[:,1], 'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')


# In[69]:


# interpolation
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d # 1-dimension

x = np.linspace(0,10,10)
print(x)

y = np.exp(-x/3.0)
print(y)

f = interp1d(x, y)
#print(f)

f2 = interp1d(x, y, kind='cubic')
#print(f2)

xnew = np.linspace(0,10,40)
plt.plot(x, y,'o', xnew, f(xnew), '-', xnew, f2(xnew), '-*')
plt.legend(['data','linear','cubic'], loc='best')
plt.show()


# In[ ]:





# In[ ]:




