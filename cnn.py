
# coding: utf-8

# In[1]:

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import urlretrieve
import cPickle as pickle
import os
import gzip
import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# # Load Cifar-10

# In[4]:

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


# In[3]:

batch1=unpickle('cifar-10-batches-py/data_batch_1')
batch2=unpickle('cifar-10-batches-py/data_batch_2')
batch3=unpickle('cifar-10-batches-py/data_batch_3')
batch4=unpickle('cifar-10-batches-py/data_batch_4')
batch5=unpickle('cifar-10-batches-py/data_batch_5')
testbatch=unpickle('cifar-10-batches-py/test_batch')


# In[4]:

X_train=np.concatenate((batch1[b'data'],batch2[b'data'],batch3[b'data'],batch4[b'data'],batch5[b'data']),axis=0)
y_train=np.concatenate((batch1[b'labels'],batch2[b'labels'],batch3[b'labels'],batch4[b'labels'],batch5[b'labels']),axis=0)
#X_val, y_val, 
X_test=testbatch[b'data']
y_test=testbatch[b'labels']


# In[5]:

labelname=unpickle('cifar-10-batches-py/batches.meta')


# In[7]:

labelname['label_names']


# X_train -= np.mean(X_train, axis = 0).astype(X_train.dtype) # zero-center
# X_train /= np.std(X_train, axis = 0).astype(X_train.dtype)

# def whiten(X,fudge=1E-18):
# 
#    # the matrix X should be observations-by-components
# 
#    # get the covariance matrix
#    Xcov = np.dot(X.T,X)
# 
#    # eigenvalue decomposition of the covariance matrix
#    d, V = np.linalg.eigh(Xcov)
# 
#    # a fudge factor can be used so that eigenvectors associated with
#    # small eigenvalues do not get overamplified.
#    D = np.diag(1. / np.sqrt(d+fudge))
# 
#    # whitening matrix
#    W = np.dot(np.dot(V, D), V.T)
# 
#    # multiply by the whitening matrix
#    X_white = np.dot(X, W)
# 
#    return X_white, W
# 

# def svd_whiten(X):
# 
#     U, s, Vt = np.linalg.svd(X)
# 
#     # U and Vt are the singular matrices, and s contains the singular values.
#     # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
#     # will be white
#     X_white = np.dot(U, Vt)
# 
#     return X_white
# 

# X_pca=[]
# for x in new_X_train:
#     x=whiten(x)
#     X_pca.append(x)

# X_svd=[]
# for x in new_X_train:
#     x=svd_whiten(x)
#     X_svd.append(x)

# cov = np.dot(X_train.T, X_train) / X_train.shape[0]

# ## CNN without Data Preprocessing

# In[15]:

X_train1=np.array(X_train)
X_train1=np.reshape(X_train, (-1,3,32,32), order='F')
y_train=np.array(y_train)
y_train= y_train.astype(np.int32) 


# In[16]:

net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            #('conv2d3', layers.Conv2DLayer),
            #('maxpool3', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 3, 32, 32),
    # layer conv2d1
    conv2d1_num_filters=32,
    conv2d1_filter_size=(3, 3),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),  
    # layer maxpool1
    maxpool1_pool_size=(2, 2),    
    # layer conv2d2
    conv2d2_num_filters=64,
    conv2d2_filter_size=(3, 3),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    maxpool2_pool_size=(2, 2),
    # layer conv2d3
    #conv2d3_num_filters=128,
    #conv2d3_filter_size=(3, 3),
    #conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool3
    #maxpool3_pool_size=(2, 2),
    # dropout1
    dropout1_p=0.5,    
    # dense
    dense_num_units=256,
    dense_nonlinearity=lasagne.nonlinearities.rectify,    
    # dropout2
    dropout2_p=0.5,    
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=10,
    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=10,
    verbose=1,
    )


# In[17]:

nn1 = net1.fit(X_train1, y_train)


# In[22]:

X_train_rescaling=X_train1/255


# In[23]:

net2 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            #('conv2d3', layers.Conv2DLayer),
            #('maxpool3', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 3, 32, 32),
    # layer conv2d1
    conv2d1_num_filters=32,
    conv2d1_filter_size=(3, 3),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),  
    # layer maxpool1
    maxpool1_pool_size=(2, 2),    
    # layer conv2d2
    conv2d2_num_filters=64,
    conv2d2_filter_size=(3, 3),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    maxpool2_pool_size=(2, 2),
    # layer conv2d3
    #conv2d3_num_filters=96,
    #conv2d3_filter_size=(3, 3),
    #conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool3
    #maxpool3_pool_size=(2, 2),
    # dropout1
    dropout1_p=0.5,    
    # dense
    dense_num_units=256,
    dense_nonlinearity=lasagne.nonlinearities.rectify,    
    # dropout2
    dropout2_p=0.5,    
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=10,
    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=10,
    verbose=1,
    )


# In[24]:

nn2 = net2.fit(X_train_rescaling, y_train)


# # Data Preprocessing
# #### RGB to single grayscale
# #### Flatten grayscale
# #### ZCA_Whitening/PCA_Whitening

# In[8]:

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def flatten_matrix(matrix):
    vector = matrix.flatten(1)
    vector = vector.reshape(1, len(vector))
    return vector    


# In[9]:

new_X_train=[]
for img in X_train:
    img=np.reshape(img, (32,32,3), order='F')
    gray=flatten_matrix(rgb2gray(img)/255)
    new_X_train.append(gray)


# In[10]:

def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
    U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
    epsilon = 0.1                #Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T) #ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs)   #Data whitening


# In[11]:

new_X_train=np.asarray(new_X_train)
X_zca=[]
for x in new_X_train:
    x=zca_whitening(x)
    X_zca.append(x)


# In[12]:

X_zca=np.array(X_zca)
X_zca=np.reshape(X_zca,(-1,1,32,32))


# In[11]:

img=np.reshape(X_train[0], (32,32,3), order='F')
gray = rgb2gray(img)/255
#img = mpimg.imread('image.png')
plt.subplot(1, 2, 1); 
plt.axis('off');
plt.imshow(img);

plt.subplot(1, 2, 2); 
plt.axis('off');
plt.imshow(gray, cmap = plt.get_cmap('gray'));

#plt.subplot(1, 3, 3); 
#plt.axis('off');
#plt.imshow(X_zca[0],cmap=cm.binary);

plt.show()


# In[13]:

net3 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            # ('conv2d3', layers.Conv2DLayer),
            # ('maxpool3', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 1, 32, 32),
    # layer conv2d1
    conv2d1_num_filters=32,
    conv2d1_filter_size=(3, 3),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),  
    # layer maxpool1
    maxpool1_pool_size=(2, 2),    
    # layer conv2d2
    conv2d2_num_filters=64,
    conv2d2_filter_size=(3, 3),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    maxpool2_pool_size=(2, 2),
    # layer conv2d3
    # conv2d3_num_filters=96,
    # conv2d3_filter_size=(3, 3),
    # conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool3
    # maxpool3_pool_size=(2, 2),
    # dropout1
    dropout1_p=0.2,    
    # dense
    dense_num_units=256,
    dense_nonlinearity=lasagne.nonlinearities.rectify,    
    # dropout2
    dropout2_p=0.2,    
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=10,
    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.005,
    update_momentum=0.9,
    max_epochs=15,
    verbose=1,
    )


# In[16]:

# Train the network
nn3 = net3.fit(X_zca, y_train)


# In[17]:

new_X_test=[]
for img in X_test:
    img=np.reshape(img, (32,32,3), order='F')
    gray=flatten_matrix(rgb2gray(img)/255)
    gray=np.asarray(gray)
    x=zca_whitening(gray)
    new_X_test.append(x)

new_X_test=np.array(new_X_test)
new_X_test=np.reshape(new_X_test,(-1,1,32,32))
y_test=np.array(y_test)
y_test= y_test.astype(np.int32)


# In[18]:

preds = net3.predict(new_X_test)


# In[40]:

from sklearn.metrics import accuracy_score
accuracy_score(y_test, preds)


# In[42]:

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, preds)


# In[19]:

cm = confusion_matrix(y_test, preds)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[22]:

visualize.plot_conv_weights(net3.layers_['conv2d1'])
plt.show()


# In[24]:

dense_layer = layers.get_output(net3.layers_['dense'], deterministic=True)
output_layer = layers.get_output(net3.layers_['output'], deterministic=True)
input_var = net3.layers_['input'].input_var
f_output = theano.function([input_var], output_layer)
f_dense = theano.function([input_var], dense_layer)


# In[39]:

instance = X_test[0][None, :,:]
get_ipython().magic('timeit -n 500 f_output(instance)')

#500 loops, best of 3: 858 µs per loop


# In[ ]:

pred = f_output(instance)
N = pred.shape[1]
plt.bar(range(N), pred.ravel())
plt.show()


# In[ ]:

pred = f_dense(instance)
N = pred.shape[1]
plt.bar(range(N), pred.ravel())
plt.show()


# In[35]:

X_test[0].shape


# In[ ]:



