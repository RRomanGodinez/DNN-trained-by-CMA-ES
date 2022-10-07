#!/usr/bin/env python
# coding: utf-8

# In[1]:

from keras.models import Sequential
from tensorflow.keras import optimizers
import itertools
import numpy as np
import cma
from sklearn.model_selection import KFold
from keras.models import Sequential
import zipfile
import pandas as pd
from tensorflow.keras.utils import to_categorical
import csv
import os.path
import shutil
from sklearn.preprocessing import MinMaxScaler, StandardScaler




from keras.layers import Dense
from DEN import DENlayer
from DMN import DMNlayer
from DSN import DSNlayer

from PreTrain.kmeans import bkmeans
from PreTrain.kmeans import ekmeans
from PreTrain.kmeans import skmeans

def write_data(File, DataSet,Acc, Std, Modelo, Neuronas, Activacion, Sigma, Xpopsize,Kfold, Scaler):
    # csv header
    fieldnames = ['DataSet', 'Acc', 'Std', 'Modelo', 'Neuronas','Activacion','Sigma','Xpopsize','Kfold','Scaler']


    # csv data
    rows = [
        {'DataSet': DataSet,
         'Acc': Acc,
         'Std': Std,
        'Modelo': Modelo,
        'Neuronas': Neuronas,
        'Activacion': Activacion,
        'Sigma': Sigma,
        'Xpopsize': Xpopsize,
        'Kfold': Kfold,
        'Scaler':Scaler
        }
    ]

    flag=True
    if os.path.exists(File) == False:
        flag = False
    with open(File, 'a', encoding='UTF8', newline='') as f:
        if flag == False:
            print(fieldnames)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)          
        else:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            #writer.writeheader()
            writer.writerows(rows)  

def get_data(Folder_dir, File_name, len_test=0.1):
    zip_ref = zipfile.ZipFile(Folder_dir+"/"+File_name+".zip", 'r')
    zip_ref.extractall("./")
    zip_ref.close()
    data = pd.read_csv(File_name+".csv")
    #data = data.drop(labels='Unnamed: 32',axis=1)
    from sklearn import preprocessing 
    label = preprocessing.LabelEncoder() 
 

    X = data.drop(['Class'], axis=1)
    if 'id' in  X.columns:
        X = X.drop(['id'],axis=1)
    print(X.head())
    label = preprocessing.LabelEncoder() 
    data['Class'] = label.fit_transform(data['Class'])
    Y = data['Class']
    X = X.values
    
    if len(Y.unique())>2:
        Y = to_categorical(Y, dtype ="uint8")
    if len_test == 0:
        x_train = X
        y_train = Y
        return x_train,y_train
        
    else:
        x_train = X[:int(1-(len(Y)*len_test))]
        y_train = Y[:int(1-(len(Y)*len_test))]
        x_test = X[int(1-(len(Y)*len_test)):]
        y_test = Y[int(1-(len(Y)*len_test)):]
        return x_train,y_train,x_test,y_test
    
        
    

   
    


def model_init(x,y,layers = ['DMN', 'DMN'], neurons=[2,2,1], activations=['tanh','sigmoid'], dendrites=[]):

    model = Sequential()
    nlayer = 1
    print("layers", layers)
    for layer in layers:
        if layer == 'P':
            if nlayer == 1:
                model.add(Dense(neurons[1], activation = activations[0], input_shape = (neurons[0],) ) )
            else:
                model.add(Dense(neurons[2], activation = activations[1]))

        if layer == 'DMN':
            if nlayer == 1:
                model.add(DMNlayer(neurons[1], dendrites, activation = activations[0], input_shape = (neurons[0],)))
            else:
                model.add(DMNlayer(neurons[2], activation = activations[1]))

        if layer == 'DEN':
            if nlayer == 1:
                model.add(DENlayer(neurons[1], dendrites, activation = activations[0], input_shape = (neurons[0],)))
            else:
                model.add(DENlayer(neurons[2], activation = activations[1]))
        
        if layer == 'DSN':
            if nlayer == 1:
                model.add(DSNlayer(neurons[1], dendrites, activation = activations[0], input_shape = (neurons[0],)))
            else:
                model.add(DSNlayer(neurons[2], activation = activations[1])) 
        nlayer = nlayer+1
   
    adam = optimizers.Adam(learning_rate=0.01)
    if activations[1] == 'softmax':
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        
    if activations[1] == 'sigmoid':
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        
       
    model.summary()    
    return model
def get_theta0(weights):
    theta0 = []
       
    if isinstance(weights, (np.ndarray, list) ):
        
        try:
            weights =list(itertools.chain(*weights))
            weights = get_theta0(weights,deep)
    
        except:
            
            try:
                weights =list(itertools.chain(*weights))
                
                for t in weights:
                    if isinstance(t, (np.ndarray) ):
                        for t_aux in t:
                            theta0.append(t_aux)
                    if isinstance( t, (np.floating, float) ):
                        theta0.append(t)
            except:
                for t in weights:
                    if isinstance(t, (np.ndarray) ):
                        for t_aux in t:
                            if isinstance(t_aux, (np.ndarray) ):
                                for t_aux_2 in t_aux:
                                    theta0.append(t_aux_2)
                            else:
                                theta0.append(t_aux)
                    if isinstance( t, (np.floating, float) ):
                        theta0.append(t)
            weights = theta0
            return weights
                
    return weights

def set_params(theta, neurons, model):
    layer = 0
    model_type = []
    for layer_type in model.layers[::]:
        
        if isinstance(layer_type, Dense):
            if layer == 0:
             
                l1 = theta[:neurons[0]*neurons[1] + neurons[1]]
                l2 = theta[neurons[0]*neurons[1] + neurons[1]:]
                w1_1 = l1[:neurons[0]*neurons[1]].reshape(neurons[0],neurons[1])
                w1_2 = l1[neurons[0]*neurons[1]:].reshape(neurons[1],)
                
            if layer == 1:
               
                w2_1 = l2[:neurons[1]*neurons[2]].reshape(neurons[1],neurons[2])
                w2_2 = l2[neurons[1]*neurons[2]:].reshape(neurons[2],)
         
        if isinstance(layer_type, DMNlayer):
            if layer == 0:
                l1 = theta[:neurons[0] * neurons[1] * 2]
                l2 = theta[neurons[0] * neurons[1] * 2:]
                w1_1 = l1[:neurons[0]*neurons[1]].reshape(neurons[1],neurons[0])
                w1_2 = l1[neurons[0]*neurons[1]:].reshape(neurons[1], neurons[0])
                
            if layer == 1:
                w2_1 = l2[:neurons[1]*neurons[2]].reshape(neurons[2],neurons[1])
                w2_2 = l2[neurons[1]*neurons[2]:].reshape(neurons[2],neurons[1])
                
            
        if isinstance(layer_type, DENlayer):
            if layer == 0:
                l1 = theta[:neurons[0] * neurons[1] + (neurons[0]**2)*neurons[1] ]
                l2 = theta[neurons[0] * neurons[1] + (neurons[0]**2)*neurons[1]:]
                w1_1 = l1[:neurons[0] * neurons[1]].reshape(neurons[1],1,neurons[0])
                w1_2 = l1[neurons[0] * neurons[1]:].reshape(neurons[1],neurons[0],neurons[0])
            if layer == 1:
                w2_1 = l2[:neurons[1]*neurons[2]].reshape(neurons[2],1,neurons[1])
                w2_2 = l2[neurons[1]*neurons[2]:].reshape(neurons[2],neurons[1],neurons[1])
            
        if isinstance(layer_type, DSNlayer):
            if layer == 0:
                l1 = theta[:neurons[0] * neurons[1] +neurons[1] ]
                l2 = theta[neurons[1] * neurons[0] + neurons[1]:]
                w1_1 = l1[:neurons[1] * neurons[0]].reshape(neurons[1],1,neurons[0])
                w1_2 = l1[neurons[1] * neurons[0]:].reshape(neurons[1],1)
        
            if layer == 1:
                w2_1 = l2[:neurons[1]*neurons[2]].reshape(neurons[2],1,neurons[1])
                w2_2 = l2[neurons[1]*neurons[2]:].reshape(neurons[2],1)
                    
        layer = 1
           
        
    theta = [w1_1,w1_2,w2_1,w2_2]
    model.set_weights(theta)
  
    
        

def train_acc(theta, neurons, xtrain, ytrain, model):
    set_params(theta, neurons, model)
    _, acc = model.evaluate(xtrain,ytrain, verbose=False)
    return -acc

def val_acc(theta, neurons, xval, yval, model):
    set_params(theta, neurons, model)
    _, val_acc = model.evaluate(xval,yval, verbose=False)
    return -val_acc

def train_loss(theta, neurons, xtrain, ytrain, model):
    set_params(theta, neurons, model)
    loss, acc = model.evaluate(xtrain,ytrain, verbose=False)
    #print("loss:",loss)
    return loss

def val_loss(theta, neurons, xval, yval, model):
    set_params(theta, neurons, model)
    loss, val_acc = model.evaluate(xval,yval, verbose=False)
    return loss


def cma_training(model, neurons, theta_0, sigma , x_train, y_label_train, x_test, y_label_test,file, cma_dict={} ):
    pop_size = 4 + int(3*np.log(neurons[0]))
    seed = None
    tolfun = 1e-11
    max_iter = 100 + 150*(neurons[0]+3)**2 // (pop_size*100)**0.5
    cmean = 0.5
    mu = int(pop_size/2)
    mu_eff = (pop_size/4.0)  #pop_size/4
    rank_one = 2/neurons[0]**2
    rank_mu = min(mu_eff/neurons[0]**2 ,1-rank_one)
    popsize_factor = 1.5
   
    dict_cma = {'popsize': pop_size,    # 'popsize': '4+int(3*np.log(n))  # population size, AKA lambda, number of
                                #new solution per iteration'   n=space dimension
            'seed': None,     # 'seed': 'time  # random number seed for `numpy.random`;
                              #`None` and `0` equate to `time`, `np.nan` means "do nothing", see also option "randn"',
            'tolfun': 1e-11,    # 'tolfun': '1e-11  #v termination criterion: tolerance in function value, quite useful'
            'maxiter': max_iter,    # 'maxiter': '100 + 150 * (N+3)**2 // popsize**0.5  #v maximum number of iterations',
            'CMA_cmean': cmean,      # 'CMA_cmean':< '1  # learning rate for the mean value', ver ecuacion 9 del tutorial
            'CMA_rankmu': rank_mu,   # 'CMA_rankmu': '1.0  # multiplier for rank-mu update learning rate of covariance matrix'
                                 #ver ecuacion 30 del tutorial
                                 #  For cμ = 1, no prior information is retained.
                                 #For cμ = 0, no learning takes place, c μ ≈ min(1, μeff/n2 ) is a reasonably choice.
            'CMA_rankone': rank_one,  # 'CMA_rankone': '1.0  # multiplier for rank-one update learning rate of covariance matrix',
            'CMA_mu': mu,  # 'CMA_mu': 'None  # parents selection parameter, default is popsize // 2',
                             #ver ec. 9 del tutorial 
            
                
           }
    if len(cma_dict)>0:
        for item,val in cma_dict.items() :
            dict_cma[item] = val
           
  
    
    es = cma.CMAEvolutionStrategy(theta_0, sigma, dict_cma )  # fijas la media y la std inicial
    fbest = 1000
    acc_best_v = 0
    tacc = 0
    vacc = 0 
    i = 0
    print(dict_cma['maxiter'])
    while not es.stop():
    #while i<max_iter:
        solutions = es.ask()
    
        
        es.tell(solutions, [train_loss(theta, neurons, x_train, y_label_train, model) for theta in solutions])
        if abs(es.result.fbest) < fbest:
            fbest = abs(es.result.fbest)
        loss_val  = abs(val_loss( es.result.xbest, neurons, x_test, y_label_test, model))
        tacc = -train_acc(es.result.xbest, neurons, x_train, y_label_train, model)
        vacc =  -val_acc(es.result.xbest, neurons, x_test, y_label_test, model)
        #print(tacc,vacc)
       
        if vacc > acc_best_v:
            acc_best_v = vacc
            thetabest = es.result.xbest
            print("*******",tacc,vacc)
            np.save(file, thetabest)
        
    
        es.logger.add()  # write data to disc to be plotted
        es.disp()  
        if vacc == 1:
            break
        
        i += 1

    
    return es

def main(model_name, neurons, activations ,kfold,cma_dict, sigma ,x,y, scaler):
   
    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []
    best_score = 0
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=kfold, shuffle=True)

    fold_no = 1
    

    file = "thetabest"+model_name[0]+model_name[1]
    global_theta_best = "global_theta_best"+model_name[0]+model_name[1]

  


    for train, test in kfold.split(x, y):
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
    
       
       
        
        
        # fit scaler on training data
        norm = MinMaxScaler((-scaler,scaler)).fit(x[train])
        x[train] = norm.transform(x[train])
        x[test] = norm.transform(x[test])
        norm = StandardScaler(with_std=False).fit(x[train])
        # transform training data
        x[train] = norm.transform(x[train])
        std = np.std(x[train])
        mean = np.mean(x[train])
        print("std",np.std(x[train]))
        print("mean",np.mean(x[train]))
        

        # transform testing dataabs
        x[test] = norm.transform(x[test])
        
        
        #(x,y,layers = ['DMN', 'DMN'], neurons=[2,2,1], activations=['tanh','sigmoid'], dendrites=[]):
        #dendrites  = bkmeans.bkmeans(x[train],y[train],[int(neurons[1])],0.01)
        
        model = model_init(x[train], y[train], model_name, neurons, activations)
        weights = model.get_weights()
        #print("-->",weights)
        #weights = np.array(weights)
        theta_0 = get_theta0(weights)

    
        #def cma_training(model, neurons, theta_0, sigma , x_train, y_label_train, x_test, y_label_test,file, cma_dict={} ):
        es = cma_training(model, neurons, theta_0, std , x[train], y[train], x[test], y[test],file, cma_dict)

        print('termination:', es.stop())
        # Generate generalization metrics
        thetabest = np.load(file+".npy")
        scores = -val_acc(thetabest,neurons,x[test], y[test], model)
        if best_score < scores:
            best_score = scores
            shutil.copy(file+".npy", global_theta_best+".npy")
        
        print(f'Score for fold {fold_no}: {model.metrics_names[1]} of {scores*100}%')
        acc_per_fold.append(scores * 100)
        
        shutil.copy(file+".npy", file+"_fold"+str(fold_no)+".npy")
    

        # Increase fold number
        fold_no = fold_no + 1

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
      print('------------------------------------------------------------------------')
      print(f'> Fold {i+1} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    #print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')
    #File = 'CMA-ES.csv', DataSet,Acc, Std, Modelo, Neuronas, Activacion, Sigma, Xpopsize,kfold
    Data=[np.mean(acc_per_fold),np.std(acc_per_fold) ,model_name[0]+model_name[1],std]
    return model, Data 
