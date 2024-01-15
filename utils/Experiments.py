# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random
import json
from .Evaluation import evaluate

class Settings():
    '''Defines the experiment settings.
        project_dir: path in the file system to store results
        model_class:  class of the model to be built and evaluated
        trials:       number of repetitions of the same experiment
        initial_seed: seed value in first trial; seeds are then incremented by one with each trial
    '''
    def __init__(self, project_dir, model_class,
                 trials=10, window_size=30, epochs=100, batch_size=128, verbose=1,
                 classes=2, features=1, metafeatures=3, domains=2, 
                 validation_split=0.1, save_models=False, initial_seed=0):
        self.project_dir = project_dir
        self.model_class = model_class
        self.trials = trials
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.classes = classes
        self.features = features
        self.metafeatures = metafeatures
        self.domains = domains
        self.validation_split = validation_split
        self.save_models = save_models
        self.initial_seed = initial_seed

class Experiment():
    '''Conducts an experiment. Trains and evaluates with the given data.
       :param data: training and test data to be used for the experiment
       :param settings: an object of class settings
       :param save_as: name of the experiment
       :param pretrained_model_path: a path to a keras model to be loaded as base model
                                       and fine-tuned in the experiment
       :param pretrained_model_iterations_path: a path to a collection of keras models to be 
                                       loaded for fine-tuning according to the current seed value
                                       (previously saved by save_model())
    '''
    
    def __init__(self, data, settings, save_as='unnamed', 
                 pretrained_model_path=None, pretrained_model_iterations_path=None):
        self.data = data
        self.settings = settings
        self.save_as = save_as
        self.pretrained_model_path = pretrained_model_path
        self.pretrained_model_iterations_path = pretrained_model_iterations_path
        
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
    def save_results(self, results, history, seed):
        '''Saves the results of an experiment'''
        path = self.settings.project_dir + "/" + self.save_as
        if not os.path.exists(path + "/training_history/"):
            os.mkdir(path)
            os.mkdir(path + "/training_history/")
         # save training history
        np.save(path + "/training_history/" + "history_" + str(seed), history.history)
         # save evaluation results    
        evaluation_file_path = path + "/" + "results.json"
        if os.path.exists(evaluation_file_path): # if there are any previous results, extend these
            with open(evaluation_file_path, "r") as f:
                previous_results = json.load(f)
                previous_results.extend(results)
                results = previous_results
                print("evaluation results are updated in", evaluation_file_path)
        for i in results:       # convert values to floats before dumping
            for j in i.items():
                 i[j[0]] = float(j[1])
        with open(evaluation_file_path, "w") as f:
            json.dump(results, f)
            
    def save_model(self, model, iteration=0):
        '''Saves a trained model during the experiment'''
        path = self.settings.project_dir + "/" + self.save_as
        if not os.path.exists(path + "/models/"):
            os.mkdir(path)
            os.mkdir(path + "/models/")
        model.save(path + "/models/model_" + str(iteration))
        
            
    def callbacks(self):
        '''Applies early stopping'''
        return [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', 
                                            patience=5, min_delta=0.00001,
                                            verbose=1, restore_best_weights=True)]
        
    def run(self):
        for i in range(self.settings.initial_seed, self.settings.initial_seed + self.settings.trials):
            print("Seed", i)
            self.set_seed(i)
            
            if self.pretrained_model_iterations_path != None:
                self.pretrained_model_path = self.pretrained_model_iterations_path + "/model_" + str(i)
                print("pretrained model:", self.pretrained_model_path)
            if self.pretrained_model_path != None:
                model = tf.keras.models.load_model(self.pretrained_model_path)
            else:
                model = self.settings.model_class(classes=self.settings.classes, 
                                                  features=self.settings.features,
                                                  metafeatures=self.settings.metafeatures,
                                                  window_size=self.settings.window_size,
                                                  batch_size=self.settings.batch_size)
                
           
            history = model.fit(self.data['x_train'], self.data['y_train'],
                                 epochs=self.settings.epochs, batch_size=self.settings.batch_size,
                                 validation_split=self.settings.validation_split,
                                 verbose=self.settings.verbose,
                                 callbacks=self.callbacks())
            try:
                results = [evaluate(model, self.data['x_test'], self.data['y_test'], verbose=2)]
                self.save_results(results, history, i)
            except:
                print("evaluation failed")
            
            if self.settings.save_models:
                self.save_model(model, i)
            
            
def read_all(project_list):
    '''Reads the results from a list of project directories and returns them in a DataFrame.'''
    df = pd.DataFrame()
    for project_dir in project_list:
        projects = []
        files = []
        for project in [d for d in os.listdir(project_dir) if ((os.path.isdir(project_dir + '/' + d)) \
                                                               & (not "." in d))]:
            projects.append(project)
            path = project_dir + '/' + project
            dirs = [d for d in os.listdir(path) if ((os.path.isdir(path + '/' + d)) & (not "." in d))]
            for d in dirs:
                for f in [f for f in list(np.concatenate([os.listdir(path + '/' + d)]).flat) if '.json' in f]:
                    files.append(project_dir + '/' + project + '/' + d + '/' + f)
        print(len(projects), "subprojects")
        print(len(files), "experiments")

        dfp = pd.DataFrame()
        for f in files:
            with open(f) as json_file:
                df0 = pd.DataFrame.from_dict(json.load(json_file))
                df0['Path'] = '/'.join(f.split('/')[:-1])
                df0['Dataset'] = df0['Path'].apply(lambda x: x.split('/')[-2])
                df0['Method'] = df0['Path'].apply(lambda x: x.split('/')[-1])
                df0['Days'] = df0['Path'].apply(lambda x: int(x.split('experiments_')[1].split('d')[0]))
                df0['Iteration'] = df0.index
                dfp = dfp.append(df0)

        projects = []
        files = []
        dfh = pd.DataFrame()
        for project in [d for d in os.listdir(project_dir) if ((os.path.isdir(project_dir + '/' + d)) \
                                                               & (not "." in d))]:
            projects.append(project)
            path = project_dir + '/' + project
            dirs = [path + '/' + d + '/training_history' for d in os.listdir(path) \
                    if ((os.path.isdir(path + '/' + d)) & (not "." in d))]
            for d in dirs:
                    try:
                        for f in [d + '/' + f for f in os.listdir(d) if not ".ipynb" in f]:
                            history = np.load(f, allow_pickle='TRUE').item()
                            row = {}
                            row['Path'] = f.split('/training_history')[0]
                            row['Iteration'] = f.split('_')[-1].split('.')[0]
                            row['Epochs'] = len(history['loss'])
                            row['loss_diff_5_avg'] = round(sum(abs(np.array(history['val_loss'][-5:]) \
                                                                   - np.array(history['loss'][-5:]))) / 5, 3)
                            row['loss_diff'] = round(abs(np.array(history['val_loss'][-1]) \
                                                         - np.array(history['loss'][-1])), 3)
                            dfh = dfh.append(row, ignore_index=True)
                            dfh['Iteration'] = dfh['Iteration'].astype('int')
                    except:
                        pass

        dfp = dfp.merge(dfh, how='left', on=['Path', 'Iteration'])
        df = pd.concat([df, dfp], ignore_index=True)
        
        print(">", len(df), df.Days.unique())
    return df