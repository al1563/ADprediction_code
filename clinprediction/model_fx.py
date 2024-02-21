import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
import seaborn as sns
import time
import joblib
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix, roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from tableone import TableOne
import json

from .omop_fx import *
def load_in_options(ndate, dxgroup, comparison):
    odir_main = output_dir + '{}_{}v{}/'.format(ndate, dxgroup, comparison)
    with open(odir_main + 'options.json', 'r') as fp:
        options = json.load(fp)
    return options

def save_updated_options(options):
    odir_main = options['odir_main'];
    with open(odir_main + 'options.json', 'w') as fp:
        json.dump(options, fp)

def lr_model(X_train, y_train, X_test, y_test, feature_names,options,
             odir_tf = None, modelsuffix = '_', n_its = 300, 
             n_bootsample = 1000, do_sparse = True, show_fig = True,
             penalty = 'elasticnet', solver = 'saga',
             l1_ratios = [.5, 1]):
    modelkind = 'lr' + modelsuffix
    print('model: ' + modelkind)
    
    # train model
    if do_sparse:
        X_train_sparse = sparse.csr_matrix(X_train)
    start = time.time()
    print('training...')
    lr = LogisticRegressionCV(cv=options['cv'], penalty = penalty, solver = solver, 
                                  max_iter = 1e5, n_jobs = -1, l1_ratios = l1_ratios,
                                  random_state=options['random_state'])
    if do_sparse: lr.fit(X_train_sparse, y_train)
    else: lr.fit(X_train, y_train)
    print('finished. took {} minutes'.format((time.time() - start) / 60))
    
    # save features
    feat_import = pd.DataFrame(lr.coef_[0], columns = ['lr_coef'], 
                               index = feature_names)
    if odir_tf is not None: 
        feat_import.to_csv(odir_tf + '{}_model_featimport.csv'.format(modelkind))
      
    # analyze model performance
    lr_metrics = analyze_clf(lr, X_train, X_test, 
                y_train, y_test, odir_tf+modelkind, 
                dxgroup, comparison, show_fig = show_fig)
    print(lr_metrics)
    
    print('bootstrapping...')
    bs_auroc, bs_auprc = bootstrap_model(lr, X_test, y_test, odir = odir_tf, 
                    model_str = modelkind, n_its = n_its, n_bootsize = n_bootsample)
      
    model_dict = {'model':lr, 'modelkind': modelkind, 
         'metrics':lr_metrics, 'bs_auroc':bs_auroc, 'bs_auprc':bs_auprc,
         'feature_names':feature_names, 'feat_import':feat_import}
    if odir_tf is not None: 
        joblib.dump(model_dict, odir_tf + '{}_model.joblib'.format(modelkind))
        
    return model_dict

def feature_context(feat_import, feature_name_info, import_col, modelkind,
                   odir_tf = None):
    # interpret top features given an importance column and feature contexts
    feat_import = feat_import.merge(feature_name_info, left_index = True, 
                                    right_index = True, how = 'left', suffixes = ('','_'))
    display(feat_import.sort_values(import_col, ascending = False))
    for ii, gg in feat_import.sort_values(import_col, ascending = False)\
            .groupby('domain_id'):
        print(ii)
        if odir_tf is not None:
            gg.to_csv(odir_tf + '{}_topconcepts{}.csv'.format(modelkind, ii))
    return feat_import

def rf_model(X_train, y_train, X_test, y_test, feature_names, options,
             odir_tf = None, modelsuffix = '_', n_its = 300, 
             n_bootsample = 1000, do_sparse = True, show_fig = True,
             do_gridsearch = True, rf_params = {'max_depth':5, 'max_features':'log2'}):
    modelkind = 'rf' + modelsuffix
    print('model: ' + modelkind)
    N_FEATURES = len(feature_names)
    
    # train model
    if do_sparse:
        X_train_sparse = sparse.csr_matrix(X_train)
    start = time.time()
    print('training...')
    if do_gridsearch:
        param_grid = {'n_estimators': [N_FEATURES, N_FEATURES*2, N_FEATURES*3],
                    'max_depth': [3, 5, 7, 9], # change
                    'ccp_alpha': [0, 0.01, 0.05], # change
                    'max_features':["sqrt","log2"]}
        scoring = ['roc_auc','balanced_accuracy','average_precision','f1_weighted']; 
        refit = scoring[0]

        rf = GridSearchCV( 
                           RandomForestClassifier(class_weight = 'balanced', 
                                random_state = options['random_state']), 
                           param_grid, cv= options['cv'], 
                           verbose=3, scoring = scoring, 
                           refit = refit, n_jobs = 50, # -1, 
                           return_train_score = True)
    else:
        rf = RandomForestClassifier(class_weight = 'balanced', 
                                n_estimators = N_FEATURES*3, 
                                max_depth = rf_params['max_depth'], 
                                max_features = rf_params['max_features'], #'log2',
                                random_state = options['random_state'])
    
    if do_sparse: rf.fit(X_train_sparse, y_train)
    else: rf.fit(X_train, y_train)
    print('finished. took {} minutes'.format((time.time() - start) / 60))
    
    if do_gridsearch: 
        print('best params:', rf.best_params_)
        rf2 = rf; rf = rf2.best_estimator_;
    
    # save features
    feat_import = pd.DataFrame(rf.feature_importances_, columns = ['rf_import'], 
                               index = feature_names)
    if odir_tf is not None: 
        feat_import.to_csv(odir_tf + '{}_model_featimport.csv'.format(modelkind))
      
    # analyze model performance
    rf_metrics = analyze_clf(rf, X_train, X_test, 
                y_train, y_test, odir_tf+modelkind, 
                dxgroup, comparison, show_fig = show_fig)
    print(rf_metrics)
    
    print('bootstrapping...')
    bs_auroc, bs_auprc = bootstrap_model(rf, X_test, y_test, odir = odir_tf, 
                    model_str = modelkind, n_its = n_its, n_bootsize = n_bootsample)
      
    model_dict = {'model':rf, 'modelkind': modelkind, 
         'metrics':rf_metrics, 'bs_auroc':bs_auroc, 'bs_auprc':bs_auprc,
         'feature_names':feature_names, 'feat_import':feat_import}
    if do_gridsearch: 
        model_dict['model_grid'] = rf2
        model_dict['grid_params'] = {'param_grid':param_grid,
                                     'scoring':scoring, 'refit':refit}
    if odir_tf is not None: 
        joblib.dump(model_dict, odir_tf + '{}.joblib'.format(modelkind))
        
    return model_dict

# function plots ROC/PRC curves and returns the area under the curves
def analyze_clf(clf, X_train, X_test, y_train, y_test, odir, dxgroup, comparison, show_fig = False):
    """
    clf_metrics = analyze_clf(clf, X_train, X_test, y_train, y_test, odir, show_fig)
    inputs: 
        clf: classifier
        X_train, X_test, y_train, y_test: data + labels
        odir: output directory to save curves, matrix, and metrics.
        dxgroup: label of group with label "1"
        comparison: label of control or comparison group
        show_fig: (default = False) show figures?
    outputs:
        clf_metrics: dictionary of metrics. includes train/test each of 
                     auroc, auprc, accuracy_score, balanced_accuracy_score, 
                     f1_score, precision_recall_fscore_support
    """

    clf_metrics = dict() 
    fig, ax = plt.subplots(2, 2, figsize = (12,6))
    roc_plot = plot_roc_curve(clf, X_train, y_train, ax = ax[0,0])
    prc_plot = plot_precision_recall_curve(clf, X_train, y_train, ax = ax[0,1])
    clf_metrics['train_auroc'] = roc_plot.roc_auc
    clf_metrics['train_auprc'] = prc_plot.average_precision
    
    roc_plot = plot_roc_curve(clf, X_test, y_test, ax = ax[1,0])
    prc_plot = plot_precision_recall_curve(clf, X_test, y_test, ax = ax[1,1])
    clf_metrics['test_auroc'] = roc_plot.roc_auc
    clf_metrics['test_auprc'] = prc_plot.average_precision
    
    if odir is not None:
        plt.savefig(odir + 'clf_traintestcurve.png', bbox_inches = 'tight', format = 'png')
    if show_fig: plt.show()
    else: plt.close()
    
    return clf_metrics

# for a trained model, bootstrap test data samples and get a distribution for AUROC/AURPC
def bootstrap_model(clf, X_test, y_test, odir = None, model_str = 'clf', 
                    n_its = 300, n_bootsize = 1000, figsize = (3,5)):
    """
    bs_auroc, bs_auprc = bootstrap_model(clf, X_test, y_test, odir = None, 
                    model_str = 'clf', n_its = 300, n_bootsize = 1000, figsize = (3,5))

    """
    bs_auroc = list()
    bs_auprc = list()
    for i in tqdm(np.arange(n_its)):
        i_pts = np.random.choice(X_test.shape[0], n_bootsize)
        if y_test[i_pts].sum() == 0: continue;
        bs_auroc.append(roc_auc_score(y_test[i_pts], 
                                      clf.predict_proba(X_test[i_pts,:])[:, 1]))
        bs_auprc.append(average_precision_score(y_test[i_pts], 
                                      clf.predict_proba(X_test[i_pts,:])[:, 1]))
    bs_auroc = np.array(bs_auroc)
    bs_auprc = np.array(bs_auprc)

    with sns.plotting_context('poster'):
        plt.figure(figsize = figsize)
        sns.boxplot(y=bs_auroc, orient = 'v', palette = 'pastel')
        plt.ylabel('AUROC'); 
        if odir is not None:
            plt.savefig(odir + '{}_bootstrapped_AUROC.png'.format(model_str), 
                        bbox_inches = "tight")
        plt.show();

        plt.figure(figsize = figsize)
        sns.boxplot(y=bs_auprc, orient = 'v', palette = 'pastel')
        plt.ylabel('AUPRC');
        if odir is not None:
            plt.savefig(odir_tf + '{}_bootstrapped_AUPRC.png'.format(model_str), 
                       bbox_inches = "tight")
        plt.show();
                        
    return bs_auroc, bs_auprc