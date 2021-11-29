# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 17:59:15 2021

@author: Patrick
"""

from utils import vec_fun, split_data, my_rf
from utils import perf_metrics, open_pickle, my_pca

if __name__ == "__main__":

    base_path = "/Users/vusalbabashov/Desktop/text-analytics/"  
              
    final_data = open_pickle(base_path, "data.pkl")
    
    my_vec_text = vec_fun(final_data.body_basic, base_path)
    
    pca_data = my_pca(my_vec_text, 0.25, base_path)
    
    X_train, X_test, y_train, y_test = split_data(
        pca_data, final_data.label, 0.1)
    
    rf_model = my_rf(
        X_train, y_train, base_path)
    
    model_metrics = perf_metrics(rf_model, X_test, y_test)

    print(model_metrics)