CFG = {
    "grid_search":
{       'clf__l1_ratio': [0, 0.1, 0.5, 1],
        'vect__ngram_range': [(1,1), (1, 2), (1, 3)],
        'vect__max_df': [0.3, 0.5, 1.0],
        'vect__norm': ['l1', 'l2'],
        'clf__alpha': [1e-7, 1e-5, 1e-3, 1e-1, 1e0, 1e1, 1e2, 1e3], # learning rate
        'clf__loss': ['modified_huber', 'hinge'], # logistic regression,
        # 'clf__penalty': ['elasticnet'],
        }
}