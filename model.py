import eli5 as eli5
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline

from loadData import DataLoader
from preprocess import CustomTransformer
from config import CFG

class Model:
    @staticmethod
    def preprocess(X, y):
        X = CustomTransformer().fit_transform(X)
        y = y.apply(lambda x: 1 if x >= 4 else 0)
        return X, y

    @staticmethod
    def model_pipeline():
        vect = TfidfVectorizer()
        clf = SGDClassifier(random_state=42, n_jobs=-1, class_weight="balanced", penalty="elasticnet")
        model = Pipeline([('vect', vect), ('clf', clf)])
        return model

    @staticmethod
    def find_params(model, X, y):
        search = RandomizedSearchCV(model, param_distributions=CFG.get('grid_search'), return_train_score=True,
                                    cv=KFold(5, shuffle=True, random_state=42), verbose=1,
                                    scoring='roc_auc', n_jobs=-1)
        search.fit(X, y)
        return search.best_params_

    @staticmethod
    def model():
        vect = TfidfVectorizer(norm="l2", ngram_range=(1, 2), max_df=1.0)
        clf = SGDClassifier(random_state=42, n_jobs=-1, class_weight="balanced", penalty="elasticnet",
                            loss="modified_huber", l1_ratio=0.1, alpha=1e-05)
        model = Pipeline([('vect', vect), ('clf', clf)])
        return model

    def run(self, X, y, X_test, y_test):
        model = self.model()
        model.fit(X, y)
        preds = cross_val_predict(model, X, y,
                                  cv=StratifiedKFold(5), n_jobs=-1,
                                  method='predict_proba')
        predds = model.predict_proba(X_test)
        DataLoader().save_data_csv(pd.DataFrame(predds[:, 1]), 'predds.csv')
        print('Test: ', roc_auc_score(y_test, predds[:, 1]))
        print('Train: ', roc_auc_score(y, preds[:, 1]))
        # eli5.show_weights(model, top=20)

