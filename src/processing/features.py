import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
    

class MutualInfoSelector(BaseEstimator, TransformerMixin):
    """
    Select features using mutual information
    """
    def __init__(self, k=10, categorical_features=None):
        """
        Parameters:
        -----------
        k : int
            Number of top features to select
        categorical_features : list of feature names or None
            String names of categorical columns in the transformed data
            If None, will be determined automatically
        """
        self.k = k
        self.categorical_features = categorical_features
        self.selector_ = None
        self.feature_names_ = None
    
    def fit(self, X, y):
        """
        Fit the selector on training data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data (already preprocessed/encoded)
        y : array-like, shape (n_samples,)
            Target values
        """
        # Determine which columns are categorical
        if self.categorical_features is None:
            # Assume all columns are numeric if not specified
            discrete_mask = [False] * X.shape[1]
        else:
            # Create boolean mask
            # discrete_mask = [True if i in self.categorical_features else False for i in X.columns] # wrong 
            # treating the encoded variables as False 

            # adjust boolean masking logic again 

            # check if the given categorical features is in each feature of X not the other way around 

            #naive 
            # masking = []
            # for i in X.columns: 
            #     status = []
            #     for j in self.categorical_features: 
            #         if j in i: 
            #             status.append(True)
            #         else: 
            #             status.append(False)
            #     if np.sum(status) == 0: 
            #         masking.append(True)
            #     else: 
            #         masking.append(False)

            # maybe dont use a dynamic list use an nparray 

            masking = np.zeros(X.shape[1], dtype=bool)

            print(X.columns)

            for i in range(X.shape[1]): 

                feature = X.columns[i]

                status = [True if j in feature else False for j in self.categorical_features]

                if np.sum(status) == 0: 
                    masking[i] = False
                else: 
                    masking[i] = True

            # discrete_mask = [False] * X.shape[1]
            # for idx in self.categorical_features:
            # discrete_mask[idx] = True
            print(masking)
        
        # Create selector
        self.selector_ = SelectKBest(
            lambda X, y: mutual_info_classif(X, y, discrete_features=masking, random_state=42),
            k=self.k
        )
        
        # Fit selector
        self.selector_.fit(X, y)
        
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns[self.selector_.get_support()].tolist()
        else:
            self.feature_names_ = self.selector_.get_feature_names_out()
        
        return self
    
    def transform(self, X):
        """
        Transform data by selecting top k features
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
        """
        if self.selector_ is None:
            raise ValueError("Must call fit() before transform()")
        
        return self.selector_.transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """
        Get names of selected features
        """
        if self.selector_ is None:
            raise ValueError("Must call fit() before get_feature_names_out()")
        
        return self.selector_.get_feature_names_out(input_features)
    