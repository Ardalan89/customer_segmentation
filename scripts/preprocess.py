from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# custom transformer

# class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.encoder = LabelEncoder()
    
#     def fit(self, X, y=None):
#         self.encoder.fit(X['Sex'].astype(str))
#         return self
    
#     def transform(self, X):
#         X_out = X.copy()
#         X_out.loc[:,'Sex'] = self.encoder.transform(X_out['Sex'].astype(str))
#         return X_out

def make_pipeline(scale=True):
    num_features = ['Annual_Income', 'Spending_Score']

    if scale:
        transformers = [('scaler', StandardScaler(), num_features)]
    else:
        transformers = [('num_passthrough', 'passthrough', num_features)]

    preprocessor = ColumnTransformer(transformers=transformers)

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    return full_pipeline

# create the processor
full_processor = make_pipeline(scale=True)


