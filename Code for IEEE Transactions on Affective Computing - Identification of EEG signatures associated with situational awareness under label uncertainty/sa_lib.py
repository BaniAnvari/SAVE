import pickle

import numpy as np
import pandas as pd

from scipy import signal
from sklearn.pipeline import Pipeline

    
class FeatureExtractor(object):
    """
    Extracts the features corresponding to single EEG sample.
    
    Inputs:
    - eeg_data - np.array where column correspond to one of 32 channels and rows are time instances
    
    Output:
    Returns the pd.DataFrame containing the spectral features for each channel measurement, and each PCA and ICA projection
    
    """
    
    def __init__(self):
        # Load ICA and PCA components
        with open('spatial_structures.pkl', 'rb') as handle:
            self.struct = pickle.load(handle)
        # Initialise the pd.DataFrame that will contain spectral features
        self.features = pd.DataFrame()
    
    # Required for sklearn Pipeline
    def fit():
        pass
    
    # Main method called to extract features
    def transform(self, eeg_data):                             
        return self._get_features(eeg_data)
                                                
    def _get_features(self, eeg_data):
        # Subtract pre-defined mean and rescale data using pre-defined scale
        eeg_data_scaled = self.struct['scaler'].transform(eeg_data)
        
        # Project data onto PCA and ICA components
        pca_ts = self.struct['pca'].transform(eeg_data_scaled)
        ica_ts = self.struct['ica'].transform(eeg_data_scaled)
        
        # Get spectral features corresponding to PCA, ICA, and each channel timeseries
        pca_f = self._get_spectral_features(pca_ts, 'PCA')
        ica_f = self._get_spectral_features(ica_ts, 'ICA')
        ts_f = self._get_spectral_features(eeg_data_scaled, 'TS')
        
        return pd.concat([pca_f, ica_f, ts_f])
                                  
    def _get_spectral_features(self, data, key):
        # Use periodogram to calculate PSD
        psd = self._get_psd(data)
        # Group PSD into 5 distinct bands
        features = self._get_bands(psd)
        return self._flatten(features, key)
        
    @staticmethod
    def _get_psd(data):
        f, pxx_den = signal.periodogram(data.T, fs=250, 
                                        window='flattop')
        return pd.DataFrame(pxx_den.T, index=f)
    
    @staticmethod
    def _get_bands(data):
        eeg_bands = {'Delta': (0, 4),
                     'Theta': (4, 8),
                     'Alpha': (8, 12),
                     'Beta': (12, 30),
                     'Gamma': (30, 45)}
        out_val = pd.DataFrame()
        for band in eeg_bands.values():
            out_val = pd.concat([out_val,
                                 data.loc[band[0]:band[1]].sum()], 
                                axis=1)
        out_val.columns = eeg_bands.keys()
        # Include also the ratio of selected bands
        out_val['Ratio'] = out_val['Beta'] / (out_val['Theta']+\
                                              out_val['Alpha'])
        # Apply logarithmic transformation
        return np.log(out_val)
        
    @staticmethod
    def _flatten(features, key):
        features.columns = [val + ' ' + key for val in \
                                features.columns]
        return features.stack()
    

class SAModel(object):
    """
    Outputs label associated with high or low SA.
    
    Inputs:
    - features - spectral features obtained using FeatureExtractor
    
    Output:
    Binary value where 0 corresponds to low SA and 1 is high SA
    
    """    
    def __init__(self):
        # Load model
        with open('rf_model_final.pkl', 'rb') as handle:
            self.model = pickle.load(handle)

        # Unpack the variable into Random Forest model and Standard Scaler
        self.scaler = self.model['scaler']
        self.model = self.model['model']
    
    # Required for sklearn Pipeline
    def fit():
        pass
    
    # Required for sklearn Pipeline
    def transform():
        pass
            
    def predict(self, features):
        # Ensure that variable is in correct format for prediction
        if isinstance(features, pd.Series):
            features = features.values
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Subtract pre-defined mean and rescale data using pre-defined scale
        features = self.scaler.transform(features)
        # Return prediction
        return self.model.predict(features)
    
# Combination of FeatureExtractor and SAModel
SAPipeline = Pipeline([('feature_extraction', FeatureExtractor()), 
                       ('SA_model', SAModel())])
        