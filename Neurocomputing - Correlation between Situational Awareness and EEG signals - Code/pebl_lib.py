import h5py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA, PCA
from scipy import signal
from sklearn.mixture import GaussianMixture
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier



class PeblInput:
    """ Reads and formats csv file containing PEBL output
    
    1) Read the data
    2) Define start time of the experiment and length (needed for time sync
       with EEG input)
    3) Split data by task (level1 SA, level2 SA, level3 SA)
    
    Input: data location of csv file
    Output: start_time, end_time, level
    """
    
    def __init__(self, data_location):
        self._read_data(data_location)
        self._boundaries()
        self.level = {}
        self.level.update({1: self._levels("Summary", [1,5,6,12,13])})
        self.level.update({2: self._levels("Click2", [1,5,6,12,13])})
        self.level.update({3: self._levels("ANGLE", [1,5,6,14,15])})
        
    def _read_data(self, data_location):
        # Hack, because the columns have different lengths
        # maximum amount of entries is 17
        # gives column names from 0 to 16
        column_names = [i for i in range(17)]
        # Read the data
        self.raw = pd.read_csv(data_location, header=None, names=column_names)
        
    def _boundaries(self):
        a = self.raw[1][0].split()
        self.start_time = int(a[3].replace(':', ''))*1000 - self.raw[0][0]
        # Get rid of seconds between 59 and 99
        if (self.start_time % 100000 > 60000):
            self.start_time = self.start_time - 40000
        lastrow = self.raw.iloc[-1] 
        self.end_time = int(lastrow[14]) if lastrow[8] == "ANGLE" \
                                        else int(lastrow[12])
        
    def _levels(self, column_name, column_index):
        data = self.raw[self.raw[8] == column_name][column_index] \
                   .astype('float64')
        data = data.T.reset_index(drop=True).T
        data.reset_index(inplace=True, drop=True)
        return data
    

class DoubleGaussian:
    """ Apply double gaussian to dataset """
    
    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            self.data = data[2]        
        else:
            self.data = data
            
    def fit(self, plot=False, bins=20):
        """ Apply double gaussian to dataset and get the threshold (intersection)
        
        Input: optional argument: plot=False, bins=20
        Output: gmm, gmm.fit, threshold optional: plot
        """
        self.gmm = GaussianMixture(n_components = 2, tol=0.0001)
        self.gmm.fit(np.expand_dims(self.data, 1))
        self._find_threshold()
        if plot:
            self._plot_gaussian(bins)
        
    def _plot_gaussian(self, bins):
        plt.figure()
        x = np.linspace(min(self.data), max(self.data), 2000)
        n = 1
        for mu, sd, p in zip(self.gmm.means_.flatten(), 
                             np.sqrt(self.gmm.covariances_.flatten()), 
                             self.gmm.weights_):
            
            g_s = stats.norm(mu, sd).pdf(x) * p
            label = 'gaussian ' + str(n)
            plt.plot(x, g_s, label=label)
            n = n+1
        sns.distplot(self.data, bins=bins, kde=False, norm_hist=True)
        #gmm gives log probability, hence the exp() function
        gmm_sum = np.exp([self.gmm.score_samples(e.reshape(-1, 1)) for e in x]) 
        plt.plot(x, gmm_sum, label='gaussian mixture')
        plt.legend()

    def _find_threshold(self):
        mu = self.gmm.means_.flatten()
        std = np.sqrt(self.gmm.covariances_.flatten())
        p = self.gmm.weights_.flatten()
        intersections = np.roots([
            1/(2*std[0]**2) - 1/(2*std[1]**2),
            mu[1]/(std[1]**2) - mu[0]/(std[0]**2),
            mu[0]**2 /(2*std[0]**2) - mu[1]**2 / (2*std[1]**2) 
            - np.log((std[1]*p[0]) / (std[0]*p[1]))
        ])
        self.threshold = np.nan
        for i in range(len(intersections)):
            if min(self.data) <= intersections[i] <= max(self.data):
                self.threshold = intersections[i]

                
class PeblAll:
    """Do all the work with the PEBL data:
    1) Read all files in data/#/#/satest-#.csv
    2) Split it by level
    3) Add all values per level in one list
    4) Apply Double Gaussian classifier
    5) Identify threshold between good and bad SA
    6) Optional: Plot Histogram with Double Gaussian
    
    Input: Optional: nr_trials(=32), plot(=False), bins(=20)
    Output: .participant[#].level[#], .threshold[#]
    """

    def __init__(self, nr_trials=32, plot=False, bins=20):
        self.threshold = {}
        
        self.participant = [PeblInput(
            'data/' + str(i) + '/' + str(i) + '/' + 'satest-' + str(i) + '.csv')
            for i in range(1,nr_trials+1)]
        self.threshold.update({1: self._compute_threshold(1, plot=plot, bins=bins)})
        self.threshold.update({3: self._compute_threshold(3, plot=plot, bins=bins)})
        for i in range(1,4):
            self._define_labels(i)

        
    def _compute_threshold(self, nr_level, plot, bins):    
        pebl_values = []
        
        for thispebl in self.participant:
            pebl_values += (list(thispebl.level[nr_level][4].values))

        dg = DoubleGaussian(pebl_values)
        dg.fit(plot=plot, bins=bins)
        
        return dg.threshold
        
    def _define_labels(self, nr_level):
        if (nr_level == 2):
            for thispebl in self.participant:
                thislabel = []
                for i in range(0,len(thispebl.level[nr_level]),2):
                    if((thispebl.level[nr_level][4][i] == 0) 
                       or (thispebl.level[nr_level][4][i+1] == 0)):
                        # Double appending, as always two trials are treated as one
                        thislabel.append(False)
                        thislabel.append(False)
                    else:
                        # Double appending, as always two trials are treated as one
                        thislabel.append(True)
                        thislabel.append(True)
                thispebl.level[nr_level][5] = thislabel
        else:
            for thispebl in self.participant:
                thispebl.level[nr_level][5] = \
                    (thispebl.level[nr_level][4] < self.threshold[nr_level])  
      
    
class EegInput:
    """ Reads and formats mat file containing EEG output
    
    1) Read the data
    2) Truncate the data according to start time and end time of PEBL experiment
    3) Extract timestamp and 32 channels from data
    
    Input: data location of mat files (2x), start_time, end_time
    Output: timestamps, data
    """
        
    def __init__(self, data_location_eeg, data_location_time, start_time, 
                 end_time):
        
        raw = self._read_data(data_location_eeg)
        raw_time = self._read_data(data_location_time)
        truncated = self._truncate(raw, raw_time, start_time, end_time)
        self._extract_channels(truncated)
        
    @staticmethod
    def _read_data(data_location):
        mat = h5py.File(data_location, 'r')
        mat = {k: np.array(v) for k, v in mat.items()}
        raw = pd.DataFrame(data=mat['y'], index=range(len(mat['y'])))
        return raw
    
    @staticmethod
    def _truncate(raw, raw_time, start_time, end_time):
        timestamps = raw_time[4] * 10000000 \
                     + raw_time[5] * 100000 \
                     + raw_time[6] * 1000
        truncated = raw.iloc[(abs(timestamps - start_time)).idxmin():]
        truncated = truncated.iloc[
            :(abs(truncated[0] - end_time / 1000)).idxmin()]
        truncated.reset_index(inplace=True, drop=True)
        return truncated
    
    def _extract_channels(self, truncated):
        self.timestamps = truncated[0]
        self.timestamps = (self.timestamps - self.timestamps[0])*1000
        self.data = truncated.T.loc[1:32].reset_index(drop=True).T    
        
        
class SliceData:
    """ Slices the EEG data and timestamp in portions and adds the label
    
    Input: pebl data, eeg data
    Output: .level[#], containing a dictionary per slice with all data
    """
    
    def __init__(self, pebl, eeg, participant_nr):
        self.participant_nr = participant_nr
        self.level = {}
        
        self.level.update({1: self._slice_data(pebl.level[1], eeg, 1)})
        self.level.update({2: self._slice_data(pebl.level[2], eeg, 2)})
        self.level.update({3: self._slice_data(pebl.level[3], eeg, 3)})
        
        
    def _slice_data(self, pebl, eeg, level):
        alldata = []
        for i in range(len(pebl)):
            if ((level == 2) and (i % 2 == 0)):
                continue
                
            eeg_data = eeg.data.iloc[
                (abs(eeg.timestamps - pebl[1][i])).idxmin()
                :(abs(eeg.timestamps - pebl[2][i])).idxmin()
            ]
            eeg_timestamps = eeg.timestamps.iloc[
                (abs(eeg.timestamps - pebl[1][i])).idxmin()
                :(abs(eeg.timestamps - pebl[2][i])).idxmin()
            ]
            this_event = {
                'participant_nr': self.participant_nr,
                'eeg_data': eeg_data, 
                'eeg_timestamps': eeg_timestamps,
                'level': level,
                'label': pebl[5][i],
                'block': pebl[0][i]
            }
            # Remove the tutorial blocks:
            if (pebl[0][i] != level):
                alldata.append(this_event)
        
        return alldata
    
class EegAllPreparing:
    """Do all the preparation work with the EEG data:
    1) Read all files in data/#/test_data_#.mat
    2) Read all files in data/#/time_#.mat
    2) Truncate the data
    3) Extract the data channels
    4) Slice the data in portions of a certain length before
       the screen blanks (default: 5 seconds)
    5) Build dictionary per slice
    
    Input: pebl, Optional: nr_trials(=32)
    Output: .participant[#].level[#]
    """

    def __init__(self, pebl, nr_trials=32):
        data = [EegInput(
            'data/' + str(i) + '/' + 'test_data_' + str(i) + '.mat',
            'data/' + str(i) + '/' + 'time_' + str(i) + '.mat',
            pebl.participant[i-1].start_time,
            pebl.participant[i-1].end_time)
            for i in range(1,nr_trials+1)]
        self.participant = [SliceData(pebl.participant[i], data[i], i+1)
                             for i in range(len(data))]
    

class ProcessEeg:
    """ Includes different methods to process the truncated EEG signals. """
    
    def __init__(self, eegdata):
        self._scale(eegdata)
        
    def _scale(self, data):
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(data)

    def ica(self, n_components=32):
        """ ICA for EEG data
        
        Input: Optional argument n_components (default: 32)
        Output: sources S, mixing matrix A
        """
        
        self.ica = FastICA(n_components=n_components)
        self.ica.fit(self.X)
    
    def pca(self, n_components=32):
        """ PCA for EEG data
        
        Input: Optional argument n_components (default: 32)
        Output: sources H, principal components pca_comp
        """
        
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.X)
    
class PowerSpectrum:
    def __init__(self, eegdata, fs=250, plot=False):
        import numpy as np

        self.fs = fs                              # Sampling rate (250 Hz)
        self.data = eegdata
        # Define EEG bands
        self.eeg_bands = {'Delta': (0, 4),
                     'Theta': (4, 8),
                     'Alpha': (8, 12),
                     'Beta': (12, 30),
                     'Gamma': (30, 45)}
        self._get_bands()
        if(plot):
            self._plot_bands()
        self.psr = self.eeg_band_fft['Beta']/(self.eeg_band_fft['Theta'] \
                                              + self.eeg_band_fft['Alpha'])

    def _get_bands(self):
        # Get real amplitudes of FFT (only in postive frequencies)
        fft_vals = np.absolute(np.fft.rfft(self.data))

        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(self.data), 1.0/self.fs)

        # Take the mean of the fft amplitude for each EEG band
        self.eeg_band_fft = dict()
        self.eeg_freqs = dict()
        for band in self.eeg_bands:  
            freq_ix = np.where((fft_freq >= self.eeg_bands[band][0]) & 
                               (fft_freq <= self.eeg_bands[band][1]))[0]
            self.eeg_band_fft[band] = np.mean(fft_vals[freq_ix])

    def _plot_bands(self):
        # Plot the data (using pandas here cause it's easy)
        import pandas as pd
        df = pd.DataFrame(columns=['band', 'val'])
        df['band'] = self.eeg_bands.keys()
        df['val'] = [self.eeg_band_fft[band] for band in self.eeg_bands]
        ax = df.plot.bar(x='band', y='val', legend=False)
        ax.set_xlabel("EEG band")
        ax.set_ylabel("Mean band Amplitude")
        
from scipy import signal
class FeatureExtractor(object):
    def __init__(self, data, spat_structures):
        self.data = data
        self.struct = spat_structures
        self.fft_features = pd.DataFrame()
        self.psd_features = pd.DataFrame()
        
    def extract_features(self):
        for i, person in enumerate(self.data.participant):
            self.participant = i
            for level, data in person.level.items():
                self.level = level
                for j, sample in enumerate(data):
                    self.sample = j
                    self.label = sample['label'] 
                    eeg_data = sample['eeg_data']
                    self._append_features(eeg_data)
    
    def _append_features(self, eeg_data):
        eeg_data_scaled = self.struct['scaler'].transform(eeg_data)
        pca_ts = self.struct['pca'].transform(eeg_data_scaled)
        ica_ts = self.struct['ica'].transform(eeg_data_scaled)
        
        pca_f = self._get_spectral_features(pca_ts, 'PCA')
        ica_f = self._get_spectral_features(ica_ts, 'ICA')
        ts_f = self._get_spectral_features(eeg_data_scaled, 'TS')
        
        self.fft_features = pd.concat([self.fft_features, \
                        self._combine(pca_f['fft'], ica_f['fft'], ts_f['fft'])],
                        axis=1)
        self.psd_features = pd.concat([self.psd_features, \
                        self._combine(pca_f['psd'], ica_f['psd'], ts_f['psd'])],
                        axis=1)
        print(str(self.participant) + " " + str(self.level) + " " + str(self.sample))
        
    def _get_spectral_features(self, signal, key):
        fft_bands = pd.DataFrame()
        psd_bands = pd.DataFrame()
        for n, thissignal in enumerate(signal.T):
            thiskey = key + '_' + str(n)
            thisfft = self._do_fft(thissignal)
            fft_bands = pd.concat([fft_bands, self._get_bands(thisfft, thiskey)], axis=1)
            
            thispsd = self._do_psd(thissignal)
            psd_bands = pd.concat([psd_bands, self._get_bands(thispsd, thiskey)], axis=1)
        return {'fft': fft_bands, 'psd': psd_bands}
    
    @staticmethod
    def _do_fft(signal, fs=250):
        fft = np.fft.fft(signal)
        realfft = np.sqrt(fft.real**2 + fft.imag**2)
        n = len(signal)
        timestep = 1/fs
        freq = np.fft.fftfreq(n, d=timestep)
        return pd.DataFrame(realfft, index=freq)
    
    @staticmethod
    def _do_psd(data):
        f, pxx_den = signal.periodogram(data.T, fs=250, 
                                        window='flattop')
        return pd.DataFrame(pxx_den.T, index=f)
    
    @staticmethod
    def _get_bands(data, key):
        data = data.reset_index()
        eeg_bands = {'Delta': (0, 4),
                     'Theta': (4, 8),
                     'Alpha': (8, 12),
                     'Beta': (12, 30),
                     'Gamma': (30, 45)}
        out_val = pd.DataFrame()
        for band in eeg_bands.values():
            out_val = pd.concat([out_val,
                                 data.loc[abs((band[0] - data['index'])).idxmin():
                                          abs((band[1] - data['index'])).idxmin()].sum()], 
                                axis=1)
        out_val.columns = [val + '_' + key for val in eeg_bands.keys()]
        out_val['Ratio_' + key] = out_val['Beta_' + key] / (out_val['Theta_' + key]+\
                                              out_val['Alpha_' + key])
        return out_val.drop('index')
    
    def _combine(self, pca_f, ica_f, obs_f):
        thisfeature = pd.concat([pca_f, ica_f, obs_f], axis=1).T        
        thisfeature.loc['Level'] = self.level
        thisfeature.loc['Participant'] = self.participant
        thisfeature.loc['Trial'] = self.sample
        thisfeature.loc['Label'] = self.label
        return thisfeature
    
class Outlier():
    def __init__(self, features, technique, removal):
        self.newfeatures = pd.DataFrame()
        self.technique = technique
        for feature in features[technique]:
            if feature == 'Level':
                break
            self.data = np.array(features[self.technique][feature]).reshape(-1,1)
            self.pred = self._isolation_tree()
            self.newdata = self._outlier_removal(removal)
            self.newfeatures = pd.concat([self.newfeatures, \
                pd.DataFrame(self.newdata.T, columns=[feature])], \
                                         axis = 1)
    
    def _isolation_tree(self):
        clf = IsolationForest()
        clf.fit(self.data)
        pred = clf.predict(self.data)
        return pred
    
    def _outlier_removal(self, removal):
        if (removal == '-999'):
            newdata = np.where(self.pred == 1, self.data.T, -999)
        if (removal == '0'):
            newdata = np.where(self.pred == 1, self.data.T, 0)
        if (removal == 'nan'):
            newdata = np.where(self.pred == 1, self.data.T, np.nan)
        if (removal == 'max'):
            newdata = np.where(self.pred == 1, self.data.T, np.nan)
            maxvalue = np.nanmax(newdata.T)
            newdata = np.where(np.isnan(newdata), maxvalue, newdata)
        if (removal == 'mean'):
            newdata = np.where(self.pred == 1, self.data.T, np.nan)
            meanvalue = np.nanmean(newdata.T)
            newdata = np.where(np.isnan(newdata), meanvalue, newdata)
        return newdata
    
class BoostedTrees(object):
    def __init__(self, removed):
        self._boosted_tree(removed)
                                
    def _boosted_tree(self, removed):
        self.allresults = {}
        removals = ['max', '0', 'nan', 'mean', 'raw']
        for technique in ['fft', 'psd']:
            for removal in removals:
                results = pd.DataFrame()
                test_case = technique + '_' + removal
                print(test_case)
                self.X_train, self.X_test, self.y_train, self.y_test \
                    = self._load_and_split(removed, test_case)
                for i in range(64):
                    params = self._get_params(i)
                    this_result = self._exec_tree(params)
                    results  = pd.concat([results, \
                        this_result], axis=1)
                results = results.T.reset_index(drop=True)
                self.allresults.update({test_case: results})
                
        
    @staticmethod
    def _load_and_split(removed, test_case, test_size=0.2):
        X = removed[test_case]
        y = removed['label']
    
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    def _exec_tree(self, params):
        clf = XGBClassifier(seed=42)
        clf.set_params(**params)
        clf.fit(self.X_train, self.y_train)
        y_pred_rf = clf.predict(self.X_test)
        y_pred_train = clf.predict(self.X_train)
        this_result = pd.DataFrame.from_dict(params, orient='index', columns=[0])
        this_result.loc['Train Accuracy'] = accuracy_score(self.y_train, y_pred_train)
        this_result.loc['Test Accuracy'] = accuracy_score(self.y_test, y_pred_rf)
        this_result.loc['Precision'] = precision_score(self.y_test, y_pred_rf)
        this_result.loc['Recall'] = recall_score(self.y_test, y_pred_rf)
        return this_result
    
    @staticmethod
    def _get_params(i):
        print(i)
        # decode the 64 options, so that every combination is checked
        o = int(i%4)
        p = int(((i-o)/4)%4)
        q = int((((i-o)/4)-p)/4)
        learning_rate = [0.01, 0.05, 0.1, 0.2]
        max_depth = [1,2,3,4]
        n_estimators = [50,100,200,400]
        params = {"learning_rate": learning_rate[o],
                  "max_depth": max_depth[p], 
                  "n_estimators": n_estimators[q]}
        return params
    
class RandomForest(object):
    def __init__(self, removed):
        self._random_forest(removed)
                                
    def _random_forest(self, removed):
        self.allresults = {}
        removals = ['max', '0', 'mean', 'raw']
        for technique in ['fft', 'psd']:
            for removal in removals:
                results = pd.DataFrame()
                test_case = technique + '_' + removal
                print(test_case)
                self.X_train, self.X_test, self.y_train, self.y_test \
                    = self._load_and_split(removed, test_case)
                this_result = self._exec_tree()
                results  = pd.concat([results, \
                        this_result], axis=1)
                self.allresults.update({test_case: results})    
        
    @staticmethod
    def _load_and_split(removed, test_case, test_size=0.2):
        X = removed[test_case]
        y = removed['label']
    
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    def _exec_tree(self,):
        clf = RandomForestClassifier(n_estimators=1, warm_start=True,
                                n_jobs=-1, random_state=42)
        n_trees = np.arange(10, 2000, 10)
        results = pd.DataFrame()
        for i in n_trees:
            print(i)
            clf.set_params(n_estimators=i)
            clf.fit(self.X_train, self.y_train)
            y_pred_rf = clf.predict(self.X_test)
            y_pred_train = clf.predict(self.X_train)

            results = results.append({
                'n_estimators': i, 
                'Train Accuracy': accuracy_score(self.y_train, y_pred_train),
                'Test Accuracy': accuracy_score(self.y_test, y_pred_rf),
                'Precision': precision_score(self.y_test, y_pred_rf),
                'Recall': recall_score(self.y_test, y_pred_rf)},
                ignore_index=True)
        
        return results