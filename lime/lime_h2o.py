"""
Functions for explaining classifiers that use tabular data (matrices).
"""
from __future__ import print_function
import numpy as np
import sys
from . import lime_tabular
from . import lime_base


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


class LimeH2OExplainer(lime_tabular.LimeTabularExplainer):
    """Explains predictions on tabular (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary feature
    that is 1 when the value is the same as the instance being explained."""
    def __init__(self, df, ntiles=4, discretizer=None,
                 categorical_names=None, kernel_width=3, verbose=False,
                 class_names=None, feature_selection='auto',
                 discretize_continuous=True):
        """Init function.

        Args:
            df: H2O data frame.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            discretize_continuous: if True, all non-categorical features will be
                discretized into quartiles.
        """
        self.ntiles = 4
        self.categorical_names = categorical_names
        cat_names = [x[0] for x in df.types.items() if x[1] == 'enum']
        self.categorical_features = [df.columns.index(x) for x in cat_names]
        if self.categorical_names is None:
            self.categorical_names = {}
        self.discretizer = None
        n_features = df.shape[1]
        if discretize_continuous:
            self.categorical_features = range(n_features)


        kernel = lambda d: np.sqrt(np.exp(-(d**2) / kernel_width ** 2))
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel, verbose)
        self.scaler = None
        self.class_names = class_names
        self.feature_names = df.columns
        self.scaler = Bunch(mean_=np.array(df.mean()),
                            scale_=np.array(df.sd()))
        self.feature_values = {}
        self.feature_frequencies = {}

        if verbose: print('Counting frequencies.')
        for i_feature in self.categorical_features:
            name = self.feature_names[i_feature]
            if verbose: print('Counting feature {} ({})'.format(i_feature, name))
            sys.stdout.flush()

            # if feature was discretized we know already it's values and frequencies
            # otherwise we need to compute it ourselves
            if df.types[name] != 'enum':
                self.feature_values[i_feature] = range(self.ntiles)
                self.feature_frequencies[i_feature] = np.ones(self.ntiles) / self.ntiles
            else:
                counts = df[i_feature].table().as_data_frame()
                counts.columns = ['level', 'freq']
                self.feature_values[i_feature] = list(counts.level.values)
                self.feature_frequencies[i_feature] = 1. * counts.freq.values / len(df)
            self.scaler.mean_[i_feature] = 0
            self.scaler.scale_[i_feature] = 1

        # H2O still have bugs, so altough I wanted not to change the data itself the descritizer change it.
        # For this reason I put this part at the end.
        if discretize_continuous:
            if verbose: print('Initializing H2ODiscritizer')
            if discretizer is None:
                discretizer = H2ODiscretizer(df, ntiles=ntiles, verbose=verbose)
            self.discretizer = discretizer



class H2ODiscretizer(object):
    """Discretizes data into quartiles."""
    def __init__(self, data, ntiles, verbose=False):
        """Initializer

        Args:
            data: H2O data frame.
        """
        self.ntiles = ntiles
        probs = list(np.linspace(0, 1, ntiles + 1)[1:-1])
        if verbose: print('Computing quantiles.')
        qts_df = data.quantile(probs).as_data_frame().set_index('Probs').T

        self.names = {}
        self.lambdas = {}
        self.means = {}
        self.stds = {}
        self.breaks = {}
        cols = data.columns
        try:
            for i_feature, name in enumerate(cols):
                if verbose:
                    print("Getting stats for feature no. {} ({})".format(i_feature, name))
                if data.types[name] == 'enum':
                    continue
                qts = list(qts_df.iloc[i_feature].values)
                self.names[i_feature] = ['{}:Q{}'.format(name, (x + 1)) for x in range(ntiles)]

                self.lambdas[i_feature] = lambda x, qts=qts: np.searchsorted(qts, x)
                self.breaks[i_feature] = [data[name].min()] + qts + [data[name].max()]

                # discritze the column and compute the mean and std of each qunatile
                data['__group__'] = data[name].cut(breaks=self.breaks[i_feature],
                                      labels=[str(x) for x in range(ntiles)],
                                      include_lowest=True)

                gr = data[['__group__', name]].group_by('__group__')
                xx = gr.mean().get_frame()
                gg = xx.as_data_frame()
                means_dict = gg.set_index('__group__')['mean_{}'.format(name)].to_dict()
                # compute stds
                gr = data[['__group__', name]].group_by('__group__')
                gg = gr.sd().get_frame().as_data_frame()
                stds_dict = gg.set_index('__group__')['sdev_{}'.format(name)].to_dict()

                # fill out the missing quantiles with zeros
                self.means[i_feature] = []
                self.stds[i_feature] = []
                for i_tile in range(ntiles):
                    t_mean = means_dict.get(i_tile, 0)
                    self.means[i_feature].append(t_mean)
                    t_std = stds_dict.get(i_tile, 0)
                    t_std += 0.00000000001
                    self.stds[i_feature].append(t_std)
        finally:
            data = data.drop('__group__')

    def discretize(self, data):
        """Discretizes the data.

        Args:
            data: numpy 2d or 1d array

        Returns:
            numpy array of same dimension, discretized.
        """
        ret = data.copy()
        for feature in self.lambdas.keys():
            if len(data.shape) == 1:
                ret[feature] = int(self.lambdas[feature](ret[feature]))
            else:
                ret[:, feature] = self.lambdas[feature](ret[:, feature]).astype(int)
        return ret

    def undiscretize(self, data):
        ret = data.copy()
        for feature in self.means:
            breaks = self.breaks[feature]
            means = self.means[feature]
            stds = self.stds[feature]
            get_inverse = lambda q: max(breaks[q], min(np.random.normal(means[q], stds[q]), breaks[q+1]))
            if len(data.shape) == 1:
                q = int(ret[feature])
                ret[feature] = get_inverse(q)
            else:
                ret[:, feature] = [get_inverse(int(x)) for x in ret[:, feature]]
        return ret
