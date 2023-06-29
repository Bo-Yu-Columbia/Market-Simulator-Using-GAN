import os
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class Pipeline:
    # This is a class that performs a sequence of transformations on the data. The transformations are defined in the steps 
    # argument, which is a list of tuples where each tuple contains the name of the transformation and the transformation object. 
    def __init__(self, steps):
        """ Pre- and postprocessing pipeline. """
        self.steps = steps

    def transform(self, x, until=None):
        x = x.clone()
        for n, step in self.steps:
            if n == until:
                break
            x = step.transform(x)
        return x

    def inverse_transform(self, x, until=None):
        for n, step in self.steps[::-1]:
            if n == until:
                break
            x = step.inverse_transform(x)
        return x


class StandardScalerTS():
    """ Standard scales a given (indexed) input vector along the specified axis. """

    def __init__(self, axis=(1)):
        self.mean = None
        self.std = None
        self.axis = axis

    def transform(self, x):
        if self.mean is None:
            self.mean = torch.mean(x, dim=self.axis)
            self.std = torch.std(x, dim=self.axis)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse_transform(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)
    

def get_yield_dataset(durations, with_vol=False): 
    """
    Get different returns series.
    """
    yield_ = pd.read_csv('./data/daily_yield_transformed_all.csv')
    start = '01/03/2017'
    end = '11/16/2022'
    lam = 0.94
    df_yield = {}
    
    for y in durations:
#         print("#############")
#         print(y)
#         print("#############")
        df_yield[y] = yield_[yield_['Symbol'] == y].drop(['Unnamed: 0'], axis=1).set_index(['Date'])[start:end]
#         print(df_yield[y])
#         print("#############")
        
    rtn = {}
#     volume = {}

    for y in durations:
        rtn[y] = df_yield[y][['close_price']].to_numpy(dtype='float32').reshape(1, -1, 1)
#         volume[y] = df_yield[y][['volume']].to_numpy(dtype='float32').reshape(1, -1, 1)

    if not with_vol:           # simulate from EWMA model
        prep_concatenate = []
        vol = {}
        for y in durations:
            var = np.zeros(len(df_yield[y]))
            ret = rtn[y].reshape(-1,1)
            for t in range(1,len(df_yield[y])):
                var[t] = lam * var[t-1] + (1-lam)*np.power(ret[t][0],2)
            vol[y] = var.reshape(1, -1, 1)

        for y in durations:
            prep_concatenate.append(rtn[y])
            prep_concatenate.append(vol[y])
        data_raw = np.concatenate(prep_concatenate, axis=-1)  
    
    else :
        vol = {}
        for y in durations:
            vol[y] = df_yield[y][['medrv']].values[-rtn[y].shape[1]:].reshape(1, -1, 1)
        
        prep_concatenate = []
        for y in durations:
            prep_concatenate.append(rtn[y])
            prep_concatenate.append(vol[y])
            prep_concatenate.append(volume[y])
    
        data_raw = np.concatenate(prep_concatenate, axis=-1)
        
    data_raw = torch.from_numpy(data_raw).float()
    pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
    data_preprocessed = pipeline.transform(data_raw)
    return pipeline, data_raw, data_preprocessed

def get_eib_dataset(durations, with_vol=False):
    """
    Get different returns series.
    """
    # yield_1yr = pd.read_csv('./data/eib_data/daily_yield_transformed_all.csv')
    # yield_5yr = pd.read_csv('./data/eib_data/daily_yield_transformed_all_10yr.csv')
    # start = '01/03/2017'
    # end = '11/16/2022'
    lam = 0.94
    df_yield = {}
    rtn = {}

    for y in durations:
        # df_yield[y] = yield_[yield_['Symbol'] == y].drop(['Unnamed: 0'], axis=1).set_index(['Date'])[start:end]
        df_yield[y] = pd.read_csv('../../data/EIB/ECB_' + y + '.csv')
        print(df_yield[y])
        rtn[y] = df_yield[y].iloc[:, 2].to_numpy(dtype='float32').reshape(1, -1, 1)
        print(rtn[y])
    #
    #     volume = {}
    #
    # for y in durations:
    #     rtn[y] = df_yield[y].iloc[:, 1].to_numpy(dtype='float32').reshape(1, -1, 1)
    #         volume[y] = df_yield[y][['volume']].to_numpy(dtype='float32').reshape(1, -1, 1)

    if not with_vol:  # simulate from EWMA model
        prep_concatenate = []
        vol = {}
        for y in durations:
            var = np.zeros(len(df_yield[y]))
            ret = rtn[y].reshape(-1, 1)
            for t in range(1, len(df_yield[y])):
                var[t] = lam * var[t - 1] + (1 - lam) * np.power(ret[t][0], 2)
            vol[y] = var.reshape(1, -1, 1)

        for y in durations:
            prep_concatenate.append(rtn[y])
            prep_concatenate.append(vol[y])
        data_raw = np.concatenate(prep_concatenate, axis=-1)

    else:
        vol = {}
        for y in durations:
            vol[y] = df_yield[y][['medrv']].values[-rtn[y].shape[1]:].reshape(1, -1, 1)

        prep_concatenate = []
        for y in durations:
            prep_concatenate.append(rtn[y])
            prep_concatenate.append(vol[y])
            prep_concatenate.append(volume[y])

        data_raw = np.concatenate(prep_concatenate, axis=-1)

    data_raw = torch.from_numpy(data_raw).float()
    pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
    data_preprocessed = pipeline.transform(data_raw)
    return pipeline, data_raw, data_preprocessed

def get_equities_dataset(assets=('SPX', 'DJI'), with_vol=True):
    """
    Get different returns series.
    """
    oxford = pd.read_csv('./data/oxfordmanrealizedvolatilityindices.csv')

    start = '2005-01-01 00:00:00+01:00'
    end = '2020-01-01 00:00:00+01:00'

    if assets == ('SPX',):
        df_asset = oxford[oxford['Symbol'] == '.SPX'].set_index(['Unnamed: 0'])  # [start:end]
        price = np.log(df_asset[['close_price']].values)
        rtn = (price[1:] - price[:-1]).reshape(1, -1, 1)
        vol = np.log(df_asset[['medrv']].values[-rtn.shape[1]:]).reshape(1, -1, 1)
        data_raw = np.concatenate([rtn, vol], axis=-1)
    elif assets == ('SPX', 'DJI'):
        df_spx = oxford[oxford['Symbol'] == '.SPX'].set_index(['Unnamed: 0'])[start:end]
        df_dji = oxford[oxford['Symbol'] == '.DJI'].set_index(['Unnamed: 0'])[start:end]
        index = df_dji.index.intersection(df_spx.index)
        df_dji = df_dji.loc[index]
        df_spx = df_spx.loc[index]
        price_spx = np.log(df_spx[['close_price']].values)
        rtn_spx = (price_spx[1:] - price_spx[:-1]).reshape(1, -1, 1)
        vol_spx = np.log(df_spx[['medrv']].values).reshape(1, -1, 1)
        price_dji = np.log(df_dji[['close_price']].values)
        rtn_dji = (price_dji[1:] - price_dji[:-1]).reshape(1, -1, 1)
        vol_dji = np.log(df_dji[['medrv']].values).reshape(1, -1, 1)
        data_raw = np.concatenate([rtn_spx, vol_spx[:, 1:], rtn_dji, vol_dji[:, 1:]], axis=-1)
    else:
        raise NotImplementedError()
    data_raw = torch.from_numpy(data_raw).float()
    pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
    data_preprocessed = pipeline.transform(data_raw)
    return pipeline, data_raw, data_preprocessed


def get_var_dataset(window_size, batch_size=5000, dim=3, phi=0.8, sigma=0.5):
    # This function generates a multivariate autoregressive dataset, processes it, and returns 
    # the data in both its raw and preprocessed forms.
    def multi_AR(window_size, dim=3, phi=0.8, sigma=0.5, burn_in=200):
        window_size = window_size + burn_in
        xt = np.zeros((window_size, dim))
        one = np.ones(dim)
        ide = np.identity(dim)
        MU = np.zeros(dim)
        COV = sigma * one + (1 - sigma) * ide
        W = np.random.multivariate_normal(MU, COV, window_size)
        for i in range(dim):
            xt[0, i] = 0
        for t in range(window_size - 1):
            xt[t + 1] = phi * xt[t] + W[t]
        return xt[burn_in:]

    var_samples = []
    for i in range(batch_size):
        tmp = multi_AR(window_size, dim, phi=phi, sigma=sigma)
        var_samples.append(tmp)
    data_raw = torch.from_numpy(np.array(var_samples)).float()

    def get_pipeline():
        transforms = list()
        transforms.append(('standard_scale', StandardScalerTS(axis=(0, 1))))  # standard scale
        pipeline = Pipeline(steps=transforms)
        return pipeline

    pipeline = get_pipeline()
    data_preprocessed = pipeline.transform(data_raw)
    return pipeline, data_raw, data_preprocessed


def get_arch_dataset(window_size, lag=4, bt=0.055, N=5000, dim=1):
    """
    Creates the dataset: loads data.

    :param data_path: :param t_lag: :param device: :return:
    """

    def get_raw_data(N=5000, lag=4, T=2000, omega=0.00001, bt=0.055, burn_in=2000):
        beta = bt * np.ones(lag)
        eps = np.random.randn(N, T + burn_in)
        logrtn = np.zeros((N, T + burn_in))

        initial_arch = omega / (1 - beta[0])

        arch = initial_arch + np.zeros((N, T + burn_in))

        logrtn[:, :lag] = np.sqrt(arch[:, :lag]) * eps[:, :lag]

        for t in range(lag - 1, T + burn_in - 1):
            arch[:, t + 1] = omega + np.matmul(beta.reshape(1, -1), np.square(
                logrtn[:, t - lag + 1:t + 1]).transpose())  # * (logrtn[:, t] < 0.)
            logrtn[:, t + 1] = np.sqrt(arch[:, t + 1]) * eps[:, t + 1]
        return arch[:, burn_in:], logrtn[:, burn_in:]

    pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
    _, logrtn = get_raw_data(T=window_size, N=N, bt=bt)
    data_raw = torch.from_numpy(logrtn[..., None]).float()
    data_pre = pipeline.transform(data_raw)
    return pipeline, data_raw, data_pre


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def rolling_window(x, x_lag, add_batch_dim=True):
    if add_batch_dim:
        x = x[None, ...]
    return torch.cat([x[:, t:t + x_lag] for t in range(x.shape[1] - x_lag)], dim=0)


def get_mit_arrythmia_dataset(filenames):
    # This function loads the MIT-BIH Arrhythmia Database, processes it, 
    # and returns the data in both its raw and preprocessed forms.
    DATA_DIR = './data/mit_db/'
    import wfdb
    records = list()
    for fn in filenames:
        records.append(wfdb.rdsamp(os.path.join(DATA_DIR, fn), sampto=3000)[0][None, ...])
    records = np.concatenate(records, axis=0)
    records = np.log(5 * (records - records.min() + 1))
    data_raw = torch.from_numpy(records).float()
    pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
    data_pre = pipeline.transform(data_raw)
    return pipeline, data_raw, data_pre

def get_exchange_dataset(exchanges, with_vol=False): 
    """
    Get different returns series.
    """
    exchange_ = pd.read_csv('./data/exchange_transformed.csv')
    start = '2017-01-02'
    end = '2022-12-09'
    lam = 0.94
    df_exchange = {}
    
    for y in exchanges:
        df_exchange[y] = exchange_[exchange_['Symbol'] == y].drop(['Unnamed: 0'], axis=1).set_index(['Date'])[start:end]

    rtn = {}
    for y in exchanges:
        price = np.log(df_exchange[y][['close_price']].values) #.to_numpy(dtype='float32')
        rtn[y] = (price[1:] - price[:-1]).reshape(1, -1, 1)
        
    if not with_vol:           # simulate from EWMA model
        prep_concatenate = []
        vol = {}
        for y in exchanges:
            ret = rtn[y].reshape(-1,1)
            var = np.zeros(len(ret))
            for t in range(1, len(ret)):
                var[t] = lam * var[t-1] + (1-lam)*np.power(ret[t][0],2)
            vol[y] = var.reshape(1, -1, 1)

        for y in exchanges:
            prep_concatenate.append(rtn[y])
            prep_concatenate.append(vol[y])
        data_raw = np.concatenate(prep_concatenate, axis=-1)  
    
    else :
        vol = {}
        for y in exchanges:
            vol[y] = df_exchange[y][['medrv']].values[-rtn[y].shape[1]:].reshape(1, -1, 1)
        
        prep_concatenate = []
        for y in exchanges:
            prep_concatenate.append(rtn[y])
            prep_concatenate.append(vol[y])
    
        data_raw = np.concatenate(prep_concatenate, axis=-1)
        
    data_raw = torch.from_numpy(data_raw).float()
    pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
    data_preprocessed = pipeline.transform(data_raw)
    return pipeline, data_raw, data_preprocessed


def get_data(data_type, p, q, **data_params):
    # This function loads the specified type of data (either 'VAR', 'STOCKS', 'YIELD', 'EXCHANGE', 'ARCH', or 'ECG'), 
    # processes it, and returns the data in a rolling window format.
    if data_type == 'VAR':
        pipeline, x_real_raw, x_real = get_var_dataset(
            40000, batch_size=1, **data_params
        )
    elif data_type == 'STOCKS':
        pipeline, x_real_raw, x_real = get_equities_dataset(**data_params)
    
    elif data_type == 'YIELD':
        pipeline, x_real_raw, x_real = get_yield_dataset(**data_params, with_vol = False)
    elif data_type == 'EXCHANGE':
        pipeline, x_real_raw, x_real = get_exchange_dataset(**data_params, with_vol = False)
    
    
    elif data_type == 'ARCH':
        pipeline, x_real_raw, x_real = get_arch_dataset(
            40000, N=1, **data_params
        )
    elif data_type == 'ECG':
        pipeline, x_real_raw, x_real = get_mit_arrythmia_dataset(**data_params)
    elif data_type == 'EIB':
        pipeline, x_real_raw, x_real = get_eib_dataset(**data_params, with_vol=False)
    else:
        raise NotImplementedError('Dataset %s not valid' % data_type)
    assert x_real.shape[0] == 1
    x_real = rolling_window(x_real[0], p + q)
    return x_real


def download_man_ahl_dataset():
    # This function downloads the Oxford-Man Institute of Quantitative Finance Realized Library, 
    # which includes realized measures for a range of assets.
    import requests
    from zipfile import ZipFile
    url = 'https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip'
    r = requests.get(url)
    with open('./oxford.zip', 'wb') as f:
        pbar = tqdm(unit="B", total=int(r.headers['Content-Length']))
        for chunk in r.iter_content(chunk_size=100 * 1024):
            if chunk:
                pbar.update(len(chunk))
                f.write(r.content)
    zf = ZipFile('./oxford.zip')
    zf.extractall(path='./data')
    zf.close()
    os.remove('./oxford.zip')


def download_mit_ecg_dataset():
    import requests
    from zipfile import ZipFile
   # url = 'https://storage.googleapis.com/mitdb-1.0.0.physionet.org/mit-bih-arrhythmia-database-1.0.0.zip'
   # url = 'https://raw.githubusercontent.com/hsd1503/PhysioNet/master/data/mit-bih-arrhythmia-database-1.0.0.zip'
   # r = requests.get(url)
    with open('./ mit_db.zip', 'wb') as f:
        pbar = tqdm(unit="B", total=int(r.headers['Content-Length']))
        for chunk in r.iter_content(chunk_size=100 * 1024):
            if chunk:
                pbar.update(len(chunk))
                f.write(r.content)
    zf = ZipFile('./ mit_db.zip')
    zf.extractall(path='./data')
    zf.close()
    os.remove('./ mit_db.zip')

