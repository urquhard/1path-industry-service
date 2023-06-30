import pandas as pd
import numpy as np
import statsmodels.api as sm # 0.13.2!!!
from tqdm import tqdm
import copy
from pydantic import BaseSettings
from statsmodels.regression.rolling import RollingOLS


import warnings
warnings.filterwarnings('ignore')


class Data(BaseSettings):
    
     
    @staticmethod
    def fixed_window_EMA(arr, window=45, alpha=0.5):
        
        # makes EMA with a fixed window, no infinite tail
        
        ret = pd.Series(index=arr.index, name=arr.name)

        arr = np.array(arr)
        l = len(arr)
        stride = arr.strides[0]

        ret.iloc[window-1:] = \
        (pd.DataFrame(np.lib.stride_tricks.as_strided(arr,(l-window+1,window),
                                                      (stride,stride))).T.ewm(alpha).mean().iloc[-1].values)
        return ret

    
    
    @staticmethod
    def resample_data(DNV, period = 7):
    
        DNV_new = copy.deepcopy(DNV)
        index_to_drop = []

        for ID in tqdm(DNV.gecko_id.unique()):

            tmp_data = copy.deepcopy(DNV[DNV.gecko_id == ID])
            tmp_data = tmp_data.sort_index()
            tmp_chains = list(tmp_data.chain.unique())
            for ch in tmp_chains:
                tmp_data2 = tmp_data[tmp_data.chain == ch]
                tmp_category = list(tmp_data2.category.unique())
                for cat in tmp_category:

                    tmp_data3 = copy.deepcopy(tmp_data2[tmp_data2.category == cat])
                    tmp_index = list(tmp_data3.index)

                    for i in range(tmp_data3.shape[0]):
                        if i == 0:
                            continue
                        elif i == tmp_data3.shape[0]-1:
                            tmp_data3.iloc[i]['volume'] = tmp_data3.iloc[i - (i % period):i].volume.sum()
                        else:

                            if i % period == 0:
                                tmp_data3.iloc[i]['volume'] = tmp_data3.iloc[i-period:i].volume.sum()
                            else:
                                index_to_drop.append(tmp_index[i])

        DNV_new.drop(index = index_to_drop, inplace = True)

        return DNV_new
    
    
    @staticmethod
    def data_preparation(DNV):
    
        DNV = DNV[(DNV['chain'] != 'pool2')&(DNV['chain'] != 'staking')&(DNV['chain'] != 'borrowed')]

        whitelisted_chains = ['Ethereum', 'Binance', 'Avalanche', 'Polygon']

        whitelisted_categories = [ "Dexes", "Lending", "CDP", "Bridge", "Yield", "Liquid Staking",
                                   "Yield Aggregator", "Derivatives", "Cross Chain"]

        DNV[['chain','submeasuere']] = DNV['chain'].str.split('-',expand=True)
        DNV = DNV.drop(['address', 'llama_id'], axis = 1).groupby(['chain','gecko_id', 'category', 'date']).\
        agg({'symbol': 'first','TVL': lambda x: x.sum(min_count=1), 'price': 'first',
             'market_cap': 'first', 'volume': 'first'}).reset_index()
        # Create modified categories and modified chanes
        DNV['mod_chain'] = DNV['chain']

        # Will not whitelisted chains and categories with 'Other'
        DNV['mod_chain'][~DNV['mod_chain'].isin(whitelisted_chains)] = 'Other'

        DNV['mod_category'] = DNV['category']
        DNV['mod_category'][~DNV['mod_category'].isin(whitelisted_categories)] = 'Other'

        # Fuse similar categories together
        DNV['mod_category'][DNV['category'] == 'Services'] = 'Yield'
        DNV['mod_category'][DNV['category'] == 'Farm'] = 'Yield'
        DNV['mod_category'][DNV['category'] == 'Services'] = 'Yield'
        DNV['mod_category'][DNV['category'] == 'Yield Aggregator'] = 'Yield'
        DNV['mod_category'][DNV['category'] == 'Levaraged Farming'] = 'Yield'
        DNV['mod_category'][DNV['category'] == 'Liquidity manager'] = 'Yield'
        DNV['mod_category'][DNV['category'] == 'Liquid Staking'] = 'Yield'

        DNV['mod_category'][DNV['category'] == 'CDP'] = 'Lending'
        DNV['mod_category'][DNV['category'] == 'Uncollateralized Lending'] = 'Lending'
        DNV['mod_category'][DNV['category'] == 'RWA Lending'] = 'Lending'
        DNV['mod_category'][DNV['category'] == 'NFT Lending'] = 'Lending'

        DNV['mod_category'][DNV['category'] == 'Cross Chain'] = 'Bridge'

        DNV['mod_category'][DNV['category'] == 'Synthetics'] = 'Derivatives'
        DNV['mod_category'][DNV['category'] == 'Options'] = 'Derivatives'
        DNV['mod_category'][DNV['category'] == 'Options Vault'] = 'Derivatives'
        DNV['mod_category'][DNV['category'] == 'Prediction Market'] = 'Derivatives'
        
        DNV['date'] = pd.to_datetime(DNV['date'])
        DNV = DNV.sort_values(['gecko_id', 'date'])

        return DNV


    @staticmethod
    def beta_preparation(DNV, returns, prices, mcap, quarantine_period = 45, 
                         momentum_period = [7, 30], multiple_momentum = False):
    
        
        whitelisted_chains = ['Ethereum', 'Binance', 'Avalanche', 'Polygon']

        style_factors = ['factor_log_mc', 'factor_log_mc_tvl', 'VMR', 'factor_momentum', 'Mbeta']
        whitelisted_categories = [ "Dexes", "Lending", "CDP", "Bridge", "Yield", "Liquid Staking", 
                                  "Yield Aggregator", "Derivatives", "Cross Chain"]

        # Calculate relative TVL exposures to modified categories
        category_exposures = DNV.groupby(['mod_category', 'date', 'gecko_id']).\
            agg({'TVL': lambda x: x.sum(min_count=1)}).reset_index().\
            pivot(index=['date', 'gecko_id'], columns='mod_category', values='TVL').reset_index().ffill().fillna(0)
        categories_columns = category_exposures.columns[category_exposures.columns.isin(whitelisted_categories)|
                                                        (category_exposures.columns == 'Other')]

        category_exposures[categories_columns] = \
        category_exposures[categories_columns].div(category_exposures[categories_columns].sum(axis = 1), axis = 0).\
        fillna(0)


        # Calculate relative TVL exposures to modified chains
        blockchain_exposures = DNV.groupby(['mod_chain', 'date', 'gecko_id']).\
             agg({'TVL': lambda x: x.sum(min_count=1)}).reset_index().\
        pivot(index=['date', 'gecko_id'], columns='mod_chain', values='TVL').reset_index().ffill().fillna(0)
        blockchain_columns = blockchain_exposures.columns[blockchain_exposures.columns.isin(whitelisted_chains)|\
                                                          (blockchain_exposures.columns == 'Other')]

        blockchain_exposures[blockchain_columns] = \
        blockchain_exposures[blockchain_columns].div(blockchain_exposures[blockchain_columns].sum(axis = 1), axis = 0).\
        fillna(0)

        #----- Merging result and calculating style factors

        # Merge data into single dataframe
        Beta = pd.merge(blockchain_exposures, category_exposures,  how='left', left_on=['gecko_id','date'],\
                        right_on =['gecko_id','date'], suffixes = ['_chain', '_category'])

        # Calculate current price retrun to explain

        current_returns_df = returns.melt(ignore_index = False).reset_index().\
        rename(columns={"value": "current_return", 'index': 'date'})

        # Calculate momentum factor

        momentum_returns = (prices.shift(momentum_period[0])/prices.shift(momentum_period[1]-1)).\
        melt(ignore_index = False).reset_index().\
        rename(columns={"value": "factor_momentum", 'index': 'date'})

        if multiple_momentum:
            tmp_momentum = copy.deepcopy(momentum_returns)
            for i in range(momentum_period[1]-3, momentum_period[0] + 7, -3):
                momentum_returns1 = pd.DataFrame(DNV.groupby(["gecko_id", 'date']).agg({'price':'last'}).reset_index().\
                                        set_index('date').groupby('gecko_id')['price'].\
                                        apply(lambda x: x.shift(momentum_period[0])/x.shift(i)-1)).\
                                        reset_index().rename(columns={"price": "factor_momentum"})
                tmp_momentum['factor_momentum_' + str(i)] = momentum_returns1['factor_momentum']

            momentum_returns['factor_momentum'] = \
            list(tmp_momentum[list(tmp_momentum.columns)[2:]].ewm(alpha=0.5, axis=1).mean().iloc[:,-1])


        # Calculate volume/mcap factor

        volumes = DNV.groupby(["gecko_id", 'date']).agg({'volume':'last'}).reset_index().\
        pivot(index = 'date', values = 'volume', columns = 'gecko_id')
        volume_mcap = np.log(volumes/mcap).rolling(30).mean().melt(ignore_index = False).\
        reset_index().rename(columns={"value": "VMR", 'index': 'date'})

        # Calculate log mcap factor and log tvl
        log_mcap = np.log(mcap).melt(ignore_index = False).reset_index().\
        rename(columns={"value": "factor_log_mc", 'index': 'date'})
        log_tvl = np.log(DNV.groupby(["gecko_id", 'date']).agg({'TVL': lambda x: x.sum(min_count=1)}).reset_index().\
                         pivot(index= 'date', columns = 'gecko_id', values = 'TVL').\
                         ffill()).reset_index().melt(id_vars=['date']).rename(columns={"value": "log_tvl"})

        # Quarantine flag (tvl and market cap should be not na X time ago)

        # Extract symbol for gecko_id
        symbol = DNV.groupby(["gecko_id", 'date']).agg({'symbol':'first'}).reset_index()


        # Merge everything
        Beta = pd.merge(Beta, current_returns_df,  how='left', left_on=['gecko_id','date'], \
                        right_on =['gecko_id','date'])
        Beta = pd.merge(Beta, momentum_returns,  how='left', left_on=['gecko_id','date'], \
                        right_on =['gecko_id','date'])
        Beta = pd.merge(Beta, volume_mcap,  how='left', left_on=['gecko_id','date'], right_on =['gecko_id','date'])
        Beta = pd.merge(Beta, log_mcap,  how='left', left_on=['gecko_id','date'], right_on =['gecko_id','date'])
        Beta = pd.merge(Beta, log_tvl,  how='left', left_on=['gecko_id','date'], right_on =['gecko_id','date'])
        Beta = pd.merge(Beta, symbol,  how='left', left_on=['gecko_id','date'], right_on =['gecko_id','date'])


        # Should be two options: non-NaN observations during quarantine_period or some rolling average of mcap or volume 
        # should be greater than some value ($100000)
        ret_q = Beta.pivot(index = 'date', columns = 'gecko_id', values= 'current_return').rolling(quarantine_period)\
        .apply(lambda x: (~x.isna()).all())
        mcap_q = Beta.pivot(index = 'date', columns = 'gecko_id', values= 'factor_log_mc').\
        replace([-np.inf, np.inf], np.nan).rolling(quarantine_period).apply(lambda x: (~x.isna()).all())


        tvl_q = Beta.pivot(index = 'date', columns = 'gecko_id', values= 'log_tvl').\
        replace([-np.inf, np.inf], np.nan).rolling(quarantine_period).apply(lambda x: (~x.isna()).all())
        quarantine_flag = (ret_q*mcap_q*tvl_q).reset_index().melt(id_vars=['date']).\
        rename(columns={"value": "quarantine_flag"})


        Beta = pd.merge(Beta, quarantine_flag,  how='left', left_on=['gecko_id','date'], \
                        right_on =['gecko_id','date'])


        market_return = (quarantine_flag.pivot(index = 'date', columns = 'gecko_id',
                                               values = 'quarantine_flag')*returns).mean(axis = 1)


        mbeta = pd.DataFrame(index = market_return.index, columns = returns.columns)
        for i in returns.columns:
            endog = returns[i]
            if (~endog.isna()).sum() > 92:
                exog = sm.add_constant(market_return)
                rols = RollingOLS(endog, exog, window=90)
                rres = rols.fit()
                mbeta[i] = [x[1] for x in rres.params.values]
        mbeta = mbeta.shift()
        Mbeta_melted = mbeta.melt(ignore_index = False).reset_index().rename(columns={"value": "Mbeta"})


        Beta = pd.merge(Beta, Mbeta_melted,  how='left', left_on=['gecko_id','date'], \
                        right_on =['gecko_id','date'])


        # Calculate log_mc_tvl factor
        Beta['factor_log_mc_tvl'] = Beta['factor_log_mc'] - Beta['log_tvl']
        # Beta['quarantine_flag'] = quarantine_flag

        # Apply restriction logic to 'other' categories

        Beta[list(set(categories_columns) - set(['Other']))] =\
        Beta[list(set(categories_columns) - set(['Other']))].sub(Beta.Other_category, axis = 0)
        Beta[list(set(blockchain_columns) - set(['Other']))] =\
        Beta[list(set(blockchain_columns) - set(['Other']))].sub(Beta.Other_chain, axis = 0)

        return Beta, categories_columns, blockchain_columns, style_factors
    
    
    @staticmethod
    def filter_data(DNV, max_treshold = 3, min_treshold = -0.7, max_dat_to_correct = 4):
        
        # filter of outliers in returns and market caps
        # max_treshold  - daily return that we consider suspicious (3 = 300%)
        # min_treshold  - likewise (-0.7 = -70%)
        # max_dat_to_correct = number of days to find and fix outliers
        start_date = DNV.date.min()
        end_date = DNV.date.max()
        idx = pd.date_range(start_date, end_date)
        

        # ------- Prices calculations
        
        prices = DNV.groupby(["gecko_id", 'date']).agg({'price':'last'}).reset_index().\
        pivot(columns = 'gecko_id', index= 'date', values = 'price').reindex(idx, fill_value=np.nan).ffill()        
        
        # ------- Returns calculations
        
        returns = prices.pct_change()        
        
        # ------- Market capitalisation calculations

        mar_cap = DNV.groupby(["gecko_id", 'date']).agg({'market_cap':'last'}).reset_index().\
        pivot(columns = 'gecko_id', index= 'date', values = 'market_cap').reindex(idx, fill_value=np.nan).ffill()
        
        critical_missing_tvl = set(['adamant', 'apex-token-2', 'aurora-dao', 'bao-finance', 'bunicorn', 
                                    'chainge-finance', 'dodo', 'dual-finance', 'elk-finance', 'hashflow', 
                                    'mdex', 'melon', 'nuls', 'pnetwork', 'quipuswap-governance-token', 
                                    'revest-finance', 'sphere-finance', 'unslashed-finance', 'wing-finance', 
                                    'yield-yak', 'youves-you-governance', 'zilswap']) 
        critical_missing_mcap = set(['cvault-finance', 'flash-stake', 'honey', 'metronome', 'moonswap',
                                     'sharedstake-governance-token'])
        critical_missing_prices = set(['aada-finance', 'arkadiko-protocol', 'cvault-finance', 'defis-network',
                                       'flash-stake', 'guarded-ether', 'kolibri-dao', 'liqwid-finance', 'meta-pool',
                                       'minswap', 'moonswap', 'nervenetwork', 'quipuswap-governance-token',
                                       'rebasing-tbt', 'sharedstake-governance-token', 'shark', 'short-term-t-bill-token',
                                       'tangible', 'wrap-governance-token', 'youves-you-governance'])
        endogenous_or_inadequate = set(['defily', 'defidollar-dao', 'fountain-protocol', 'huobi-btc', 'malinka', 
                                        'mercurial', 'proxy','mobius-money', 'nervenetwork', 'wrap-governance-token',
                                        'ichi-farm', 'sashimi', 'equilibrium-eosdt', 'sperax-usd', 'frax-ether'])
        to_drop = list(critical_missing_tvl.union(critical_missing_mcap, critical_missing_prices, endogenous_or_inadequate))
        

        ret_copy = copy.deepcopy(returns)
        price_copy = copy.deepcopy(prices)

        ret_copy = ret_copy[list(set(ret_copy.columns) - set(to_drop))]
        price_copy = price_copy[list(set(price_copy.columns) - set(to_drop))]

        indexes = list(ret_copy.index)
        
        # list of gecko_id for analysis
        ID_to_test = []
        max_id = ret_copy.max()
        min_id = ret_copy.min()
        ID_to_test += list(max_id[max_id >= max_treshold].index)
        ID_to_test += list(min_id[min_id <= min_treshold].index)

        for ID in tqdm(ID_to_test): 

            for q in range(max_dat_to_correct):

                for i in range(1, len(indexes) - (q+1)):
                    
                    # checking if there was a outlier, or if the token really rose/dropped a lot
                    if ((ret_copy[ID].loc[indexes[i]] > max_treshold) and \
                        (ret_copy[ID].loc[indexes[i+1+q]] < min_treshold / 2))\
                    or ((ret_copy[ID].loc[indexes[i]] < min_treshold) and \
                        (ret_copy[ID].loc[indexes[i+1+q]] > max_treshold / 2)):

                        # linear interpolation for nearest normal points
                        p1 = price_copy.loc[indexes[i-1]][ID]
                        p2 = price_copy.loc[indexes[i+1+q]][ID]
                        mar_cap1 = mar_cap.loc[indexes[i-1]][ID]
                        mar_cap2 = mar_cap.loc[indexes[i+1+q]][ID]

                        for k in range(q+1):

                            if price_copy.loc[indexes[i+1+q]][ID] > price_copy.loc[indexes[i-1]][ID]:

                                price_copy.loc[indexes[i+k]][ID] = p1 + ((k+1) / (q + 1)) * (p2-p1)
                                mar_cap.loc[indexes[i+k]][ID] = mar_cap1 + ((k+1) / (q+1)) * (mar_cap2 - mar_cap1)

                            else:

                                price_copy.loc[indexes[i+k]][ID] = p1 - ((k+1) / (q + 1)) * (p2-p1)
                                mar_cap.loc[indexes[i+k]][ID] = mar_cap1 - ((k+1) / (q+1)) * (mar_cap2 - mar_cap1)

                            ret_copy.loc[indexes[i+k]][ID] = (price_copy.loc[indexes[i+k]][ID] - \
                                                              price_copy.loc[indexes[i-1+k]][ID]) / \
                                                              price_copy.loc[indexes[i-1+k]][ID]

                            ret_copy.loc[indexes[i+1+k]][ID] = (price_copy.loc[indexes[i+1+k]][ID] - \
                                                                price_copy.loc[indexes[i+k]][ID]) /\
                                                                price_copy.loc[indexes[i+k]][ID]


        return ret_copy, price_copy, mar_cap

data = Data()
    