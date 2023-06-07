import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from datetime import timedelta
import time
import copy

import warnings
warnings.filterwarnings('ignore')

# Функция качает из DefiLlama информацию о протоколах (адреса, названия, слаги) и выкидывает ненужные протоколы

def get_all_llama_slugs():
    
    url = 'https://api.llama.fi/protocols'
    request = requests.get(url).json()
    llama_data = pd.DataFrame(request)
    llama_data = llama_data[['name', 'address', 'symbol', 'chain', 'slug', 'gecko_id', 'category', 'tvl']]
    llama_data = llama_data[llama_data['tvl'] >= 1000000]
    llama_data = llama_data.sort_values('symbol')
    llama_data = llama_data[llama_data['symbol'] != 'BNB']
    llama_data = llama_data[llama_data['symbol'] != 'MATIC']
    llama_data = llama_data[llama_data['symbol'] != '-'].reset_index(drop = True)
    
    no_gecko = llama_data[llama_data['gecko_id'].isna() == True]
    yes_gecko = llama_data[llama_data['gecko_id'].isna() == False]
    
    no_gecko['gecko_id'][no_gecko['symbol'] == 'AAVE'] = 'aave'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'ABC'] = 'abc-pos-pool'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'ACA'] = 'acala'               # CHAIN
    no_gecko['gecko_id'][no_gecko['symbol'] == 'AGI'] = 'auragi'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'ALPACA'] = 'alpaca-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'AQUA'] = 'planet-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'ASX'] = 'asymetrix'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BAL'] = 'balancer'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BANANA'] = 'apeswap-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BEND'] = 'benddao'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BLUR'] = 'blur'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BNB'] = 'binancecoin'         # CHAIN
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BNC'] = 'bifrost-native-coin'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BNT'] = 'bancor'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BOW'] = 'archerswap-bow'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'CAKE'] = 'pancakeswap-token'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'CAP'] = 'cap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'CNC'] = 'conic-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'COMP'] = 'compound-governance-token'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'CREAM'] = 'cream-2'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'CRV'] = 'curve-dao-token'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'DCHF'] = 'defi-franc'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'DFI'] = 'defichain'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'EMPIRE'] = 'empire-network'   # CHAIN
    no_gecko['gecko_id'][no_gecko['symbol'] == 'FLDX'] = 'flair-dex'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'FLOW'] = 'velocimeter-flow'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'FOREX'] = 'handle-fi'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'FXS'] = 'frax-share'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'GAMMA'] = 'green-planet'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'GRAIL'] = 'camelot-token'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'HEC'] = 'hector-dao'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'HFT'] = 'hashflow'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'INTR'] = 'interlay'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'INV'] = 'inverse-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'IZI'] = 'izumi-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'JBX'] = 'juicebox'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'JOE'] = 'joe'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'JST'] = 'just'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'KLEVA'] = 'kleva'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'KNC'] = 'kyber-network-crystal'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'KSWAP'] = 'kyotoswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'LIF3'] = 'lif3'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'LSD'] = 'lsdx-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'LVL'] = 'level'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MARE'] = 'mare-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MEGA'] = 'megaton-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MER'] = 'mercurial'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MET'] = 'metronome'
#     no_gecko['gecko_id'][no_gecko['slug'] == 'mm-finance-arbitrum'] = 'mmfinance-arbitrum'
    no_gecko['gecko_id'][no_gecko['slug'] == 'mm-finance-cronos'] = 'mmfinance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MMO'] = 'mad-meerkat-optimizer'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MORPHO'] = 'morpho'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MPX'] = 'mpx'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MTA'] = 'meta'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'NEST'] = 'nest'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'NPT'] = 'neopin'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'OREO'] = 'oreoswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'PLY'] = 'plenty-ply'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'QUICK'] = 'quickswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'RBN'] = 'ribbon-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'RDNT'] = 'radiant-capital'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'RF'] = 'reactorfusion'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SAPR'] = 'swaprum'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SDL'] = 'stake-link'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SEAN'] = 'starfish-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SNEK'] = 'solisnek'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SOV'] = 'sovryn'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SPA'] = 'sperax'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SPIRIT'] = 'spiritswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'STELLA'] = 'stellaswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SUN'] = 'sun-token'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SUSHI'] = 'sushi'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'TETHYS'] = 'tethys-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'TETU'] = 'tetu'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'THE'] = 'thena'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'THL'] = 'thala'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'TOMB'] = 'tomb'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'UMEE'] = 'umee'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'UNI'] = 'uniswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'VARA'] = 'equilibre'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'VC'] = 'velocore'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'VOLT'] = 'voltswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'WOO'] = 'woo-network'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'WYND'] = 'wynd'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'Y2K'] = 'y2k'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'YOK'] = 'yokaiswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'ZYB'] = 'zyberswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'aUSD'] = 'acala-dollar-acala'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'bLUSD'] = 'boosted-lusd'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'csMatic'] = 'claystack-staked-matic'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'frxETH'] = 'frax-ether'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'kDAO'] = 'kolibri-dao'
    
    llama_data = pd.concat([no_gecko, yes_gecko])
    llama_data = llama_data.sort_values('symbol')
    llama_data = llama_data[llama_data['gecko_id'].isna() == False].reset_index(drop = True)
    
    return llama_data

# Функция обрезает ряды из TVL по введенным датам и заполняет пробелы в данных

def cut_and_fill_series(series, start_date, end_date, max_space = 3):

    start_timestamp = pd.to_datetime(start_date)
    end_timestamp = pd.to_datetime(end_date)

    series = series[series.index >= start_timestamp]
    series = series[series.index <= end_timestamp]
    
    return series

def fill_series_with_nans(series, start_time, end_time):
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    for i in range((end_time - start_time  + timedelta(days = 1)).days):
        date = (start_time + timedelta(days = i))  #.strftime('%Y-%m-%d')
        try:
            series.loc[date]
        except KeyError:
            series.loc[date] = np.nan 
    filled_series = series.sort_index()
    return filled_series

# Функция выкачивает TVL из DefiLlama и записывает их в словарик

def get_available_TVL_from_DefiLlama(llama_data, start_date, end_date):

    
    chain_tokens_series = pd.Series(['Ethereum', 'Binance', 'Polygon', 'Avalanche', 'Kava'], \
                                index = ['ETH', 'BNB', 'MATIC', 'AVAX', 'KAVA'])

    count_protocol = 0
    tvl_dict = {}
    print('Loading TVL of all protocols')
    for i in tqdm(range(len(llama_data))):

        ID = llama_data['slug'][i]
        token = llama_data['symbol'][i]

        if token in chain_tokens_series:
            continue
        
        key_token = llama_data['symbol'][i]
        if (i != len(llama_data['symbol']) - 1) and (i != 0): 
            if (llama_data['symbol'][i] == llama_data['symbol'][i + 1]):
                count_protocol += 1
                key_token = llama_data['symbol'][i] + '_' + str(count_protocol)
            elif (llama_data['symbol'][i] == llama_data['symbol'][i - 1]):
                count_protocol += 1
                key_token = llama_data['symbol'][i] + '_' + str(count_protocol)
                count_protocol = 0
            else:
                count_protocol = 0
        
        llama_tvl_endpoint = "https://api.llama.fi/protocol/"
        request = requests.get(llama_tvl_endpoint + ID).json()
        try:
            chain_list = list(request['chainTvls'].keys())
        except KeyError:
            print('Bad Llama ID:', ID)
            print('Bad Token:', token)
            print('-' * 30)
            continue

        mini_dict = {}
        for chain in chain_list:
            tvls = pd.DataFrame(request['chainTvls'][chain]['tvl'])[ : -1]
            tvls['date'] = pd.to_datetime(tvls['date'], unit = 's')
            tvls['date'] = tvls['date'].apply(lambda x: x.round(freq = 'D'))
            symbol = token
            protocol_chain_tvl = pd.Series(tvls['totalLiquidityUSD'].values, index = tvls['date'].values, name = symbol)
            
            cutted_protocol_chain_tvl = cut_and_fill_series(protocol_chain_tvl, start_date, end_date)
                        
            filled_protocol_chain_tvl = fill_series_with_nans(cutted_protocol_chain_tvl, start_date, end_date)
          
            filled_protocol_chain_tvl.index = filled_protocol_chain_tvl.index.strftime('%Y-%m-%d')
            mini_dict[chain] = filled_protocol_chain_tvl.to_dict()
            
        tvl_dict[key_token] = mini_dict

    return tvl_dict

# Функция достает из CoinGecko все id нужных нам протоколов и убирает ненужные

def get_all_gecko_ids(list_of_tokens):
    
    gecko_endpoint = 'https://api.coingecko.com/api/v3/coins/list'
    request = requests.get(gecko_endpoint).json()
    cg_info = pd.DataFrame(request)

    cg_info = cg_info.sort_values('symbol').reset_index(drop = True)
    cg_info['symbol'] = cg_info['symbol'].str.upper()
    cg_info['symbol'].replace('BABYDOGE', 'BabyDoge', inplace = True)
    cg_info = cg_info[cg_info['symbol'].isin(list_of_tokens)].reset_index(drop = True)

    cg_info = cg_info[(cg_info['symbol'] != 'ALPHA') | (cg_info['id'] == 'alpha-finance')]
    cg_info = cg_info[(cg_info['symbol'] != 'APE') | (cg_info['id'] == 'apecoin')]
    cg_info = cg_info[(cg_info['symbol'] != 'APX') | (cg_info['id'] == 'apollox-2')]
    cg_info = cg_info[(cg_info['symbol'] != 'AURA') | (cg_info['id'] == 'aura-finance')]
    cg_info = cg_info[(cg_info['symbol'] != 'AVAX') | (cg_info['id'] == 'avalanche-2')]
    cg_info = cg_info[(cg_info['symbol'] != 'AXL') | (cg_info['id'] == 'axelar')]
    cg_info = cg_info[(cg_info['symbol'] != 'BabyDoge') | (cg_info['id'] == 'baby-doge-coin')]
    cg_info = cg_info[(cg_info['symbol'] != 'BEND') | (cg_info['id'] == 'benddao')]
    cg_info = cg_info[(cg_info['symbol'] != 'BETA') | (cg_info['id'] == 'beta-finance')]
    cg_info = cg_info[(cg_info['symbol'] != 'BIFI') | (cg_info['id'] == 'beefy-finance')]
    cg_info = cg_info[(cg_info['symbol'] != 'BNB') | (cg_info['id'] == 'binancecoin')]
    cg_info = cg_info[(cg_info['symbol'] != 'BONE') | (cg_info['id'] == 'bone-shibaswap')]
    cg_info = cg_info[(cg_info['symbol'] != 'CHESS') | (cg_info['id'] == 'tranchess')]
    cg_info = cg_info[(cg_info['symbol'] != 'COMP') | (cg_info['id'] == 'compound-governance-token')]
    cg_info = cg_info[(cg_info['symbol'] != 'COW') | (cg_info['id'] == 'coinwind')]
    cg_info = cg_info[(cg_info['symbol'] != 'CREAM') | (cg_info['id'] == 'cream-2')]
    cg_info = cg_info[(cg_info['symbol'] != 'CTR') | (cg_info['id'] == 'concentrator')]
    cg_info = cg_info[(cg_info['symbol'] != 'DFI') | (cg_info['id'] == 'defichain')]
    cg_info = cg_info[(cg_info['symbol'] != 'DOGE') | (cg_info['id'] == 'dogecoin')]
    cg_info = cg_info[(cg_info['symbol'] != 'DYDX') | (cg_info['id'] == 'dydx')]
    cg_info = cg_info[(cg_info['symbol'] != 'ERN') | (cg_info['id'] == 'ethernity-chain')]
    cg_info = cg_info[(cg_info['symbol'] != 'ETH') | (cg_info['id'] == 'ethereum')]
    cg_info = cg_info[(cg_info['symbol'] != 'FLOW') | (cg_info['id'] == 'flow')]
    cg_info = cg_info[(cg_info['symbol'] != 'GFI') | (cg_info['id'] == 'goldfinch')]
    cg_info = cg_info[(cg_info['symbol'] != 'HOP') | (cg_info['id'] == 'hop-protocol')]
    cg_info = cg_info[(cg_info['symbol'] != 'LDO') | (cg_info['id'] == 'lido-dao')]
    cg_info = cg_info[(cg_info['symbol'] != 'LINA') | (cg_info['id'] == 'linear')]
    cg_info = cg_info[(cg_info['symbol'] != 'LVL') | (cg_info['id'] == 'level')]
    cg_info = cg_info[(cg_info['symbol'] != 'MANA') | (cg_info['id'] == 'decentraland')]
    cg_info = cg_info[(cg_info['symbol'] != 'MDX') | (cg_info['id'] == 'mdex')]
    cg_info = cg_info[(cg_info['symbol'] != 'ORC') | (cg_info['id'] == 'orbit-chain')]
    cg_info = cg_info[(cg_info['symbol'] != 'QI') | (cg_info['id'] == 'benqi')]
    cg_info = cg_info[(cg_info['symbol'] != 'QUICK') | (cg_info['id'] == 'quickswap')]
    cg_info = cg_info[(cg_info['symbol'] != 'SAND') | (cg_info['id'] == 'the-sandbox')]
    cg_info = cg_info[(cg_info['symbol'] != 'SHIB') | (cg_info['id'] == 'shiba-inu')]
    cg_info = cg_info[(cg_info['symbol'] != 'THE') | (cg_info['id'] == 'thena')]
    cg_info = cg_info[(cg_info['symbol'] != 'TRU') | (cg_info['id'] == 'truefi')]
    cg_info = cg_info[(cg_info['symbol'] != 'UNI') | (cg_info['id'] == 'uniswap')]
    cg_info = cg_info[(cg_info['symbol'] != 'VTX') | (cg_info['id'] == 'vector-finance')]
    cg_info = cg_info[(cg_info['symbol'] != 'XRP') | (cg_info['id'] == 'ripple')]
    cg_info = cg_info[(cg_info['symbol'] != 'APE') | (cg_info['id'] == 'apecoin')]
    cg_info = cg_info.reset_index(drop = True)
    
    return cg_info


# Функция достает PMCV протокола по gecko_id

def get_token_data_from_coingecko(gecko_id):
    
    url = 'https://api.coingecko.com/api/v3/coins/' + gecko_id + '/market_chart?vs_currency=usd&days=max&interval=daily'
    request = requests.get(url).json()

    token_info_df = pd.DataFrame.from_dict(request)
    try:
        split_prices = pd.DataFrame(token_info_df['prices'].tolist(), columns = ['timestamp', 'prices'])
    except KeyError:
        print(request)
        return
    split_market_caps = pd.DataFrame(token_info_df['market_caps'].tolist(), columns = ['time1', 'market_caps'])
    split_volumes = pd.DataFrame(token_info_df['total_volumes'].tolist(), columns = ['time2', 'total_volumes'])

    token_info_df = pd.concat([split_prices, split_market_caps, split_volumes], axis = 1)
    token_info_df.index = pd.to_datetime(token_info_df['timestamp'] / 1000, unit = 's')
    token_info_df = token_info_df.drop(['time1', 'time2', 'timestamp'], axis = 1)
    return token_info_df

# Функция обрезает датафреймы из PMCV по введенным датам и заполняет пробелы в данных

def cut_and_fill_dataframe(dataframe, start_date, end_date, max_space = 3):
  
    df = copy.deepcopy(dataframe)[ : -1]
    token = df.index.name
    start_timestamp = pd.to_datetime(start_date)
    end_timestamp = pd.to_datetime(end_date)

    df = df[df.index >= start_timestamp]
    df = df[df.index <= end_timestamp]

    return df

def fill_dataframe_with_nans(dataframe, start_time, end_time):
 
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    for i in range((end_time - start_time  + timedelta(days = 1)).days):
        date = (start_time + timedelta(days = i))   #.strftime('%Y-%m-%d')
        try:
            dataframe.loc[date]
        except KeyError:
            dataframe.loc[date] = np.nan 
    filled_dataframe = dataframe.sort_index()
    
    return filled_dataframe

# Функция достает все данные из CoinGecko и сохраняет их в словарике

def download_full_gecko_data(gecko_info, start_date, end_date):
    
    list_of_tokens = list(gecko_info['symbol'])
    list_of_gecko_ids = list(gecko_info['gecko_id'])
    gecko_dict = {}
    missing_tokens_in_gecko = []
    
    print()
    print('Loading PMCV of all tokens')
    for i in tqdm(range(len(list_of_gecko_ids))):
        gecko_id = list_of_gecko_ids[i]
        token_name = list_of_tokens[i]
        try:
            if (i % 14 == 0) and (i != 0):    # TIMEOUT
                time.sleep(120)
            token_data = get_token_data_from_coingecko(gecko_id)
        except ValueError:
            print(token_name)
            missing_tokens_in_gecko.append(token_name)
            continue
            
            
        token_data.index = token_data.index.round(freq = 'D')
        token_data = token_data[~token_data.index.duplicated(keep = 'first')]
        token_data = token_data.rename_axis(token_name)
        
        cutted_token_data = cut_and_fill_dataframe(dataframe = token_data, start_date = start_date, end_date = end_date)
        
        filled_token_data = fill_dataframe_with_nans(dataframe = cutted_token_data, start_time = start_date, end_time = end_date)
        
        filled_token_data = filled_token_data.reset_index()
        filled_token_data[token_name] = filled_token_data[token_name].apply(lambda x: x.strftime('%Y-%m-%d'))
        filled_token_data = filled_token_data.set_index(token_name)
        
        
        prices_minidict = filled_token_data['prices'].to_dict()
        market_caps_minidict = filled_token_data['market_caps'].to_dict()
        volumes_minidict = filled_token_data['total_volumes'].to_dict()
        minidict = {'prices': prices_minidict, 'market_caps': market_caps_minidict, 'volumes': volumes_minidict}
        gecko_dict[token_name] = minidict
        
    return gecko_dict, missing_tokens_in_gecko

# Функция, которая сортирует словарь по ключам
def sort_dict_by_keys(dictionary):
    keys = list(dictionary.keys())
    keys.sort()
    sorted_dict = {i: dictionary[i] for i in keys}
    return sorted_dict

# Функция, которая преобразует словарь в длинный датафрейм
def dict_to_dataframe(dictionary):
    
    df_columns = ['date', 'symbol', 'gecko_id', 'llama_id', 'category', 'chain', 'address', 'price', 'market_cap', 'volume', 'TVL']
    long_df = pd.DataFrame(columns = df_columns)
    chain_tokens = ['ETH', 'BNB', 'MATIC', 'AVAX', 'KAVA']
    
    for token_key in dictionary.keys():
        try:
            for chain_key in dictionary[token_key]['TVL'].keys():
                average_not_small_df = pd.DataFrame(columns = df_columns)
                average_not_small_df['date'] = list(dictionary[token_key]['prices'].keys())
                average_not_small_df['symbol'] = dictionary[token_key]['symbol']
                average_not_small_df['gecko_id'] = dictionary[token_key]['gecko_id']
                average_not_small_df['llama_id'] = dictionary[token_key]['llama_id']
                average_not_small_df['category'] = dictionary[token_key]['category']
                average_not_small_df['address'] = dictionary[token_key]['address']
                average_not_small_df['chain'] = chain_key
                average_not_small_df['price'] = dictionary[token_key]['prices'].values()
                average_not_small_df['market_cap'] = dictionary[token_key]['market_caps'].values()
                average_not_small_df['volume'] = dictionary[token_key]['volumes'].values()
                average_not_small_df['TVL'] = dictionary[token_key]['TVL'][chain_key].values()
                long_df = pd.concat([long_df, average_not_small_df])
                
        except AttributeError:
            print(token_key)
            average_not_small_df = pd.DataFrame(columns = df_columns)
            average_not_small_df['date'] = list(dictionary[token_key]['prices'].keys())
            average_not_small_df['symbol'] = dictionary[token_key]['symbol']
            average_not_small_df['gecko_id'] = dictionary[token_key]['gecko_id']
            average_not_small_df['llama_id'] = dictionary[token_key]['llama_id']
            average_not_small_df['category'] = dictionary[token_key]['category']
            average_not_small_df['chain'] = None
            average_not_small_df['price'] = dictionary[token_key]['prices'].values()
            average_not_small_df['market_cap'] = dictionary[token_key]['market_caps'].values()
            average_not_small_df['volume'] = dictionary[token_key]['volumes'].values()
            average_not_small_df['TVL'] = None
            long_df = pd.concat([long_df, average_not_small_df])
        
        long_df = long_df.reset_index(drop = True)
        
    return long_df     

# Главная функция

def get_full_data(start_date, end_date):
    
    # Достаем TVL
    llama_data = get_all_llama_slugs()
    TVL_dict = get_available_TVL_from_DefiLlama(llama_data = llama_data, start_date = start_date, end_date = end_date)
    
    
    # Достаем PMCV
    PMCV_dict, missing_tokens_from_gecko = download_full_gecko_data(gecko_info = llama_data, 
                                                                    start_date = start_date, 
                                                                    end_date = end_date)
    
    # Составляем словарь
    BIG_DICT = {}
    count_protocol = 0

    # Добавляем протоколы, которые есть в DefiLlama
    for i in range(len(llama_data)):

        key_token = llama_data['symbol'][i]
        
        if key_token in missing_tokens_from_gecko:
            continue    
            
        if (i != len(llama_data) - 1) and (i != 0): 
            if (llama_data['symbol'][i] == llama_data['symbol'][i + 1]):
                count_protocol += 1
                key_token = llama_data['symbol'][i] + '_' + str(count_protocol)
            elif (llama_data['symbol'][i] == llama_data['symbol'][i - 1]):
                count_protocol += 1
                key_token = llama_data['symbol'][i] + '_' + str(count_protocol)
                count_protocol = 0
            else:
                count_protocol = 0

        symbol = llama_data['symbol'][i]
        name = llama_data['name'][i]
        address = llama_data['address'][i]
        category = llama_data['category'][i]
        llama_id = llama_data['slug'][i]
        gecko_id = llama_data['gecko_id'][i]
        
        mini_dict = {'symbol': symbol, 'name': name, 'address': address, 'category': category, 'llama_id': llama_id, 'gecko_id': gecko_id}
        BIG_DICT[key_token] = mini_dict
        
        BIG_DICT[key_token]['prices'] = PMCV_dict[symbol]['prices']
        BIG_DICT[key_token]['market_caps'] = PMCV_dict[symbol]['market_caps']
        BIG_DICT[key_token]['volumes'] = PMCV_dict[symbol]['volumes']
        BIG_DICT[key_token]['TVL'] = TVL_dict[key_token]
        
    final_dict = sort_dict_by_keys(BIG_DICT)
    final_dataframe = dict_to_dataframe(final_dict)
    
    final_dataframe.loc[final_dataframe['symbol'] == 'CELR', 'address'] = '0x4f9254c83eb525f9fcf346490bbb3ed28a81c667'
    final_dataframe.loc[final_dataframe['symbol'] == 'FLM', 'address'] = 'neo:4d9eab13620fe3569ba3b0e56e2877739e4145e3'
    final_dataframe.loc[final_dataframe['symbol'] == 'ION', 'address'] = 'cosmos:uion'
    final_dataframe.loc[final_dataframe['symbol'] == 'ORAIX', 'address'] = '0x4c11249814f11b9346808179Cf06e71ac328c1b5'
    final_dataframe.loc[final_dataframe['symbol'] == 'SRM', 'address'] = '0x476c5e26a75bd202a9683ffd34359c0cc15be0ff'
    final_dataframe.loc[final_dataframe['symbol'] == 'STRD', 'address'] = 'osmosis:ibc/A8CA5EE328FA10C9519DF6057DA1F69682D28F7D0F5CCC7ECB72E3DCA2D157A4'
    final_dataframe.loc[final_dataframe['symbol'] == 'SUNDAE', 'address'] = 'cardano:9a9693a9a37912a5097918f97918d15240c92ab729a0b7c4aa144d7753554e444145'
    final_dataframe.loc[final_dataframe['symbol'] == 'VC', 'address'] = 'zksync:0x85d84c774cf8e9ff85342684b0e795df72a24908'
    final_dataframe.loc[final_dataframe['symbol'] == 'ZWAP', 'address'] = 'zilliqa:zil1p5suryq6q647usxczale29cu3336hhp376c627'
    
    return final_dataframe




"""
# Usage example
yesterday_date = (datetime.today() - timedelta(days = 1)).strftime('%Y-%m-%d')
full_dataframe = get_full_data(start_date = '2021-08-31', end_date = yesterday_date)
"""
