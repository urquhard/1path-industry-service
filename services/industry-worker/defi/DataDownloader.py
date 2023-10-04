import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from datetime import timedelta
from datetime import datetime
import time
import copy
from ast import literal_eval

from .addGeckoData import addGeckoData

import warnings
warnings.filterwarnings('ignore')


def get_coin_addresses(id):
    try:
        url = "https://api.coingecko.com/api/v3/coins/" + id
        request = requests.get(url).json()
        platform_dict = request["platforms"]
        return platform_dict
    except KeyError:
        print(id + ':', request)
        return {'': ''}


def filter_valid_addresses(token_info_series):
    flag = True
    if token_info_series['address'] == np.nan:
        flag = False
    if str(token_info_series['address'])[-1] == '-':
        flag = False
    split_address = str(token_info_series['address']).split(':')
    if len(split_address) == 1:
        if split_address[0][ : 2] != '0x':
            flag = False
    return flag


def fill_token_addresses(badTokens, tokenAddresses_dict, chain_data):
    for i, id in enumerate(badTokens['gecko_id'].unique()):
        address_minidict = tokenAddresses_dict[id]
        # print(address_minidict)
        flag = 0
        for key in reversed(address_minidict.keys()):
            value = address_minidict[key]

            # case 0: (chain == '')
            if key == '':

                # case 0.0: (chain == '') and (address == '') --- EQ
                if value == '':
                    try:
                        chainName = badTokens['chain'][badTokens['gecko_id'] == id].iloc[0]
                        chainID = chain_data[chain_data['name'] == chainName]['gecko_id'].iloc[0]
                        awfulAddress = chainID + ':-'
                    except IndexError:
                        cringeAddress = '-'


                # case 0.1: (chain == '') and (address != '') --- ZWAP
                else:
                    try:
                        chainName = badTokens['chain'][badTokens['gecko_id'] == id].iloc[0]
                        chainID = chain_data[chain_data['name'] == chainName]['gecko_id'].iloc[0]
                        fineAddress = chainID + ':' + value
                    except IndexError:
                        fineAddress = 'undefined-chain:' + value

                    
            # case 1: (chain != '')
            if key != '':
                geckoChainName = chain_data[chain_data['gecko_id'] == key]['name'].iloc[0]
                llamaChainNames = list(badTokens['chain'][badTokens['gecko_id'] == id])
                try:
                    key = chain_data[chain_data['name'] == geckoChainName]['new_id'].iloc[0]
                except IndexError:
                    key = key
            
                # case 1.0: (chain != '') and (address == '') --- FLM
                if value == '':
                    try:
                        chainName = badTokens['chain'][badTokens['gecko_id'] == id].iloc[0]
                        chainID = chain_data[chain_data['name'] == chainName]['gecko_id'].iloc[0]
                        badAddress = chainID + ':-'
                    except IndexError:
                        badAddress = key + ':-'

                # case 1.1: (chain != '') and (address != '')
                if value != '':

                    # case 1.1.0: chain is NOT the same as in dataframe --- CELR
                    goodAddress = key + ':' + value

                    if geckoChainName in llamaChainNames:

                        # case 1.1.1: (chain != '') and chain is the same as in dataframe --- ABEL
                        perfectAddress = key + ':' + value

            #         else:
            #             continue
            #     else:
            #         continue
            # else:
            #     continue
        
        try:
            badTokens['address'][badTokens['gecko_id'] == id] = perfectAddress
        except NameError:
            try:
                badTokens['address'][badTokens['gecko_id'] == id] = goodAddress
            except NameError:
                try:
                    badTokens['address'][badTokens['gecko_id'] == id] = fineAddress
                except NameError:
                    try:
                        badTokens['address'][badTokens['gecko_id'] == id] = badAddress
                    except NameError:
                        try:
                            badTokens['address'][badTokens['gecko_id'] == id] = awfulAddress
                        except NameError:
                            badTokens['address'][badTokens['gecko_id'] == id] = cringeAddress

        if 'perfectAddress' in locals():
            del perfectAddress
        if 'goodAddress' in locals():
            del goodAddress
        if 'fineAddress' in locals():
            del fineAddress
        if 'badAddress' in locals():
            del badAddress
        if 'awfulAddress' in locals():
            del awfulAddress
        if 'cringeAddress' in locals():
            del cringeAddress

    # print(key, '-', geckoChainName)
    return badTokens


def update_good_address(address, mapping):
    address_split = str(address).split(':')
    if len(address_split) == 1:
        address = 'ethereum:' + address
    else:
        llama_chain = address_split[0]
        # print(llama_chain, mapping[llama_chain])
        try:
            address = mapping[llama_chain] + ':' + ':'.join(address_split[1:])
            # if len(address_split) > 2:
            #     print(address)
        except KeyError:
            address = address
    return address


def get_new_slugs(oldLlamaDF):

    url = 'https://api.llama.fi/protocols'
    request = requests.get(url).json()
    llama_data = pd.DataFrame(request)
    llama_data = llama_data[['name', 'address', 'symbol', 'chain', 'slug', 'gecko_id', 'category', 'tvl']]
    llama_data = llama_data[llama_data['tvl'] >= 1000000].sort_values('symbol')    # [llama_data['symbol'].isin(token_list)]
    llama_data = llama_data[llama_data['symbol'] != 'BNB'][llama_data['symbol'] != 'MATIC'][llama_data['symbol'] != '-'].reset_index(drop = True)
    llama_data = llama_data[~llama_data['slug'].isin(oldLlamaDF['slug'])].reset_index(drop = True)
    llama_data = addGeckoData(llama_data)
    llama_data['update_date'] = datetime.today().strftime('%Y-%m-%d')
    return llama_data


def updateLlamaDataFrame(oldLlamaDF, chain_data, addresses_dict, update_file = False):

    # Get mapping chain dict
    shorty = copy.deepcopy(chain_data[['llama_id', 'new_id']])
    count = len(shorty)
    for i in range(len(shorty)):
        llama_id = shorty['llama_id'][i]
        new_id = shorty['new_id'][i]
        if type(llama_id) == list:
            for id in llama_id:
                shorty.loc[count] = [id, new_id]
                count += 1
            # print(llama_id)
            shorty.drop(i, inplace = True)
            i -=1  
    shorty = shorty.sort_values('new_id').reset_index(drop = True)
    mapping_dict = dict(zip(shorty['llama_id'], shorty['new_id']))

    # Get new tokens
    newLlamaDF = get_new_slugs(oldLlamaDF = oldLlamaDF)

    if len(newLlamaDF) == 0:
        return oldLlamaDF

    # Separate good addresses from bad
    nanTokens = newLlamaDF[newLlamaDF['gecko_id'].isna()]
    notNanTokens = newLlamaDF[~newLlamaDF['gecko_id'].isna()]
    badAddressTokens = notNanTokens[~notNanTokens.apply(filter_valid_addresses, axis = 1)].reset_index(drop = True)
    goodAddressTokens = pd.concat([notNanTokens, badAddressTokens]).drop_duplicates(keep = False).reset_index(drop = True)
    
    # Update token addresses
    badAddressTokens = fill_token_addresses(badTokens = badAddressTokens, tokenAddresses_dict = addresses_dict, chain_data = chain_data) 
    goodAddressTokens['address'] = goodAddressTokens['address'].apply(lambda x: update_good_address(x, mapping_dict))

    updatedTokenData = pd.concat([nanTokens, goodAddressTokens, badAddressTokens]).sort_values('symbol').reset_index(drop = True)
    newLlamaDF = pd.concat([updatedTokenData, oldLlamaDF]).sort_values('symbol').reset_index(drop = True)
    
    if update_file == True:
        newLlamaDF.to_csv('DefiLlamaData.csv', index = 0)

    return newLlamaDF


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
    missing_token_ids = []
    
    chain_tokens_series = pd.Series(['Ethereum', 'Binance', 'Polygon', 'Avalanche'], \
                                index = ['ETH', 'BNB', 'MATIC', 'AVAX'])

    count_protocol = 0
    tvl_dict = {}
    print('Loading TVL of all protocols')
    for i in tqdm(range(len(llama_data))):

        ID = llama_data['slug'][i]
        token = llama_data['symbol'][i]
        if token in list(chain_tokens_series.index):
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
        try:
            request = requests.get(llama_tvl_endpoint + ID).json()
            chain_list = list(request['chainTvls'].keys())
        except KeyError:
            missing_token_ids.append(ID)
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

    print()
    print('Loading TVL of necessary chains')
    for chain_token in tqdm(chain_tokens_series.index):
        mini_dict = {}
        llama_chain_endpoint = 'https://api.llama.fi/v2/historicalChainTvl/' + chain_tokens_series[chain_token]
        request = requests.get(llama_chain_endpoint).json()
        chain_tvls = pd.DataFrame(request)
        chain_tvls['date'] = pd.to_datetime(chain_tvls['date'], unit = 's')
        chain_tvls['date'] = chain_tvls['date'].apply(lambda x: x.round(freq = 'D'))
        chain_tvl_series = pd.Series(chain_tvls['tvl'].values, index = chain_tvls['date'])
        cutted_chain_tvls = cut_and_fill_series(chain_tvl_series, start_date, end_date)
        filled_chain_tvls = fill_series_with_nans(cutted_chain_tvls, start_date, end_date)
        filled_chain_tvls.index = filled_chain_tvls.index.strftime('%Y-%m-%d')
        mini_dict[chain_tokens_series[chain_token]] = filled_chain_tvls.to_dict()
        tvl_dict[chain_token] = mini_dict
    return tvl_dict, missing_token_ids


# Функция достает PMCV протокола по gecko_id
def get_token_data_from_coingecko(gecko_id):
    
    url = 'https://api.coingecko.com/api/v3/coins/' + str(gecko_id) + '/market_chart?vs_currency=usd&days=max&interval=daily'
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
                average_not_small_df['id_collection'] = dictionary[token_key]['id_collection']
                average_not_small_df['update_date'] = dictionary[token_key]['update_date']
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
            average_not_small_df['id_collection'] = dictionary[token_key]['id_collection']
            average_not_small_df['update_date'] = dictionary[token_key]['update_date']
            average_not_small_df['price'] = dictionary[token_key]['prices'].values()
            average_not_small_df['market_cap'] = dictionary[token_key]['market_caps'].values()
            average_not_small_df['volume'] = dictionary[token_key]['volumes'].values()
            average_not_small_df['TVL'] = None
            long_df = pd.concat([long_df, average_not_small_df])
        
        long_df = long_df.reset_index(drop = True)
        
    return long_df     


# Главная функция
def get_full_data(oldLlama_df, start_date, end_date, chain_data, addresses_dict, update_file = True):
    
    chain_tokens = ['ETH', 'BNB', 'MATIC', 'AVAX']

    # Достаем TVL
    # llama_data = get_all_llama_slugs()
    llama_data = updateLlamaDataFrame(oldLlamaDF = oldLlama_df, chain_data = chain_data, addresses_dict = addresses_dict, update_file = update_file)
    llama_data = llama_data[~llama_data['gecko_id'].isna()].reset_index(drop = True)

    TVL_dict, missing_token_ids = get_available_TVL_from_DefiLlama(llama_data = llama_data, start_date = start_date, end_date = end_date)
    
    # Достаем PMCV
    PMCV_dict, missing_tokens_from_gecko = download_full_gecko_data(gecko_info = llama_data, start_date = start_date, end_date = end_date)
    
    # Составляем словарь
    BIG_DICT = {}
    count_protocol = 0

    # Добавляем протоколы, которые есть в DefiLlama
    for i in range(len(llama_data)):
        llama_id = llama_data['slug'][i]
        
        key_token = llama_data['symbol'][i]
        
        if key_token in missing_tokens_from_gecko:
            continue
        if llama_id in missing_token_ids:
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
        gecko_id = llama_data['gecko_id'][i]
        id_collection = llama_data['id_collection'][i]
        update_date = llama_data['update_date'][i]
        
        mini_dict = {'symbol': symbol, 'name': name, 'address': address, 'category': category, 'llama_id': llama_id, \
                     'gecko_id': gecko_id, 'id_collection': id_collection, 'update_date': update_date}
        BIG_DICT[key_token] = mini_dict
        
        BIG_DICT[key_token]['prices'] = PMCV_dict[symbol]['prices']
        BIG_DICT[key_token]['market_caps'] = PMCV_dict[symbol]['market_caps']
        BIG_DICT[key_token]['volumes'] = PMCV_dict[symbol]['volumes']
        BIG_DICT[key_token]['TVL'] = TVL_dict[key_token]  

    final_dict = sort_dict_by_keys(BIG_DICT)
    final_dataframe = dict_to_dataframe(final_dict)
    
    return final_dataframe


### Начало работы большой функции

# oldLlamaData = pd.read_csv('DefiLlamaData.csv', converters = {'id_collection': str})
# tokenAddressesDF = pd.read_csv('tokenAddresses.csv', index_col = 0, converters = {'addresses': literal_eval})
# tokenAddressesDict = dict(zip(tokenAddressesDF.index, tokenAddressesDF['addresses']))
# mapping_df = pd.read_csv('chainIdMapping.csv')
# mapping_df['llama_id'] = mapping_df['llama_id'].apply(lambda x: literal_eval(x) if str(x)[0] == '[' else str(x))

# yesterday_date = (datetime.today() - timedelta(days = 1)).strftime('%Y-%m-%d')
# full_dataframe = get_full_data(oldLlama_df = oldLlamaData, start_date = '2021-08-31', end_date = yesterday_date)

# full_dataframe.to_csv('AUGUST_23_BIG_DATA.csv', index = False)


# ### JSON, который нужен для телеги

# DNV = pd.read_csv('AUGUST_23_BIG_DATA.csv')
# DNV = data.data_preparation(DNV)
# # returns, prices, mcaps = data.filter_data(DNV)
# # Beta, categories_columns, blockchain_columns, style_factors = \
# # data.beta_preparation(DNV, returns, prices, mcaps, quarantine_period = 45, momentum_period = [7, 30], multiple_momentum = False)

def get_telegram_dictionary(bigData, tokenAddresses):
    teleDict = {}
    mapCorrectNames = {'Ethereum': 'ethereum', 'Binance': 'binance-smart-chain', 'Polygon': 'polygon-pos', 'Avalanche': 'avalanche'}
    for chainName in mapCorrectNames.keys():
        chainTokenData = bigData[bigData['mod_chain'] == chainName]
        chainTokenInfoDict = []
        for geckoID in chainTokenData['gecko_id'].unique():
            symbol = chainTokenData[chainTokenData['gecko_id'] == geckoID]['symbol'].iloc[-1]
            price = chainTokenData[chainTokenData['gecko_id'] == geckoID]['price'].iloc[-2]
            try:
                tokenInfo = tokenAddresses[geckoID]
            except KeyError:
                print('Token is not in tokenAddresses dictionary:', geckoID, chainName)
                continue
            microDict = {}
            try:
                microDict['address'] = tokenInfo[mapCorrectNames[chainName]]['contract_address']
                microDict['decimals'] = tokenInfo[mapCorrectNames[chainName]]['decimal_place']
            except KeyError:
                print('Token does not exist in tokenAddresses dictionary in certain chain:', geckoID, chainName)
                continue
            microDict['symbol'] = symbol
            microDict['gecko_id'] = geckoID
            microDict['usd_price'] = price
            chainTokenInfoDict.append(microDict)
        teleDict[mapCorrectNames[chainName]] = chainTokenInfoDict
    return teleDict

# TeleDict = get_telegram_dictionary(DNV, tokenAddressesDict)
# with open('TeleDict.json', 'w') as f:
#     json.dump(TeleDict, f)