import numpy as np
import pandas as pd
import requests
import json
from tqdm import tqdm
from datetime import timedelta
from datetime import datetime
from ast import literal_eval
import time
import copy
import gc

from .data import data
from .addGeckoData import addGeckoData

import warnings
warnings.filterwarnings('ignore')


def get_coin_addresses(id):
    try:
        url = "https://api.coingecko.com/api/v3/coins/" + id
        request = requests.get(url).json()
        platform_dict = request["detail_platforms"]
        return platform_dict
    except KeyError:
        print("Function: get_coin_addresses")
        print(id + ':', request, '\n')
        return {'': {'decimal_place': None, 'contract_address': ''}}


def filter_valid_addresses(addressDF):
    flag = True
    if (addressDF['address'] == np.nan) or (str(addressDF['address'])[-1] == '-'):
        flag = False
    splitAddress = str(addressDF['address']).split(':')
    if (len(splitAddress) == 1) and (splitAddress[0][ : 2] != '0x'):
        flag = False
    return flag


def update_good_address(address, mapping):
    splitAddress = str(address).split(':')
    if len(splitAddress) == 1:
        address = 'ethereum:' + address
    else:
        llama_chain = splitAddress[0]
        try:
            address = mapping[llama_chain] + ':' + ':'.join(splitAddress[1:])
        except KeyError:
            pass
    return address


def fill_token_addresses(badTokens, tokenAddresses_dict, chain_data):
    for i, id in enumerate(badTokens['gecko_id'].unique()):
        address_minidict = tokenAddresses_dict[id]
        flag = 0
        for key in reversed(address_minidict.keys()):
            value = address_minidict[key]['contract_address']
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
                try:
                    geckoChainName = chain_data[chain_data['gecko_id'] == key]['name'].iloc[0]
                except:
                    continue
                llamaChainNames = list(badTokens['chain'][badTokens['gecko_id'] == id])
                try:
                    key = chain_data[chain_data['name'] == geckoChainName]['new_id'].iloc[0]
                except IndexError:
                    pass
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

    return badTokens


def get_new_slugs(oldLlamaDF):
    chainTokens = ['ETH', 'BNB', 'MATIC', 'AVAX']
    url = 'https://api.llama.fi/protocols'
    request = requests.get(url).json()
    llama_data = pd.DataFrame(request)
    llama_data = llama_data[['name', 'address', 'symbol', 'chain', 'slug', 'gecko_id', 'category', 'tvl']]
    llama_data = llama_data[~(llama_data['symbol'].isin(chainTokens)) & (llama_data['symbol'] != '-')]
    llama_data = llama_data[llama_data['tvl'] >= 1000000].sort_values('symbol').reset_index(drop = True)
    llama_data = llama_data[~llama_data['slug'].isin(oldLlamaDF['slug'])].reset_index(drop = True)
    llama_data = addGeckoData(llama_data)
    llama_data['update_date'] = datetime.today().strftime('%Y-%m-%d')
    return llama_data


def updateLlamaDataFrame(oldLlamaDF, chain_data, addresses_dict, update_file = False):

    # Get mapping chain dict
    df = chain_data[['llama_id', 'new_id']].dropna()
    mapping_dict = dict(zip(df['llama_id'], df['new_id']))

    # Get new tokens
    newLlamaDF = get_new_slugs(oldLlamaDF = oldLlamaDF)
    if len(newLlamaDF) == 0:
        return oldLlamaDF

    # Separate good addresses from bad addresses
    nanTokens = newLlamaDF[newLlamaDF['gecko_id'].isna()]
    notNanTokens = newLlamaDF[~newLlamaDF['gecko_id'].isna()]
    badAddressTokens = notNanTokens[~notNanTokens.apply(filter_valid_addresses, axis = 1)].reset_index(drop = True)
    goodAddressTokens = pd.concat([notNanTokens, badAddressTokens]).drop_duplicates(keep = False).reset_index(drop = True)
    
    # Add new addresses if necessary
    if badAddressTokens.shape != (0,0):
        badAddressTokensList = list(badAddressTokens['gecko_id'])
        tokensInAddressesDict = list(addresses_dict.keys())
        tokensToAddToAddressesDict = [token for token in badAddressTokensList if token not in tokensInAddressesDict]
        if len(tokensToAddToAddressesDict) != 0:
            print('Adding new addresses...')
            loop = tqdm(tokensToAddToAddressesDict, ascii = "-#")
            for id in loop:
                loop.set_description(f"Token {id}")
                time.sleep(10)
                addresses = get_coin_addresses(id)
                addresses_dict[id] = addresses
            print('Done!\n')
            copy_addresses_dict = copy.deepcopy(addresses_dict)
            updatedAddressesDF = pd.DataFrame([copy_addresses_dict.keys(), copy_addresses_dict.values()], index = [None, 'addresses']).transpose().set_index(None).sort_index()
            updatedAddressesDF.to_csv('defi/tokenAddresses.csv')
    
        # Update token addresses
        badAddressTokens = fill_token_addresses(badTokens = badAddressTokens, tokenAddresses_dict = addresses_dict, chain_data = chain_data) 
    goodAddressTokens['address'] = goodAddressTokens['address'].apply(lambda x: update_good_address(x, mapping_dict))

    updatedTokenData = pd.concat([nanTokens, goodAddressTokens, badAddressTokens]).sort_values('symbol').reset_index(drop = True)
    newLlamaDF = pd.concat([updatedTokenData, oldLlamaDF]).sort_values('symbol').reset_index(drop = True)
    
    if update_file == True:
        newLlamaDF.to_csv('defi/DefiLlamaData.csv', index = 0)

    return newLlamaDF


def get_request_with_exceptions(full_endpoint, id, service):
    # timeoutCount = 0
    while True:
        request = requests.get(full_endpoint)

        if request.status_code == 200:
            request_json = request.json()
            return request_json

        elif request.status_code == 404:
            print(f"Error 404 occured: coin {id} not found")
            request_json = None
            if service == 'llama':
                initialLlamaData = pd.read_csv('defi/DefiLlamaData.csv')
                initialLlamaData = initialLlamaData[initialLlamaData['slug'] != id].reset_index(drop = True)
                initialLlamaData.to_csv('defi/DefiLlamaData.csv', index = 0)
            elif service == 'gecko':
                initialLlamaData = pd.read_csv('defi/DefiLlamaData.csv')
                ind = initialLlamaData.index[initialLlamaData['gecko_id'] == id][0]
                initialLlamaData['gecko_id'].loc[ind] = None
                initialLlamaData['id_collection'].loc[ind] = None
                initialLlamaData.to_csv('defi/DefiLlamaData.csv', index = 0)
            break

        elif (request.status_code == 429) and (service == 'gecko'):
            time.sleep(7)

        elif (request.status_code == 502):
            time.sleep(7)
        
        else:
            print(f'Something wrong with {id}')
            print(f'Request status code: {request.status_code}')
            print(f'Error: {request.text}\n')
            request_json = None
            break
    return request_json

def get_token_data_from_defillama(newLlamaDF, startDate, endDate):
    print('Downloading token TVL data from DefiLlama...')
    tvlDict = {}
    missingLlamaTokens = []
    url = "https://api.llama.fi/protocol/"
    loop = tqdm(newLlamaDF['slug'].dropna(), ascii = "-#")
    for id in loop:
        loop.set_description(f"Token {id}")
        try:
            rawSlugData = get_request_with_exceptions(full_endpoint = url + id, id = id, service = 'llama')['chainTvls']
        except TypeError:
            missingLlamaTokens.append(id)
            continue
        df = pd.DataFrame()
        for chain in rawSlugData.keys():
            minidict = dict(zip([pd.to_datetime(oneDayData['date'], unit = 's') for oneDayData in rawSlugData[chain]['tvl']],
                                [oneDayData['totalLiquidityUSD'] for oneDayData in rawSlugData[chain]['tvl']]))
            minidict.popitem()
            df = pd.concat([df, pd.Series(minidict, name = chain)], axis = 1)
        df = df.dropna(axis = 1, how = 'all').sort_index().reindex(pd.date_range(startDate, endDate), fill_value = np.nan)
        tvlDict[id] = df
    print("Done!\n")
    return tvlDict, missingLlamaTokens


def get_chain_data_from_defillama(newLlamaDF, startDate, endDate):
    print('Downloading chain TVL data from DefiLlama...')
    chainTvlDict = {}
    url = 'https://api.llama.fi/v2/historicalChainTvl/'
    loop = tqdm(newLlamaDF[newLlamaDF['slug'].isna()]['chain'], ascii = "-#")
    for id in loop:
        loop.set_description(f"Chain {id}")
        rawChainData = get_request_with_exceptions(full_endpoint = url + id, id = id, service = 'llama')
        if rawChainData == None:
            continue
        df = pd.DataFrame()
        minidict = dict(zip([pd.to_datetime(oneDayData['date'], unit = 's') for oneDayData in rawChainData],
                            [oneDayData['tvl'] for oneDayData in rawChainData]))
        df = pd.concat([df, pd.Series(minidict, name = id)], axis = 1)
        df = df.dropna(axis = 1, how = 'all').sort_index().reindex(pd.date_range(startDate, endDate), fill_value = np.nan)
        chainTvlDict[id] = df
    print("Done!\n")
    return chainTvlDict


def get_data_from_coingecko(newLlamaDF, startDate, endDate):
    print('Downloading token data from CoinGecko...')
    geckoDict = {}
    missingGeckoTokens = []
    loop = tqdm(newLlamaDF['gecko_id'].unique(), ascii = "-#")
    for id in loop:
        loop.set_description(f"Token {id}")
        time.sleep(7)
        url = 'https://api.coingecko.com/api/v3/coins/' + id + '/market_chart?vs_currency=usd&days=max&interval=daily'
        
        rawTokenData = pd.DataFrame(get_request_with_exceptions(full_endpoint = url, id = id, service = 'gecko'))
        if rawTokenData.shape == (0, 0):
            missingGeckoTokens.append(id)
            continue
    
        rawTokenData = pd.concat([pd.DataFrame(rawTokenData[column].to_list(), columns = [None, column]) for column in rawTokenData.columns], axis = 1)
        df = rawTokenData.loc[:, ~rawTokenData.columns.duplicated()].copy().set_index(None)[:-1]
        df.index = pd.to_datetime(df.index / 1000, unit = 's')
        df = df.dropna(axis = 1, how = 'all').sort_index().reindex(pd.date_range(startDate, endDate), fill_value = np.nan)
        geckoDict[id] = df
    print("Done!\n")
    return geckoDict, missingGeckoTokens


def merge_data(tokenInfoDF, tokenLlamaDict, chainLlamalDict, tokenGeckoDict, missingLlamaTokens, missingGeckoTokens):
    print('Merge collected data...')
    # gc.collect()
    llamaData = copy.deepcopy(tokenInfoDF[['name', 'address', 'symbol', 'slug', 'gecko_id', 'category', 'id_collection', 'update_date']])
    tokenProtocolsList = [df for _, df in llamaData.groupby('gecko_id')]
    loop = tqdm(tokenProtocolsList, ascii = "-#")
    nativeChainTokenSeries = pd.Series(['Avalanche', 'Binance', 'Ethereum', 'Polygon'],
                               index = ['avalanche-2', 'binancecoin', 'ethereum', 'matic-network'])
    # fullDataList = []
    fullDataDF = pd.DataFrame()
    for protocols in loop:
        gecko_id = protocols['gecko_id'].iloc[0]
        loop.set_description(f"Token {gecko_id}")
        df = pd.DataFrame()
        if gecko_id in missingGeckoTokens:
            continue
        elif gecko_id in nativeChainTokenSeries.index:
            chain = nativeChainTokenSeries[gecko_id]
            geckoDF = tokenGeckoDict[gecko_id].reset_index(drop = True)
            llamaDF = pd.melt(chainLlamalDict[chain], value_vars = [chain], \
                              value_name = 'TVL', var_name = 'chain', ignore_index = False).reset_index(names = 'date')
            df = pd.concat([pd.concat([protocols] * len(llamaDF)).reset_index(drop = True), geckoDF, llamaDF], axis = 1)
            fullDataDF = pd.concat([fullDataDF, df])
            # print("KALL", gecko_id)
        else:
            # minilist = []
            geckoDF = tokenGeckoDict[gecko_id].reset_index(drop = True)
            for i in range(len(protocols)):
                minidf = protocols[i : i + 1]
                llama_id = minidf['slug'].iloc[0]
                if llama_id in missingLlamaTokens:
                    continue
                llamaDF = pd.melt(tokenLlamaDict[llama_id], value_vars = tokenLlamaDict[llama_id].columns, \
                                  value_name = 'TVL', var_name = 'chain', ignore_index = False).reset_index(names = 'date')
                df = pd.concat([pd.concat([minidf] * len(llamaDF)).reset_index(drop = True), 
                                pd.concat([geckoDF] * len(tokenLlamaDict[llama_id].columns)).reset_index(drop = True), llamaDF], axis = 1)
                # minilist.append(df)
                fullDataDF = pd.concat([fullDataDF, df])
                # print("GOVNO", llama_id)
            # fullDataList.append(pd.concat(minilist).reset_index(drop = True))
    # fullDataDF = pd.concat(fullDataList).reset_index(drop = True)
    print("HUI")
    fullDataDF = fullDataDF[['date', 'symbol', 'gecko_id', 'slug', 'category', 'chain', 'address', \
                             'id_collection', 'update_date', 'prices', 'market_caps', 'total_volumes', 'TVL']]
    fullDataDF = fullDataDF.rename(columns = {'slug': 'llama_id', 'prices': 'price', 'market_caps': 'market_cap', 'total_volumes': 'volume'})
    print("Done!")
    return fullDataDF

'''
def merge_data(tokenInfoDF, tokenLlamaDict, chainLlamalDict, tokenGeckoDict, missingLlamaTokens, missingGeckoTokens):
    logger = []  # Use a list to store log messages

    logger.append('Merge collected data...')
    # llamaData = copy.deepcopy(tokenInfoDF[['name', 'address', 'symbol', 'slug', 'gecko_id', 'category', 'id_collection', 'update_date']])
    tokenProtocolsList = [df for _, df in tokenInfoDF.drop("chain").drop("tvl").groupby('gecko_id')]
    loop = tqdm(tokenProtocolsList, ascii="-#")
    
    nativeChainTokenSeries = pd.Series(['Avalanche', 'Binance', 'Ethereum', 'Polygon'],
                                       index=['avalanche-2', 'binancecoin', 'ethereum', 'matic-network'])

    fullDataList = []

    for protocols in loop:
        gecko_id = protocols['gecko_id'].loc[0]
        loop.set_description(f"Token {gecko_id}")

        if gecko_id in missingGeckoTokens:
            continue

        df = pd.DataFrame()
        geckoDF = tokenGeckoDict[gecko_id].reset_index(drop=True)

        if gecko_id in nativeChainTokenSeries.index:
            chain = nativeChainTokenSeries[gecko_id]
            df = pd.concat([pd.concat([protocols] * len(pd.melt(chainLlamalDict[chain], value_vars=[chain], value_name='TVL', var_name='chain',
                              ignore_index=False).reset_index(names='date'))).reset_index(drop=True), geckoDF, pd.melt(chainLlamalDict[chain], value_vars=[chain], value_name='TVL', var_name='chain',
                                                ignore_index=False).reset_index(names='date')], axis=1)
        else:
            minilist = []
            for i in range(len(protocols)):
                minidf = protocols[i:i + 1]
                llama_id = minidf['slug'].loc[0]
                if llama_id in missingLlamaTokens:
                    continue

                df = pd.concat([pd.concat([minidf] * len(pd.melt(tokenLlamaDict[llama_id], value_vars=tokenLlamaDict[llama_id].columns,
                                  value_name='TVL', var_name='chain', ignore_index=False).reset_index(names='date'))).reset_index(drop=True),
                                pd.concat([geckoDF] * len(tokenLlamaDict[llama_id].columns)).reset_index(drop=True),
                                pd.melt(tokenLlamaDict[llama_id], value_vars=tokenLlamaDict[llama_id].columns,
                                                  value_name='TVL', var_name='chain', ignore_index=False).reset_index(names='date')], axis=1)
                minilist.append(df)

            df = pd.concat(minilist).reset_index(drop=True)

        fullDataList.append(df)

    fullDataDF = pd.concat(fullDataList).reset_index(drop=True)
    fullDataDF = fullDataDF[['date', 'symbol', 'gecko_id', 'slug', 'category', 'chain', 'address',
                             'id_collection', 'update_date', 'prices', 'market_caps', 'total_volumes', 'TVL']]
    fullDataDF = fullDataDF.rename(columns={'slug': 'llama_id', 'prices': 'price', 'market_caps': 'market_cap',
                                            'total_volumes': 'volume'})

    logger.append("Done!")

    return fullDataDF, logger
'''

def all_together(oldLlamaDF, chain_data, addresses_dict, start_date, end_date, update_file = True):

    newLlamaData = updateLlamaDataFrame(oldLlamaDF = oldLlamaDF, chain_data = chain_data, addresses_dict = addresses_dict, update_file = update_file)
    newLlamaData = newLlamaData[~newLlamaData['gecko_id'].isna()].reset_index(drop = True)
    tokenLlamaData, missingLlamaTokens = get_token_data_from_defillama(newLlamaDF = newLlamaData, startDate = start_date, endDate = end_date)
    chainLlamaData = get_chain_data_from_defillama(newLlamaDF = newLlamaData, startDate = start_date, endDate = end_date)
    tokenGeckoData, missingGeckoTokens = get_data_from_coingecko(newLlamaDF = newLlamaData, startDate = start_date, endDate = end_date)
    finalData = merge_data(tokenInfoDF = newLlamaData, tokenLlamaDict = tokenLlamaData, chainLlamalDict = chainLlamaData, \
                           tokenGeckoDict = tokenGeckoData, missingGeckoTokens = missingGeckoTokens, missingLlamaTokens=missingLlamaTokens)
    return finalData

"""
startDate = '2021-08-31'
endDate = (datetime.today() - timedelta(days = 1)).strftime('%Y-%m-%d')
oldLlamaData = pd.read_csv("defi/DefiLlamaData.csv")
chainData = pd.read_csv("defi/chainIdMapping.csv")
tokenAddressesDF = pd.read_csv('defi/tokenAddresses.csv', index_col = 0, converters = {'addresses': literal_eval})
tokenAddressesDict = dict(zip(tokenAddressesDF.index, tokenAddressesDF['addresses']))

full_dataframe = all_together(oldLlamaDF = oldLlamaData, chain_data = chainData, addresses_dict = tokenAddressesDict, start_date = startDate, end_date = endDate)

full_dataframe.to_csv('defi/NASRAL.csv', index = False)
"""


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
"""
TeleDict = get_telegram_dictionary(bigData = DNV, tokenAddresses = tokenAddressesDict)
with open('defi/TeleDict.json', 'w') as f:
    json.dump(TeleDict, f)
    """