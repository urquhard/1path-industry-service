import asyncio
from sqlalchemy import create_engine
import datetime
from datetime import timedelta
import pandas as pd
from ast import literal_eval
from web3 import Web3
from web3.middleware import geth_poa_middleware
import json

from defi.DataDownloader import all_together
from settings.env import env
from defi.data import data
from defi.model import factor_model

engine = create_engine(f"postgresql://{env.username}:{env.password}@{env.host}:5432/{env.database}")

def update_data():
    print("Started")
    startDate = '2021-08-31'
    endDate = (datetime.datetime.today() - timedelta(days = 1)).strftime('%Y-%m-%d')
    oldLlamaData = pd.read_csv("defi/DefiLlamaData.csv")
    chainData = pd.read_csv("defi/chainIdMapping.csv")
    tokenAddressesDF = pd.read_csv('defi/tokenAddresses.csv', index_col = 0, converters = {'addresses': literal_eval})
    tokenAddressesDict = dict(zip(tokenAddressesDF.index, tokenAddressesDF['addresses']))

    full_dataframe = all_together(oldLlamaDF = oldLlamaData, chain_data = chainData, addresses_dict = tokenAddressesDict, start_date = startDate, end_date = endDate)
    with engine.begin() as connection:    
        full_dataframe.to_sql("test_data_3", con=connection, if_exists="replace", chunksize=100, method="multi")
    full_dataframe.to_csv('defi/NASRAL.csv', index = False)
    print("DONE-1")
    """
    print("Started")
    oldLlamaData = pd.read_csv('defi/DefiLlamaData.csv', converters = {'id_collection': str})
    tokenAddressesDF = pd.read_csv('defi/tokenAddresses.csv', index_col = 0, converters = {'addresses': literal_eval})
    tokenAddressesDict = dict(zip(tokenAddressesDF.index, tokenAddressesDF['addresses']))
    mapping_df = pd.read_csv('defi/chainIdMapping.csv')
    mapping_df['llama_id'] = mapping_df['llama_id'].apply(lambda x: literal_eval(x) if str(x)[0] == '[' else str(x))

    yesterday_date = (datetime.datetime.today() - datetime.timedelta(days = 1)).strftime('%Y-%m-%d')
    full_dataframe = get_full_data(oldLlama_df = oldLlamaData, start_date = '2021-08-31', end_date = yesterday_date, chain_data=mapping_df, addresses_dict=tokenAddressesDict)
    
    full_dataframe.to_sql('test_data_3', engine, if_exists='replace', chunksize=100, method="multi")
    print("DONE-1")
    """

def update_weights():
    print("Started updating weights")
    df = pd.read_sql_table("test_data_3", engine, columns=["index", "date", "symbol", "gecko_id", "llama_id", "category", "chain", "address", "price", "market_cap", "volume", "TVL"])
    
    df = data.data_preparation(df)
    ret_copy, price_copy, mar_cap = data.filter_data(df)
    Beta, categories_columns, blockchain_columns, style_factors = data.beta_preparation(df, ret_copy, price_copy, mar_cap)
    industry_returns, x_frame, fitted_frame, resid_frame, p_value_data, res_test, vif_values, se, geckoId_to_symbol = factor_model(Beta, categories_columns, blockchain_columns, style_factors, "power", 0.25)
    res = []
    with engine.begin() as connection:    
        res.append(Beta.to_sql("beta-1", con=connection, if_exists='replace', chunksize=100, method="multi"))
        res.append(industry_returns.to_sql("industry_returns-1", con=connection, if_exists='replace', chunksize=100, method="multi"))
        res.append(x_frame.to_sql("x_dict-1", con=connection, if_exists='replace', chunksize=100, method="multi"))
        res.append(resid_frame.to_sql("resid_frame-1", con=connection, if_exists='replace', chunksize=100, method="multi"))
        res.append(p_value_data.to_sql("p_value_data-1", con=connection, if_exists='replace', chunksize=100, method="multi"))
        res.append(res_test.to_sql("res_test-1", con=connection, if_exists='replace', chunksize=100, method="multi"))
        res.append(vif_values.to_sql("vif_values-1", con=connection, if_exists='replace', chunksize=100, method="multi"))
        res.append(se.to_sql("se-1", con=connection, if_exists='replace', chunksize=100, method="multi"))
        res.append(fitted_frame.to_sql("fitted_frame-1", con=connection, if_exists='replace', chunksize=100, method="multi"))
        res.append(geckoId_to_symbol.to_sql("geckoId_to_symbol-1", con=connection, if_exists='replace', chunksize=100, method="multi"))
        res.append(ret_copy.to_sql("ret_copy-1", con=connection, if_exists='replace', chunksize=100, method="multi"))
    print("Ended updating weights")

def update_shares():
    web3 = Web3(Web3.HTTPProvider(f"{env.bsc_http_provider_url}"))
    web3.middleware_onion.inject(geth_poa_middleware, layer=0)
    with open("VaultV1.json") as f:
        VaultABI = json.load(f)
    Vault = web3.eth.contract(address="0xbe9080Fe628F073633DC2dcFA9d3CC0cc38D4805", abi=VaultABI["abi"])
    sh_frame = pd.DataFrame(index = [], columns = ['1 share dollar value'])
    try:
        sh_frame.loc[pd.to_datetime(datetime.now()).strftime('%Y-%m-%d %H-%M-%S')] = \
        round(Vault.functions.exchangeRate().call() / 10 ** 18, 3)
    except:
        sh_frame.loc[pd.to_datetime(datetime.now()).strftime('%Y-%m-%d %H-%M-%S')] = \
        round(Vault.functions.exchangeRate().call() / 10 ** 18, 3)
    sh_frame.to_sql("share_perf_frame", engine, if_exists='append')
        
# add timedelta depending on env.delay
# add index validation
# TODO add try except validation for each action
# TODO add one day downloader
async def update_data_scheduler() -> None:
    """
    Used for scheduling updates of TVL in postgres
    Time for update is located in .env file

    Returns
    -------
    None.

    """
    while True:
        update_data()
        
        await asyncio.sleep(env.finance_data_delay)
        
async def update_weights_scheduler() -> None:
    """
    Used for scheduling updates of TVL in postgres
    Time for update is located in .env file

    Returns
    -------
    None.

    """
    while True:
        update_weights()
        
        await asyncio.sleep(env.finance_data_delay)

async def update_shares_scheduler() -> None:
    """
    Used for scheduling updates of shares in postgres
    Time for update is located in .env file

    Returns
    -------
    None.

    """
    while True:
        update_shares()
        
        await asyncio.sleep(env.shares_data_delay)
        

async def main():
    """
    Main asyncio function. Initializes TVL and sets it into postgres,
    then updating TVL. 

    Returns
    -------
    None.

    """

    
    while True:
        await asyncio.gather(
            update_data_scheduler(),
            update_weights_scheduler(),
            update_shares_scheduler()
        )

if __name__ == "__main__":
    asyncio.run(main())

