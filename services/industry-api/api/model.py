import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
from datetime import timedelta
import json

from database import get_db_session
from models import IndustryReturns, PValueData, ResTest, SE, VifValues, XDict, Shares, Returns


model_router = APIRouter()

logger = logging.getLogger(__name__)

def get_model_data_by_date(db: Session, table, start_date: str, end_date: str):
    filters = []
    
    if start_date != "None":
        filters.append(table.index >= start_date)
    if end_date != "None":
        filters.append(table.index <= end_date)
    
    if len(filters) != 0:
        return db.query(table).filter(*filters).all()

    return db.query(table).all()

@model_router.get("/returns")
def get_returns(db: Session = Depends(get_db_session), start_date="2022-01-01"):
    target_weights_dict = {
                                  'uniswap': 0.530646,
                                  'pancakeswap-token': 0.418644,
                                  'woo-network': 0.050710
                             }
    performance_frame = pd.DataFrame(0.0, index = [(pd.to_datetime(start_date) - timedelta(days = 1)).strftime('%Y-%m-%d')],
                                        columns = list(target_weights_dict.keys()))
    table = Returns
    query = db.query(table).filter(table.index >= start_date)
    returns_frame = pd.read_sql(query.statement, query.session.bind)
    returns_frame.set_index("index", inplace=True)
    for day in returns_frame.loc[start_date:].index:
        for token in performance_frame.columns:
            performance_frame.loc[day, token] = returns_frame.loc[day][token] * target_weights_dict[token]

    per_to_plot = pd.DataFrame(index = list(performance_frame.index)[1:], columns = ['performance'])
    per_to_plot['performance'] = list(performance_frame.sum(axis=1).add(1).cumprod().shift(1).iloc[1:])
    return per_to_plot

@model_router.get("/weights")
def get_weights():
    tokens_dict = {'UNI': '0xBf5140A22578168FD562DCcF235E5D43A02ce9B1',
     'CAKE': '0x0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82',
     'WOO': '0x4691937a7508860F876c9c0a2a617E7d9E945D4B'}
    
    from web3 import Web3
    
    minABI = [
                        {
                            "constant": True,
                            "inputs": [{'name': "_owner", 'type': "address"}],
                            "name": "balanceOf",
                            "outputs": [{'name': "balance", 'type': "uint256"}],
                            "type": "function",
                        },
                        {
                            "constant": True,
                             "inputs":[],
                             "name":"decimals",
                             "outputs":[{"name":"","type":"uint8"}],
                             "payable": False,
                             "stateMutability":"view",
                             "type":"function"
                        }
                    ]
    
    web3 = Web3(Web3.HTTPProvider("https://bsc-mainnet.nodereal.io/v1/58416516ddbb492a8a9acd27ee7c09cd"))
    
    from pycoingecko import CoinGeckoAPI
    cg = CoinGeckoAPI()
    suka = cg.get_price(ids=['pancakeswap-token', 'uniswap', 'woo-network'], vs_currencies = 'usd')
    
    quotes = {"CAKE": suka["pancakeswap-token"]["usd"], "UNI": suka["uniswap"]["usd"], "WOO": suka["woo-network"]["usd"]}
    
    dollar_value_to_token = {}
    for token in tokens_dict.keys():
        try:
            contract = web3.eth.contract(address=tokens_dict[token], abi=minABI)
            token_balance = contract.functions.balanceOf('0x2b50BCd2A3f3568dCad84Efbc38f908e49a6F463').call()
            dec = contract.functions.decimals().call()
            dollar_value_to_token[token] = token_balance / 10 ** dec * quotes[token]
        except Exception as e:
            print(e)
            try:
                contract = web3.eth.contract(address=tokens_dict[token], abi=minABI)
                token_balance = contract.functions.balanceOf('0x2b50BCd2A3f3568dCad84Efbc38f908e49a6F463').call()
                dec = contract.functions.decimals().call()
                dollar_value_to_token[token] = token_balance / 10 ** dec * quotes[token]
            except:
                pass
    s = 0
    for val in dollar_value_to_token.values():
        s += val
        
    for key, val in dollar_value_to_token.items():
        dollar_value_to_token[key] /= s
    
    return dollar_value_to_token

@model_router.get("/shares_count")
def get_shares_count():
    tokens_dict = {'UNI': '0xBf5140A22578168FD562DCcF235E5D43A02ce9B1',
     'CAKE': '0x0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82',
     'WOO': '0x4691937a7508860F876c9c0a2a617E7d9E945D4B'}
    
    from web3 import Web3
    
    with open("VaultV1.json") as f:
        vault_abi = json.load(f)
    web3 = Web3(Web3.HTTPProvider("https://bsc-mainnet.nodereal.io/v1/58416516ddbb492a8a9acd27ee7c09cd"))
    Vault = web3.eth.contract(address="0xbe9080Fe628F073633DC2dcFA9d3CC0cc38D4805", abi=vault_abi["abi"])
    total_share_supply = round(Vault.functions.totalSupply().call() / 10**18, 3)
    
    
    return total_share_supply

@model_router.get("/{table_name}")
def api_data(table_name: str, start_date: str = "None",
             end_date: str = "None", db: Session = Depends(get_db_session)):
    if table_name == "IndustryReturns":
        table = IndustryReturns
    elif table_name == "PValueData":
        table = PValueData
    elif table_name == "ResTest":
        table = ResTest
    elif table_name == "SE":
        table = SE
    elif table_name == "VifValues":
        table = VifValues
    elif table_name == "XDict":
        table = XDict
    elif table_name == "Shares":
        table = Shares
    elif table_name == "Returns":
        table = Returns
    else: 
        raise HTTPException(status_code=404, detail="Table not found")
    data = get_model_data_by_date(db, table, start_date, end_date)
    return data

