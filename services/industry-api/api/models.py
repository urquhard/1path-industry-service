from database import Base
from sqlalchemy import Column, Float, Integer, String, LargeBinary, DateTime


class CeleryTaskmeta(Base):
    __tablename__ = "celery_taskmeta"

    index = Column("id", Integer, primary_key=True)
    task_id = Column("task_id", String)
    status = Column("status", String)
    result = Column("result", LargeBinary)
    date_done = Column("date_done", DateTime)
    traceback = Column("traceback", String)
    name = Column("name", String)
    args = Column("args", LargeBinary)
    kwargs = Column("kwargs", LargeBinary)
    worker = Column("worker", String)
    retries = Column("retries", Integer)
    queue = Column("queue", String)

class IndustryReturns(Base):
    __tablename__ = "industry_returns-1"

    index = Column("index", DateTime, primary_key=True)
    const = Column("const", Float)
    Derivatives = Column("Derivatives", Float)
    Dexes = Column("Dexes", Float)
    Yield = Column("Yield", Float)
    Bridge = Column("Bridge", Float)
    Lending = Column("Lending", Float)
    Ethereum = Column("Ethereum", Float)
    Polygon = Column("Polygon", Float)
    Avalanche = Column("Avalanche", Float)
    Binance = Column("Binance", Float)
    factor_log_mc = Column("factor_log_mc", Float)
    factor_log_mc_tvl = Column("factor_log_mc_tvl", Float)
    VMR = Column("VMR", Float)
    factor_momentum = Column("factor_momentum", Float)
    other_chains = Column("other_chains", Float)
    other_categories = Column("other_categories", Float)
    
class PValueData(Base):
    __tablename__ = "p_value_data-1"

    index = Column("index", DateTime, primary_key=True)
    const = Column("const", Float)
    Derivatives = Column("Derivatives", Float)
    Dexes = Column("Dexes", Float)
    Yield = Column("Yield", Float)
    Bridge = Column("Bridge", Float)
    Lending = Column("Lending", Float)
    Ethereum = Column("Ethereum", Float)
    Polygon = Column("Polygon", Float)
    Avalanche = Column("Avalanche", Float)
    Binance = Column("Binance", Float)
    factor_log_mc = Column("factor_log_mc", Float)
    factor_log_mc_tvl = Column("factor_log_mc_tvl", Float)
    VMR = Column("VMR", Float)
    factor_momentum = Column("factor_momentum", Float)

class ResTest(Base):
    __tablename__ = "res_test-1"

    index = Column("index", DateTime, primary_key=True)
    R_2 = Column("R_2", Float)
    R_2_adj = Column("R_2_adj", Float)
    F_test_regression = Column("F_test_regression", Float)
    F_test_chains = Column("F_test_chains", Float)
    F_test_categories = Column("F_test_categories", Float)
    F_test_styles = Column("F_test_styles", Float)
    residual_std = Column("residual_std", Float)
    
class SE(Base):
    __tablename__ = "se-1"

    index = Column("index", DateTime, primary_key=True)
    const = Column("const", Float)
    Derivatives = Column("Derivatives", Float)
    Dexes = Column("Dexes", Float)
    Yield = Column("Yield", Float)
    Bridge = Column("Bridge", Float)
    Lending = Column("Lending", Float)
    Ethereum = Column("Ethereum", Float)
    Polygon = Column("Polygon", Float)
    Avalanche = Column("Avalanche", Float)
    Binance = Column("Binance", Float)
    factor_log_mc = Column("factor_log_mc", Float)
    factor_log_mc_tvl = Column("factor_log_mc_tvl", Float)
    VMR = Column("VMR", Float)
    factor_momentum = Column("factor_momentum", Float)

class Finance_data(Base):
    __tablename__ = "test_data_3"

    index = Column("index", Integer, primary_key = True)
    date = Column("date", String)
    symbol = Column("symbol", String)
    gecko_id = Column("gecko_id", String)
    llama_id = Column("llama_id", String)
    category = Column("category", String)
    chain = Column("chain", String)
    address = Column("address", String)
    price = Column("price", Float)
    market_cap = Column("market_cap", Float)
    volume = Column("volume", Float)
    TVL = Column("TVL", Float)
    id_collection = Column("id_collection", String)
    update_date = Column("update_date", String)
    
class VifValues(Base):
    __tablename__ = "vif_values-1"

    index = Column("index", DateTime, primary_key=True)
    const = Column("const", Float)
    Derivatives = Column("Derivatives", Float)
    Dexes = Column("Dexes", Float)
    Yield = Column("Yield", Float)
    Bridge = Column("Bridge", Float)
    Lending = Column("Lending", Float)
    Ethereum = Column("Ethereum", Float)
    Polygon = Column("Polygon", Float)
    Avalanche = Column("Avalanche", Float)
    Binance = Column("Binance", Float)
    factor_log_mc = Column("factor_log_mc", Float)
    factor_log_mc_tvl = Column("factor_log_mc_tvl", Float)
    VMR = Column("VMR", Float)
    factor_momentum = Column("factor_momentum", Float)
    
class XDict(Base):
    __tablename__ = "x_dict-1"

    gecko_id = Column("gecko_id", String, primary_key=True)
    const = Column("const", Float)
    Derivatives = Column("Derivatives", Float)
    Dexes = Column("Dexes", Float)
    Yield = Column("Yield", Float)
    Bridge = Column("Bridge", Float)
    Lending = Column("Lending", Float)
    Ethereum = Column("Ethereum", Float)
    Polygon = Column("Polygon", Float)
    Avalanche = Column("Avalanche", Float)
    Binance = Column("Binance", Float)
    factor_log_mc = Column("factor_log_mc", Float)
    factor_log_mc_tvl = Column("factor_log_mc_tvl", Float)
    VMR = Column("VMR", Float)
    factor_momentum = Column("factor_momentum", Float)
    other_chains = Column("other_chains", Float)
    other_categories = Column("other_categories", Float)
    index = Column("date", DateTime)

class Shares(Base):
    __tablename__ = "share_perf_frame"

    index = Column("index", DateTime, primary_key=True)
    share_dollar_value = Column("1 share dollar value", Float)
    
class Returns(Base):
    __tablename__ = "ret_copy-1"

    index = Column("index", DateTime, primary_key=True)
    pancakeswap_token = Column("pancakeswap-token", Float)
    uniswap = Column("uniswap", Float)
    woo_network = Column("woo-network", Float)
