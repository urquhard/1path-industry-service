import asyncio
from sqlalchemy import create_engine
from datetime import datetime
import pandas as pd


from defi.worker import get_full_data
from settings.env import env
from defi.data import data
from defi.model import factor_model

engine = create_engine(f"postgresql://{env.username}:{env.password}@{env.host}:5432/{env.database}")


def update_data():
    today_date = datetime.today().strftime('%Y-%m-%d')
    df = get_full_data(start_date = '2021-08-31',
                       end_date = today_date)
    df.to_sql('test_data', engine, if_exists='replace')

def update_weights():
    df = pd.read_sql_table("test_data", engine)
    
    df = data.data_preparation(df)
    ret_copy, price_copy, mar_cap = data.filter_data(df)
    Beta, categories_columns, blockchain_columns, style_factors = data.beta_preparation(df, ret_copy, price_copy, mar_cap)
    industry_returns, x_frame, fitted_frame, resid_frame, p_value_data, res_test, vif_values, se, geckoId_to_symbol = factor_model(Beta, categories_columns, blockchain_columns, style_factors, "power", 0.25)
    res = []
    res.append(Beta.to_sql("beta", engine, if_exists='replace'))
    res.append(industry_returns.to_sql("industry_returns", engine, if_exists='replace'))
    res.append(x_frame.to_sql("x_dict", engine, if_exists='replace'))
    res.append(resid_frame.to_sql("resid_frame", engine, if_exists='replace'))
    res.append(p_value_data.to_sql("p_value_data", engine, if_exists='replace'))
    res.append(res_test.to_sql("res_test", engine, if_exists='replace'))
    res.append(vif_values.to_sql("vif_values", engine, if_exists='replace'))
    res.append(se.to_sql("se", engine, if_exists='replace'))
    res.append(fitted_frame.to_sql("fitted_frame", engine, if_exists='replace'))
    res.append(geckoId_to_symbol.to_sql("geckoId_to_symbol", engine, if_exists='replace'))
    res.append(ret_copy.to_sql("ret_copy", engine, if_exists='replace'))

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
        try:
            update_data()
        except Exception as e:
            print(e)
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
            update_weights_scheduler()
        )

if __name__ == "__main__":
    asyncio.run(main())

