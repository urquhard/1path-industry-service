import asyncio
import requests
from sqlalchemy import create_engine, Metadata, Table, Session
from datetime import datetime, timedelta


from defi.worker import get_full_data
from settings.env import env


engine = create_engine(f"postgresql://{env.username}:{env.password}@{env.host}:5432/{env.database}")

# add timedelta depending on env.delay
# add index validation
# TODO add try except validation for each action
async def update_scheduler() -> None:
    """
    Used for scheduling updates of TVL in postgres
    Time for update is located in .env file

    Returns
    -------
    None.

    """
    while True:
        await asyncio.sleep(env.finance_data_delay)

        today_date = datetime.today().strftime('%Y-%m-%d')
        previous_date = datetime.today() - timedelta(days = env.finance_data_delay / 86400)
        new_data = get_full_data(start_date = previous_date,
                                 end_date = today_date)
        new_data.to_sql('test_data', engine, if_exists='append')
        
        # Adding task data
        res = requests.get("https://celery-api/api/model/update/").json()
        res["time"] = datetime.today().strftime('%Y-%m-%d-%H-%M')
        res["task_info"] = "Update weights"
        metadata = Metadata(bind=engine)
        task_table = Table('tasks', metadata, autoload = True)
        query = task_table.insert()
        query.values(**res)
    
        my_session = Session(engine)
        my_session.execute(query)
        my_session.close()
        


async def main():
    """
    Main asyncio function. Initializes TVL and sets it into postgres,
    then updating TVL. 

    Returns
    -------
    None.

    """
    if "True" in env.initialize:
        today_date = datetime.today().strftime('%Y-%m-%d')
        df = get_full_data(start_date = '2021-08-31',
                           end_date = today_date)
        df.to_sql('test_data', engine, if_exists='replace')
        
        res = requests.get("https://celery-api/api/model/").json()
        res["time"] = datetime.today().strftime('%Y-%m-%d-%H-%M')
        res["task_info"] = "Init weights"
        metadata = Metadata(bind=engine)
        task_table = Table('tasks', metadata, autoload = True)
        query = task_table.insert()
        query.values(**res)
    
        my_session = Session(engine)
        my_session.execute(query)
        my_session.close()
    
    while True:
        await asyncio.gather(
            update_scheduler()
        )

if __name__ == "__main__":
    asyncio.run(main())

