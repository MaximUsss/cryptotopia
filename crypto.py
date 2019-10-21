import requests
import pandas as pd
# import matplotlib.pyplot as plt


# These are the coins that we will use
coins = {"BTC": "Bitcoin", "ETH": "Etherium", "XRP": "Ripple"}


class CCurrency(object):
    """
    Main cryptocurrency class. Includes currencies that we use
        and methods fot getiing data from cryptocompare API
    """

    def __init__(self):
        self.coins = coins

    def get_data(self, date, simbol):
        """ Query the API for 2000 days historical price data starting from "date". """
        url = f"https://min-api.cryptocompare.com/data/histoday?fsym={simbol}&tsym=USD&limit=2000&toTs={date}"
        r = requests.get(url)
        ipdata = r.json()
        return ipdata

    def get_df(self, from_date, to_date, simbol):
        """ Get historical price data between two dates. """
        date = to_date
        holder = []
        # While the earliest date returned is later than the earliest date requested, keep on querying the API
        # and adding the results to a list.
        while date > from_date:
            data = self.get_data(date, simbol)
            holder.append(pd.DataFrame(data['Data']))
            date = data['TimeFrom']
        # Join together all of the API queries in the list.
        df = pd.concat(holder, axis=0)
        # Remove data points from before from_date
        df = df[df['time'] > from_date]
        # Convert to timestamp to readable date format
        df['time'] = pd.to_datetime(df['time'], unit='s')
        # Make the DataFrame index the time
        df.set_index('time', inplace=False)
        # And sort it so its in time order
        df.sort_index(ascending=False, inplace=True)
        return df


'''
curr = CCurrency()
df = curr.get_df(1504435200, 1534435200, "BTC")

fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(df[['low', 'close', 'high']])
ax.set_ylabel('BTC Price (USD)')
ax.set_xlabel('Date')
ax.legend(['Low', 'Close', 'High'])
plt.show()
'''
