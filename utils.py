import pandas as pd

def change_wg_to_diff(df):


	# make a difference column that doesn't have negatives
	df["net_energy_use"] = df["TOTAL_ENERGY_DESK_TODAY"].diff()
	df.net_energy_use.loc[df.net_energy_use<0] = 0

	# groupby dates
	df["TimeStamp"] = pd.to_datetime(df.TimeStamp)
	temp_df = df
	temp_df["month"] = [str(t.month) if t.month>9 else \
	    "0" + str(t.month) for t in df["TimeStamp"]]
	temp_df["year"] = [str(t.year) for t in df["TimeStamp"]]
	temp_df["hour"] = [str(t.hour) if t.hour>9 else \
	    "0" + str(t.hour) for t in df["TimeStamp"]]
	temp_df["day"] = [str(t.day) if t.day>9 else \
	    "0" + str(t.day) for t in df["TimeStamp"]]
	df["date_hour"] =  (temp_df[['year', 'month',"day", "hour"]]
	                                  .apply(lambda x: ''.join(x), axis=1))
	b_grouped = df.groupby("date_hour")
	print(b_grouped)

	final = b_grouped.agg({"Id":"first",
	              "TimeStamp": "first",
	              "TOTAL_ENERGY_DESK_TODAY":"first",
	              "net_energy_use":"sum"})
	return final
