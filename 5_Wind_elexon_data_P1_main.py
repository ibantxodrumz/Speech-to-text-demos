## Dependencies (Python + R)

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from datetime import datetime, timedelta
from pathlib import Path
import cx_Oracle  # type: ignore
import rpy2.robjects as ro  # type: ignore
from rpy2.robjects import pandas2ri  # type: ignore
from rpy2.robjects.packages import importr  # type: ignore
import rpy2.robjects.packages as rpackages  # type: ignore
import rpy2.robjects.pandas2ri as pandas2ri  # type: ignore
import json
import warnings
import xmltodict  # type: ignore
import http.client
import urllib.parse

warnings.filterwarnings("ignore")

# Activate the automatic conversion between R and pandas
pandas2ri.activate()


##--------------------- Define Dates and Directories------------------------##

Year = 2024
clreg = "G"

# Dates for actuals
sd1 = 20241027
ed1 = int(datetime.today().strftime("%Y%m%d")) - 2

# Start date of forecasts
sd3 = datetime.today().strftime("%Y-%m-%d")

# Directories
incentive_forecast_dir = Path(
    "S:/OandT/OptRisk/Energy_Requirements/09 - Demand forecasting (Forecasting team)/00 - Audit documents/Wind Incentive/Data"
)
s_dir = Path("//uk.corporg.net/ngtdfs$/GROUP")
energy_dir = s_dir / "OandT/OptRisk/Energy_Requirements"
op_market_dir = energy_dir / "29_-_Operational_Market_Information"
MI_dir = op_market_dir / "Fundamentals & Insight"
margin_risk_dir = MI_dir / "Margin_risk"
simulation_dir = margin_risk_dir / "simulation_output"
conn_dir = Path(
    "//uk.corporg.net/ngtdfs$/GROUP/OandT/OptRisk/Energy_Requirements/29_-_Operational_Market_Information/Running files"
)
dft_dir = (
    energy_dir
    / "09 - Demand forecasting (Forecasting team)/10 - Daily Files/Demand Forecasting System"
)
dft_hh_dir = dft_dir / "Output/Forecasts/Recent/HH/"
winter24_dir = margin_risk_dir / "Winter_24_live"
Oct_update_sim_dir = simulation_dir / "2024_G/WO_final"

# Define base URL--- this will retrieve data for df2 from Elexon
base_url = "/bmrs/api/v1/generation/outturn/summary?"

# -------------------------------------- Functions-------------------------------------##


def read_rds(file_path):
    """
    Reads an RDS file from the specified file path using R's readRDS function and returns the result.
    """
    readRDS = ro.r["readRDS"]
    return readRDS(str(file_path))  # type: ignore


def fetch_data_for_chunk(start_date, end_date):
    """
    Fetches wind generation data from the Elexon API for the specified date range
    and returns it as a list of dictionaries.
    """
    # Ensure end_date includes the full day
    end_date = end_date + timedelta(seconds=86399)  # Equivalent to 23:59:59

    params = {
        "startTime": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "endTime": end_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "includeNegativeGeneration": "true",
        "format": "xml",
    }
    url = base_url + urllib.parse.urlencode(params)

    conn = http.client.HTTPSConnection("data.elexon.co.uk")
    conn.request("GET", url)
    response = conn.getresponse()

    rows = []
    if response.status == 200:
        xml_data = response.read().decode("utf-8")

        try:
            data_dict = xmltodict.parse(xml_data)
            if "ArrayOfOutturnGenerationBySettlementPeriod" in data_dict:
                outturn_list = data_dict[
                    "ArrayOfOutturnGenerationBySettlementPeriod"
                ].get("OutturnGenerationBySettlementPeriod", [])

                if isinstance(outturn_list, dict):
                    outturn_list = [outturn_list]

                for outturn_generation in outturn_list:
                    date_str = outturn_generation["StartTime"][:10]
                    settlement_period = int(outturn_generation["SettlementPeriod"])
                    data = outturn_generation["Data"]["OutturnGenerationValue"]

                    row = {
                        "Date": date_str,
                        "SettlementPeriod": settlement_period,
                        "WIND": 0,
                    }

                    if data:
                        if isinstance(data, list):
                            for generation in data:
                                fuel_type = generation["FuelType"]
                                if fuel_type == "WIND":
                                    gen_value = float(generation["Generation"])
                                    row["WIND"] = gen_value
                        elif isinstance(data, dict):
                            fuel_type = data["FuelType"]
                            if fuel_type == "WIND":
                                gen_value = float(data["Generation"])
                                row["WIND"] = gen_value
                    rows.append(row)
            else:
                print(f"No data available for the period {start_date} to {end_date}")
        except Exception as e:
            print(
                f"Error: Failed to process XML for period {start_date} to {end_date}. {e}"
            )
    else:
        print(f"Error: Failed to retrieve data. Status code: {response.status}")

    conn.close()
    return rows


def generate_date_chunks(start_date, end_date, chunk_size):
    """
    Generates a list of date ranges (chunks) between the given start and end dates,
    with each chunk having the specified number of days.
    """
    chunks = []
    current_start = start_date

    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=chunk_size - 1), end_date)
        chunks.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)

    # Ensure the final chunk ends on the end_date
    if chunks[-1][1] != end_date:
        chunks[-1] = (chunks[-1][0], end_date)

    # Print chunks for debugging
    for chunk in chunks:
        print(f"Chunk: Start: {chunk[0]}, End: {chunk[1]}")

    return chunks


def download_generation_data(start_date, end_date):
    """
    Downloads wind generation data from the Elexon API for the specified date range in 10-day chunks,
    checks for missing dates, and returns the data as a DataFrame.
    """
    all_data = []
    chunk_size = 10

    # Generate date ranges in chunks of 10 days
    chunks = generate_date_chunks(start_date, end_date, chunk_size)

    for chunk_start_date, chunk_end_date in chunks:
        print(f"Fetching data for: Start: {chunk_start_date}, End: {chunk_end_date}")
        chunk_data = fetch_data_for_chunk(chunk_start_date, chunk_end_date)
        all_data.extend(chunk_data)

    df = pd.DataFrame(all_data)

    if df.empty:
        print("No data retrieved for the specified date range.")
        return df

    # Check for missing dates
    all_dates = pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d")
    existing_dates = set(df["Date"])
    missing_dates = set(all_dates) - existing_dates

    # Fetch missing dates data
    if missing_dates:
        for missing_date in missing_dates:
            missing_start_date = pd.Timestamp(missing_date)
            chunk_data = fetch_data_for_chunk(missing_start_date, missing_start_date)
            all_data.extend(chunk_data)

    df = pd.DataFrame(all_data)
    existing_dates = set(df["Date"])
    missing_dates = set(all_dates) - existing_dates

    if missing_dates:
        print(f"Warning: Missing data for the following dates: {sorted(missing_dates)}")

    df.sort_values(by=["Date", "SettlementPeriod"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def merge_and_process_data(df1, df2):
    """
    Merges two DataFrames on the "Date" column, filters by matching "SettlementPeriod",
    and returns a processed DataFrame with selected columns.
    """
    df2["Date"] = pd.to_datetime(df2["Date"])
    elexon = pd.merge(df1, df2, on="Date", how="inner")
    elexon["SettlementPeriod"] = elexon["SettlementPeriod"].astype(int)
    elexon1 = elexon[elexon["SETTLEMENT_PERIOD"] == elexon["SettlementPeriod"]]
    elexon1.reset_index(drop=True, inplace=True)

    elexon2 = elexon1[["Date", "SettlementPeriod", "WIND"]]

    return elexon2


def save_to_csv(df, file_path):
    """
    Saves the DataFrame to a CSV file at the specified file path and prints the resolved file path.
    """
    df.to_csv(file_path, index=False)
    print("CSV file path:", file_path.resolve())


def read_credentials_from_file(file_path):
    """
    Reads and returns credentials from a JSON file at the specified file path.
    Returns None if an error occurs.
    """
    try:
        with open(file_path, "r") as f:
            credentials = json.load(f)
        return credentials
    except Exception as e:
        print(f"Error reading credentials from file: {e}")
        return None


def connect_to_database(credentials):
    """
    Connects to an Oracle database using the provided credentials and returns the connection object.
    Returns None if a connection error occurs.
    """
    try:
        dsn_tns = cx_Oracle.makedsn(
            host=credentials["host"],
            port=credentials["port"],
            service_name=credentials["service_name"],
        )
        conn = cx_Oracle.connect(
            user=credentials["user"], password=credentials["password"], dsn=dsn_tns
        )
        return conn
    except cx_Oracle.DatabaseError as e:
        print(f"Database connection error: {e}")
        return None


def execute_sql_query(conn, sql_query, params=None):
    """
    Executes an SQL query using the provided connection and optional parameters,
    returning the result as a DataFrame. Returns None if an error occurs.
    """
    try:
        with conn.cursor() as curs:
            if params is None:
                curs.execute(sql_query)
            else:
                curs.execute(sql_query, params)
            data_extract = pd.DataFrame(
                curs.fetchall(), columns=[x[0] for x in curs.description]
            )
        return data_extract
    except cx_Oracle.DatabaseError as e:
        print(f"SQL query execution error: {e}")
        return None


def standardize_date_column(df):
    """
    Standardizes various date columns in the DataFrame to a single "Date" column in datetime format,
    dropping original date columns.
    """
    df = df.copy()
    date_column_formats = {
        "CDATE": "%Y%m%d",
        "GDATE": "%Y%m%d",
        "SETTLEMENT_DATE": "%d-%b-%Y",
        "Datetime": "%Y-%m-%d %H:%M:%S",
        "Date": None,
    }
    date_columns_present = [col for col in date_column_formats if col in df.columns]
    if not date_columns_present:
        print("No recognized date column found in the dataframe.")
        return df
    for date_column in date_columns_present:
        format = date_column_formats[date_column]
        if df[date_column].dtype == "int64" or df[date_column].dtype == "object":
            df[date_column] = df[date_column].astype(str)
            if format:
                df[date_column] = pd.to_datetime(
                    df[date_column], format=format, errors="coerce"
                )
            else:
                df[date_column] = pd.to_datetime(
                    df[date_column], errors="coerce", dayfirst=True
                )
        else:
            df[date_column] = pd.to_datetime(
                df[date_column], errors="coerce", dayfirst=True
            )
        print(f"Converted column '{date_column}' to datetime.")
    df = df.dropna(subset=date_columns_present, how="all")
    if "Date" not in date_columns_present:
        df["Date"] = df[date_columns_present].bfill(axis=1).iloc[:, 0]
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    columns_to_drop = [col for col in date_columns_present if col != "Date"]
    df = df.drop(columns=columns_to_drop)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.reset_index(drop=True)
    return df


##----------------------------------MAIN----------------------------------------------##


def main():
    """
    Main function to read RDS files, process wind data, query a database for demand data,
    standardize date columns, and save the processed data to a CSV file.
    """
    # Read RDS files
    a1 = read_rds(Oct_update_sim_dir / "demand_weather_1_1.RDS")
    a2 = read_rds(Oct_update_sim_dir / "demand_weather_1_2.RDS")
    a3 = read_rds(Oct_update_sim_dir / "demand_weather_1_3.RDS")

    wind1 = np.array(a1.rx2("bmu_wind_m"))
    wind2 = np.array(a2.rx2("bmu_wind_m"))
    wind3 = np.array(a3.rx2("bmu_wind_m"))

    wind1_df = pd.DataFrame(wind1)
    wind2_df = pd.DataFrame(wind2)
    wind3_df = pd.DataFrame(wind3)

    # create dataframe out with quantiles

    Wind = pd.concat([wind1_df, wind2_df, wind3_df], axis=1)

    Out = Wind.quantile([0.05, 0.5, 0.95], axis=1).transpose()
    Out.columns = ["p5", "p50", "p95"]

    start_date = datetime(2024, 10, 28)
    date_range = pd.date_range(start_date, periods=len(Out), freq="D")

    Out["Date"] = date_range
    Out = Out.reset_index(drop=True)

    print(Out.head())
    print(Out.tail())
    print(Out.info())

    # access database

    credentials_file = conn_dir / "efs_conn1_python.txt"
    credentials = read_credentials_from_file(credentials_file)

    # Check if Oracle client is already initialized
    try:
        cx_Oracle.init_oracle_client(
            lib_dir=r"C:\Users\ivan.sanz\Downloads\instantclient_21_7",
            config_dir=r"C:\TNSNames",
        )
    except cx_Oracle.ProgrammingError as e:
        print(f"Oracle client initialization error: {e}")

    connection = connect_to_database(credentials)

    qdemand = """
    select v.cdate, v.ctime, 
    to_char(to_date(v.cdate,'YYYYMMDD'),'DD-MON-YYYY') as settlement_date,
    rank() over (partition by v.cdate order by v.gdate, v.gtime) as settlement_period,
    sum(case when arnum = 99 then metdem else 0 end) as nd
    from vdemand v
    where v.cdate between :start_date and :end_date
    group by v.gdate, v.gtime, v.cdate, v.ctime
    order by v.cdate, v.ctime
    """
    params = {"start_date": sd1, "end_date": ed1}
    update_hh = execute_sql_query(connection, qdemand, params)

    if update_hh is not None:
        print("Query executed successfully.")
        update_hh["CDATE"] = update_hh["CDATE"].astype(str).str.strip()
        update_hh["CTIME"] = update_hh["CTIME"].astype(str).str.strip()

        mask_2400 = update_hh["CTIME"] == "2400"
        update_hh.loc[mask_2400, "CTIME"] = "0000"
        update_hh.loc[mask_2400, "CDATE"] = (
            pd.to_datetime(update_hh.loc[mask_2400, "CDATE"], format="%Y%m%d")
            + timedelta(days=1)
        ).dt.strftime("%Y%m%d")

        update_hh["CTIME"] = update_hh["CTIME"].apply(lambda x: x.zfill(4))
        combined_datetime = update_hh["CDATE"] + update_hh["CTIME"]
        update_hh["Datetime"] = pd.to_datetime(
            combined_datetime, format="%Y%m%d%H%M", errors="coerce"
        )

        if update_hh["Datetime"].isnull().any():
            print(
                "Some datetimes could not be parsed. Check the following rows for inconsistencies:"
            )
            problematic_rows = update_hh.loc[
                update_hh["Datetime"].isnull(), ["CDATE", "CTIME", "Datetime"]
            ]
            print(problematic_rows)
            raise ValueError(
                "Some datetimes could not be parsed. Check the CDATE and CTIME fields for inconsistencies."
            )

        peak_day = update_hh.loc[update_hh.groupby("CDATE")["ND"].idxmax()].copy()
        peak_day.reset_index(drop=True, inplace=True)

        print("Peak demand data processed successfully.")
    else:
        print("Query execution failed.")

    peak_day = standardize_date_column(peak_day)
    print("Processing df1")
    df1 = standardize_date_column(peak_day)

    # Fixed start date
    start_date = datetime.strptime("2024-10-28", "%Y-%m-%d")

    # Dynamic end date (today - 2 days)
    end_date = datetime.today() - timedelta(days=2)

    df2 = download_generation_data(start_date, end_date)

    elexon2 = merge_and_process_data(df1, df2)

    csv_file = winter24_dir / "Elexon_data_python.csv"
    save_to_csv(elexon2, csv_file)


if __name__ == "__main__":
    main()
