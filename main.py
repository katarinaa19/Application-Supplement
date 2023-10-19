# Import Libraries
import pandas as pd
import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Transform raw data into more readily used formats
def process_assets(innercodes):
    """
    Process asset data based on a list of inner codes and create merged DataFrames
    Return DataFrams ready for constructing the Risk Parity Model and Mean-Variance Model

    Parameters:
    innercodes (list): A list of inner codes for assets to be processed.

    Returns:
    data_MV (DataFrame): DataFrams ready for constructing the Mean-Variance Model
    data_RP (DataFrame): DataFrams ready for constructing the Risk Parity Model
    """
    # Change the current working directory to the specified folder and import files
    folder_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(folder_path)
    # Ingest data from csv
    secumain_df = pd.read_csv("secumain.csv", encoding="gbk")
    qt_commindexquote_df = pd.read_csv("qt_commindexquote.csv")
    qt_indexquote_df = pd.read_csv("qt_indexquote.csv")
    bond_bondindexquote_df = pd.read_csv("bond_bondindexquote.csv")
    # Process dataframes
    asset_df = pd.concat(
        [qt_commindexquote_df, qt_indexquote_df, bond_bondindexquote_df], axis=0
    )
    data_frames = []
    for i, inner_code in enumerate(innercodes, 1):
        # Create a data frame for the asset
        asset_index = (
            secumain_df.merge(asset_df, on="InnerCode", how="inner")
            .query(f'TradingDay >= "2010-01-01" and InnerCode == {inner_code}')
            .rename(
                columns={
                    "ChiName": f"ASSET{i}_Name",
                    "InnerCode": f"ASSET{i}_Code",
                    "TradingDay": "DATE",
                    "ClosePrice": f"ASSET{i}_ClosePrice",
                }
            )
        )
        asset_index[f"ASSET{i}_Return"] = asset_index[
            f"ASSET{i}_ClosePrice"
        ].pct_change()
        data_frames.append(asset_index)
    merged_df = data_frames[0]
    for df in data_frames[1:]:
        merged_df = pd.merge(merged_df, df, on="DATE", how="inner")
    data_MV = merged_df
    data_RP = merged_df
    return data_MV, data_RP


# Examine assets
def Calculate_Evaluation_Indicators(df, rf=0.00, ALL_Indicator=False, net_value_str=""):
    """
    Calculate various evaluation indicators for a DataFrame of financial data.
    Parameters:
        df (DataFrame): The input DataFrame containing financial data.
        rf (float, optional): The risk-free rate, default is 0.00.
        ALL_Indicator (bool, optional): Whether to calculate all indicators (True) or not (False), default is False.
        net_value_str (str, optional): The column name in the DataFrame representing net values, default is ''.

    Returns:
        tuple or float: Depending on the value of ALL_Indicator:
    """
    df = df.set_index("DATE")
    df_local = df.copy()
    print(df_local[net_value_str])
    rr = round(
        (
            (df_local[net_value_str].iloc[-1] / df_local[net_value_str].iloc[0])
            ** (
                365.0
                / (df_local[net_value_str].index[-1] - df[net_value_str].index[0]).days
            )
            - 1.0
        ),
        8,
    )
    std = round((df_local[net_value_str].pct_change().std() * np.sqrt(252.0)), 8)
    if std != 0:
        sharpe = round((rr - rf) / std, 8)
    else:
        sharpe = 0
    maxdrawdown = round(
        (
            1.0 - (df_local[net_value_str] / df_local[net_value_str].expanding().max())
        ).max(),
        8,
    )
    if not ALL_Indicator:
        return rr, std, maxdrawdown, sharpe
    else:
        d_std = df_local[net_value_str].pct_change().dropna().apply(
            lambda x: min(x, 0)
        ).std() * np.sqrt(252.0)
        if d_std != 0:
            sortino = round((rr - rf) / d_std, 8)
        else:
            sortino = 0
        if maxdrawdown != 0:
            calmar = round(rr / maxdrawdown, 8)
        else:
            calmar = 0
        return rr, std, maxdrawdown, sharpe, sortino, calmar


# Data Anlaysis
def R_P(subset, T, rf):
    """
    Calculate the weights with Risk Parity Model

    Parameters:
    subset (DataFrame): Subset of data containing asset returns.
    T (int): Number of periods.
    rf (float): Risk-free rate.

    Returns:
    tuple: A tuple containing the optimal portfolio weights (w_op), expected return (r_op), and portfolio volatility (sigma_op).
    """
    R = subset.apply(pd.to_numeric, errors="coerce")
    R_mean = R.mean(axis=0)
    Sigma = R.cov()
    N = len(R_mean)
    R_mean = np.array(R_mean).reshape(N)
    Sigma = np.array(Sigma).reshape(N, N)

    eps = 1e-10
    w0 = np.ones(N) / N

    def fun(w):
        return np.dot(
            w - np.dot(np.dot(w, Sigma), np.transpose(w)) / N / np.dot(w, Sigma),
            np.transpose(
                w - np.dot(np.dot(w, Sigma), np.transpose(w)) / N / np.dot(w, Sigma)
            ),
        )

    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # 限制条件一：全部投资
        {"type": "ineq", "fun": lambda w: w - eps},  # 限制条件二：不可做空
    )
    res = minimize(fun, w0, method="SLSQP", constraints=cons)

    w_op = res.x
    r_op = np.dot(res.x, R_mean)
    sigma_op = np.dot(np.dot(w_op, Sigma), np.transpose(w_op)) / T
    return w_op, r_op, sigma_op


def M_V(subset):
    """
    Calculate the weight with Mean-Variance Model

    Parameters:
    subset (DataFrame): Subset of data containing asset returns.

    Returns:
    tuple: A tuple containing the optimal portfolio weights (w_op) and expected return (r_op).
    """
    R = subset.apply(pd.to_numeric, errors="coerce")
    R_mean = R.mean(axis=0)
    R_cov = R.cov()
    N = len(R_mean)
    R_mean = np.array(R_mean).reshape(N)
    R_cov = np.array(R_cov).reshape(N, N)

    def f(w):
        w = np.array(w)
        Rp_opt = np.sum(w * R_mean)
        Vp_opt = np.sqrt(np.dot(w, R_cov @ w.T))
        rf = ((1 + 0.0264) ** (0.25)) - 1
        sharpe_r = np.dot(w, R_mean.T - rf) / Vp_opt
        return np.array([Rp_opt, -sharpe_r])

    def Vmin_f(w):
        return f(w)[1]

    cons = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        {"type": "eq", "fun": lambda x: f(x)[0] - 0},
    )
    bnds = tuple((0, 1) for _ in range(len(R_mean)))
    result = minimize(
        Vmin_f,
        len(R_mean)
        * [
            1.0 / len(R_mean),
        ],
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
    )
    w_op = result.x
    r_op = np.dot(result.x, R_mean)
    return w_op, r_op


def apply_strategy(data, target_column_names, T, strategy_type="MV"):
    """
    This function applies a portfolio investment strategy to the given Asset data using a rolling window approach.
    It calculates portfolio weights for each rolling window and tracks portfolio returns and net asset values.
    The resulting portfolio weights, returns, and net asset values are added as new columns to the input DataFrame.

    Parameters:
    data (DataFrame): The input DataFrame containing Asset data.
    target_column_names (list): List of column names representing asset returns.
    T (int): Rolling window size.
    target (float): Target return for the strategy.
    strategy_type (str, optional): Type of investment strategy ('MV' for Mean-Variance or 'RP' for Risk Parity). Default is 'MV'(Mean-Variance).
    """
    dates = data["DATE"]
    date_length = len(dates)

    # Initialize net asset value
    data.loc[0:T, "PORT_NAV"] = 1

    for i in range(T, date_length, T):
        # Calculate weights for the rolling window
        if i - 250 <= 0:
            if strategy_type == "MV":
                weights = M_V(data.iloc[0:i][target_column_names])[0]
            elif strategy_type == "RP":
                weights = R_P(data.iloc[0:i][target_column_names], T, 0.02)[0]
        else:
            if strategy_type == "MV":
                weights = M_V(data.iloc[i - 250 : i][target_column_names])[0]
            elif strategy_type == "RP":
                weights = R_P(data.iloc[i - 250 : i][target_column_names], T, 0.02)[0]

        ticker_no = 0
        # Append weight
        for column in target_column_names:
            data.loc[i : i + T, column + "_WEIGHT"] = weights[ticker_no]
            ticker_no += 1

        # Calculate portfolio return
        for j in data.index[i : i + T]:
            Portfolio_return = 0
            for column in target_column_names:
                Portfolio_return += float(data.loc[j, column + "_WEIGHT"]) * float(
                    data.loc[j, column]
                )
            data.loc[j, "PORT_RETURN"] = Portfolio_return
            data.loc[j, "PORT_NAV"] = float(data.loc[j - 1, "PORT_NAV"]) * (
                1 + float(data.loc[j, "PORT_RETURN"])
            )


def calculate_cumulative_return(data, target_column_names):
    """
    This function calculates cumulative returns for the specified columns in the input DataFrame.
    It adds new columns with names like 'COLUMN_NAME_CUMRETURN' to the DataFrame, where 'COLUMN_NAME' is
    the name of the original column.


    Parameters:
    data (DataFrame): The input DataFrame containing Asset data.
    target_column_names (list): List of column names for which cumulative returns should be calculated.

    Returns:
    None

    """
    for column in target_column_names:
        data[column + "_CUMRETURN"] = (1 + data[column]).cumprod() - 1


def draw_return_rate(data, target_column_names, file_name):
    """
    This function generates a plot showing cumulative returns (PORT_RETURN_CUMRETURN) and individual returns
    for the specified columns (target_column_names) over time.

    Parameters:
        data (DataFrame): The input DataFrame containing financial data.
        target_column_names (list): List of column names representing returns.
        file_name (str): The name of the output plot file.

    """
    x = data["DATE"]
    y = data["PORT_RETURN_CUMRETURN"]

    plt.figure(figsize=(12, 7))
    plt.rcParams["font.family"] = ["Times New Roman"]
    plt.scatter(x, y, c="darkseagreen", s=4)

    colors = [
        "brown",
        "peru",
        "lightsteelblue",
        "olive",
        "slategray",
        "dimgrey",
        "darkkhaki",
        "cadetblue",
    ]
    color_index = 0
    for column in target_column_names:
        current_color = colors[color_index]
        plt.plot(
            x,
            data[column + "_CUMRETURN"],
            label=column + "_Cumulative",
            lw=2,
            color=current_color,
        )
        color_index += 1
        if color_index == len(colors):
            color_index = 0
    plt.xlabel("DATE")
    plt.ylabel("Cumulative_Return")
    plt.legend(loc="upper left")
    plt.title(file_name, fontsize=14)
    plt.savefig(file_name, bbox_inches="tight")
    plt.show()


def draw_weights(data, target_column_names, file_name):
    """
    This function generates a plot showing how portfolio weights change over time for the specified columns
    (target_column_names).

    Parameters:
        data (DataFrame): The input DataFrame containing financial data.
        target_column_names (list): List of column names representing portfolio weights.
        file_name (str): The name of the output plot file.

    """
    stacked_data = data[["DATE"] + target_column_names]
    stacked_data.set_index("DATE", inplace=True)
    normalized_data = stacked_data.div(stacked_data.sum(axis=1), axis=0)
    plt.figure(figsize=(12, 7))
    plt.rcParams["font.family"] = ["Times New Roman"]
    plt.stackplot(
        normalized_data.index,
        normalized_data.values.T,
        labels=normalized_data.columns,
        colors=["brown", "peru", "cadetblue", "olive", "slategray", "tan", "darkkhaki"],
    )
    plt.xlabel("DATE")
    plt.ylabel("Weights")
    plt.legend(loc="upper left")
    plt.title(file_name, fontsize=14)
    plt.savefig(file_name, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Define Variable
    innercodes = [203163, 6455, 7551, 4982, 15410]
    data_MV, data_RP = process_assets(innercodes)
    T = 90

    # Risk Parity Model
    apply_strategy(
        data_RP,
        [
            "ASSET1_Return",
            "ASSET2_Return",
            "ASSET3_Return",
            "ASSET4_Return",
            "ASSET5_Return",
        ],
        T,
        strategy_type="RP",
    )

    calculate_cumulative_return(
        data_RP,
        [
            "PORT_RETURN",
            "ASSET1_Return",
            "ASSET2_Return",
            "ASSET3_Return",
            "ASSET4_Return",
            "ASSET5_Return",
        ],
    )

    draw_return_rate(
        data_RP,
        [
            "PORT_RETURN",
            "ASSET1_Return",
            "ASSET2_Return",
            "ASSET3_Return",
            "ASSET4_Return",
            "ASSET5_Return",
        ],
        "Risk Parity Model: Asset Return",
    )

    draw_weights(
        data_RP,
        [
            "ASSET1_Return_WEIGHT",
            "ASSET2_Return_WEIGHT",
            "ASSET3_Return_WEIGHT",
            "ASSET4_Return_WEIGHT",
            "ASSET5_Return_WEIGHT",
        ],
        "Risk Parity Model: Asset Weight",
    )

    # Mean-Variance Model
    apply_strategy(
        data_MV,
        [
            "ASSET1_Return",
            "ASSET2_Return",
            "ASSET3_Return",
            "ASSET4_Return",
            "ASSET5_Return",
        ],
        T,
        strategy_type="MV",
    )

    calculate_cumulative_return(
        data_MV,
        [
            "PORT_RETURN",
            "ASSET1_Return",
            "ASSET2_Return",
            "ASSET3_Return",
            "ASSET4_Return",
            "ASSET5_Return",
        ],
    )

    draw_return_rate(
        data_MV,
        [
            "PORT_RETURN",
            "ASSET1_Return",
            "ASSET2_Return",
            "ASSET3_Return",
            "ASSET4_Return",
            "ASSET5_Return",
        ],
        "Mean-Variance Model: Asset Return",
    )

    draw_weights(
        data_MV,
        [
            "ASSET1_Return_WEIGHT",
            "ASSET2_Return_WEIGHT",
            "ASSET3_Return_WEIGHT",
            "ASSET4_Return_WEIGHT",
            "ASSET5_Return_WEIGHT",
        ],
        "Mean-Variance Model: Asset Weight",
    )
