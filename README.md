# Application-Supplement
# Multi-Asset Portfolio Allocation Project

## Overview

This project focuses on the creation of two asset allocation models for a multi-asset portfolio. The primary objective is to optimize the allocation of assets to achieve desirable outcomes. The two models utilized in this project are the Risk Parity Model and the Mean-Variance Model.

## Project Components

1. **Data Import and Processing:** The project begins with the import of necessary libraries, including pandas, numpy, scipy, and matplotlib. Data processing functions are included to transform raw asset data into more usable formats.

2. **Financial Evaluation Indicators:** To evaluate the performance of the asset allocation models, various financial evaluation indicators are calculated. These indicators provide insights into the risk and return characteristics of the portfolio.

3. **Risk Parity Model:** The Risk Parity Model aims to distribute assets in a way that minimizes risk while maintaining a balanced allocation. The model calculates portfolio weights and performance indicators over time.

4. **Mean-Variance Model:** The Mean-Variance Model seeks to maximize return while considering the trade-off between risk and return. Similar to the Risk Parity Model, it calculates portfolio weights and performance indicators.

5. **Rolling Window Strategy:** Both allocation models are applied using a rolling window approach. This approach involves generating portfolio weights, returns, and net asset values for different time intervals, allowing for dynamic asset allocation.

6. **Cumulative Returns:** Cumulative returns are calculated to visualize the performance of the portfolio over time. These visuals provide insights into the effectiveness of the allocation models.

7. **Visualizations:** The project includes functions for plotting and visualizing cumulative returns and portfolio weights over time. These visualizations are essential for understanding the performance of the portfolio.

## Results
The project includes functions for plotting and visualizing cumulative returns and portfolio weights over time. These visualizations are essential for understanding the performance of the portfolio.

   - Risk Parity Model: Asset Return
     ![Risk Parity Model: Asset Return](Visualization/Risk_Parity_Model_Asset_Returns.png)

   - Mean-Variance Model: Asset Return
     ![Mean-Variance Model: Asset Return](Visualization/Mean_Variance_Model_Asset_Returns.png)

   - Risk Parity Model: Asset Weight
     ![Risk Parity Model: Asset Weight](Visualization/Risk_Parity_Model_Asset_Weights.png)

   - Mean-Variance Model: Asset Weight
     ![Mean-Variance Model: Asset Weight](Visualization/Mean_Variance_Model_Asset_Weights.png)

...


## Conclusion

This project aims to optimize asset allocation for a multi-asset portfolio by employing two distinct models: the Risk Parity Model and the Mean-Variance Model. It encompasses data processing, the calculation of financial evaluation indicators, and the generation of visualizations to assess the performance of these allocation strategies.

The project provides a comprehensive approach to asset allocation and financial analysis, offering insights into the trade-offs between risk and return. It is a valuable resource for individuals and organizations seeking to make informed investment decisions and achieve their financial goals.

For code details and implementation, please refer to the provided code in the Python script.
