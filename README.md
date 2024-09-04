# Portfolio Risk Management

This project explores risk management models applied to Hedge Fund Indices, focusing on evaluating risk metrics such as Value at Risk (VaR), Conditional Value at Risk (CVaR), and semi-deviation. The goal is to analyze and compare the risk profiles and tail risks of various hedge fund strategies using statistical models, including Gaussian and Cornish-Fisher adjustments for skewness and kurtosis.

## Project Overview

### Objective
- Analyze and compare risk metrics for Hedge Fund Indices:
  - **Semi-deviation**: Measures downside risk.
  - **VaR (Value at Risk)**: Estimates potential losses.
  - **CVaR (Conditional VaR)**: Quantifies risk beyond VaR.
  - **Gaussian and Cornish-Fisher Adjustments**: Adjust for skewness and kurtosis in returns.

### Dataset
Data is sourced from the EDHEC Business School’s **Investment Management with Python and Machine Learning Specialization**, which provides a comprehensive introduction to investment management using data science and machine learning techniques.

#### Datasets
- **File 1: "Hedge_Fund_Strategies_Analysis.csv"**
  - **Description**: Monthly hedge fund returns from 1997, covering 12 strategies. The date column indicates the end of the return period.
  
- **File 2: "ME_Portfolio_Metrics_Monthly.csv"**
  - **Description**: Returns and metrics for portfolios based on market equity, with assets equally weighted across various categories (e.g., small cap, large cap).

## Features of the Analysis

### Risk Metrics Calculated
1. **Semi-deviation**: Captures downside risk by focusing on negative returns.
2. **VaR**:
   - **Historical**: Based on past returns.
   - **Gaussian**: Assumes normally distributed returns.
   - **Cornish-Fisher**: Adjusts for skewness and kurtosis.
3. **CVaR**: Average loss beyond the VaR threshold.

### Insights
- **Semi-Deviation**: Indicates higher downside risk for Emerging Markets and Short Selling; lower for Equity Market Neutral.
- **VaR**:
  - **Historical**: Highest for Short Selling and Emerging Markets.
  - **Gaussian**: Provides a baseline but may underestimate risk for skewed distributions.
  - **Cornish-Fisher**: Highlights greater risk for Emerging Markets and Short Selling, reflecting non-normal returns.
- **Skewness and Kurtosis**:
  - **Skewness**: Many strategies show negative skewness, indicating a tendency toward larger negative returns.
  - **Kurtosis**: High values in strategies like Fixed Income Arbitrage suggest fat tails and higher likelihoods of extreme events.

## Results Summary

### Semi-Deviation
- Emerging Markets and Short Selling show the highest semi-deviation, indicating greater downside risk.

### Value at Risk (VaR)
- **Historical VaR**: Highest for Short Selling and Emerging Markets.
- **Gaussian VaR**: Underestimates risk for non-normal distributions.
- **Cornish-Fisher VaR**: Adjusted for skewness and kurtosis, revealing greater risks in Emerging Markets and Short Selling.

### Conditional Value at Risk (CVaR)
- Highest in Short Selling and Emerging Markets, indicating susceptibility to extreme negative returns.

## Technologies Used

### Programming Languages and Libraries
- **Python**
  - **Data Processing**: pandas, numpy
  - **Risk Analysis**: scipy
  - **Visualization**: matplotlib, seaborn

### Risk Models
- **VaR**: Historical, Gaussian, Cornish-Fisher
- **CVaR**
- **Semi-Deviation**

## Observations and Conclusion

### Key Observations
1. **Downside Risk**: High for Short Selling and Emerging Markets; lower for Equity Market Neutral and Merger Arbitrage.
2. **Fat Tails**: High kurtosis in many strategies indicates more frequent extreme events.
3. **Negative Skewness**: Many strategies exhibit a higher likelihood of extreme negative returns.

### Conclusion
This analysis provides insights into the tail risks and return distributions of various hedge fund strategies. Utilizing VaR and CVaR models helps investors understand potential losses and prepare for extreme market conditions. Strategies with high kurtosis and negative skewness may need more robust risk management strategies.

## Support/Contact
For further inquiries or support, please contact:

- **Email:** mdngandu@gmail.com
- **LinkedIn:** [Mwansa Ng'andu](https://www.linkedin.com/in/mwansangandu)
- **GitHub:** [Mwansa Ng'andu](https://github.com/MwansaNgandu)

Feel free to reach out with any questions or for more information about this project.