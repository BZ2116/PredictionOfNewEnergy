"""
author:Bruce Zhao
date: 2025/4/1 12:02
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# 数据输入
years = [2019, 2020, 2021, 2022, 2023, 2024]
market_share = [4.68, 5.4, 13.4, 23.92, 31.55, 40.92]
data = pd.Series(market_share, index=years)

# 参数选择：遍历 p 和 q，d=1
best_aic = np.inf
best_order = None
for p in range(3):
    for q in range(3):
        try:
            model = ARIMA(data, order=(p, 1, q)).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_order = (p, 1, q)
        except:
            continue

print(f"最优 ARIMA 参数: {best_order}, AIC: {best_aic}")

# 拟合最优模型
model = ARIMA(data, order=best_order).fit()
print(model.summary())

# 预测 2025 年
forecast = model.forecast(steps=1)

# 检查 forecast 的类型并正确索引
if isinstance(forecast, pd.Series):
    forecast_value = forecast.iloc[0]  # 使用 iloc 获取第一个预测值
else:
    forecast_value = forecast[0]  # 对于 ndarray，直接索引第一个值

print(f"2025 年市场份额预测: {forecast_value:.2f}%")
