import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def convert_to_numeric(value):
    try:
        return float(value)
    except ValueError:
        return np.nan

# 加载数据
data = pd.read_csv('pokemon_data.csv')  # 假设数据集已经存在并命名为pokemon_data.csv

# 将所有列转换为数值类型，非数值转换为NaN
for col in data.columns:
    data[col] = data[col].apply(convert_to_numeric)

# 使用均值填充NaN值
data.fillna(data.mean(), inplace=True)

# 数据预处理
# 假设我们需要预测的是'base_total'属性
X = data.drop(['base_total', 'name'], axis=1)  # 从特征矩阵中删除'name'列
y = data['base_total']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("True Base Total")
plt.ylabel("Predicted Base Total")
plt.title("True vs Predicted Base Total")
plt.show()

# 输出前十位对普通玩家最友好的精灵（根据预测结果排序）
friendly_pokemons = data.sort_values(by='base_total', ascending=False).head(10)
print("\nTop 10 Friendly Pokemons for Beginners:")
print(friendly_pokemons[['name', 'base_total']])