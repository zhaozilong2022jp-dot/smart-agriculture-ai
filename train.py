import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
print("🚀 开始训练模型...")
# 读取数据
data = pd.read_csv("agri_data.csv")

# 特征 & 标签
X = data[['temperature', 'humidity', 'soil_moisture']]
y = data['watering_needed']

# 训练模型
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# 保存模型
joblib.dump(model, "model.pkl")

print("✅ 模型训练完成，已保存为 model.pkl")