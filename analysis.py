import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['MS Gothic']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 读取数据
data = pd.read_csv("agri_data.csv")
print("前5行数据：")
print(data.head())
print("\n数据统计：")
print(data.describe())
# AI模型
X = data[['temperature', 'humidity', 'soil_moisture']]
y = data['watering_needed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("\n模型准确率:", accuracy)
print("特征重要性:", model.feature_importances_)


# 可视化
# 按是否需要浇水分组
need_water = data[data['watering_needed'] == 1]
no_water = data[data['watering_needed'] == 0]

# 预测
print("\n请输入当前环境数据：")
temp = float(input("请输入温度: "))
hum = float(input("请输入湿度: "))
soil = float(input("请输入土壤湿度: "))
new_data = pd.DataFrame([[temp, hum, soil]],
                        columns=['temperature', 'humidity', 'soil_moisture'])
prediction = model.predict(new_data)[0]
if prediction == 1:
    print("🌱 建议：需要浇水！土壤偏干，注意补水。")
else:
    print("🌿 建议：暂时不需要浇水，土壤湿度正常。")
# 画历史数据
plt.scatter(need_water['temperature'], need_water['soil_moisture'],
            label='Need Water', alpha=0.7)

plt.scatter(no_water['temperature'], no_water['soil_moisture'],
            label='No Water', alpha=0.7)

# 🔴 标出当前输入（重点！！）
plt.scatter(temp, soil, color='red', s=120, label='Your Input')

# 📌 标注预测结果
if prediction == 1:
    label_text = "Need Water"
else:
    label_text = "No Water"

plt.text(temp, soil, label_text, fontsize=12, color='red')

# 图设置
plt.xlabel("Temperature")
plt.ylabel("Soil Moisture")
plt.title("Smart Agriculture Decision System")
plt.legend()
plt.grid()

plt.show()



