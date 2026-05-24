from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')  # ✅ 关闭GUI，改为后台绘图
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# 创建 static 文件夹（存图）
if not os.path.exists("static"):
    os.makedirs("static")

# =========================
# 训练模型
# =========================
data = pd.read_csv("agri_data.csv")

X = data[['temperature', 'humidity', 'soil_moisture']]
y = data['watering_needed']

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# =========================
# 首页
# =========================
import time  # ⭐ 放在最上面
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None
    # 可以提前算好，放到函数外
    need_water = data[data['watering_needed'] == 1]
    no_water = data[data['watering_needed'] == 0]
    if request.method == "POST":
        temp = float(request.form["temperature"])
        hum = float(request.form["humidity"])
        soil = float(request.form["soil"])

        new_data = pd.DataFrame([[temp, hum, soil]],
                                columns=['temperature', 'humidity', 'soil_moisture'])

        prediction = model.predict(new_data)[0]

        if prediction == 1:
            result = "🌱 需要浇水！"
        else:
            result = "🌿 不需要浇水"

        # =========================
        # 生成图表（关键🔥）
        # =========================
        plt.figure()

        plt.scatter(need_water['temperature'], need_water['soil_moisture'], label='Need Water')
        plt.scatter(no_water['temperature'], no_water['soil_moisture'], label='No Water')

        # 🔴 当前输入
        plt.scatter(temp, soil, color='red', s=100, label='Your Input')

        plt.xlabel("Temperature")
        plt.ylabel("Soil Moisture")
        plt.title("Smart Agriculture AI System")
        plt.legend()
        plt.grid()


        plt.savefig("static/result.png")
        plt.close()
        # ⭐ 防缓存关键！
        image_path = f"/static/result.png?t={int(time.time())}"
    return render_template("index.html", result=result, image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)