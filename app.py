from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
import pandas as pd
import joblib
import os


app = Flask(__name__)
CORS(app)  # ⭐ 允许前端调用API

model = joblib.load("model.pkl")
# =========================
# 加载模型（必须）
# =========================
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ model.pkl 不存在，请先运行 train.py")

model = joblib.load(MODEL_PATH)


# =========================
# API接口
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # 获取输入
        temp = float(data["temperature"])
        hum = float(data["humidity"])
        soil = float(data["soil_moisture"])

        # 转换成模型输入
        input_data = pd.DataFrame([[temp, hum, soil]],
                                   columns=['temperature', 'humidity', 'soil_moisture'])

        # 预测
        prediction = model.predict(input_data)[0]

        # 返回结果
        result_text = "需要浇水" if prediction == 1 else "不需要浇水"

        return jsonify({
            "prediction": int(prediction),
            "result": result_text
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


# =========================
# # 健康检查接口（可选但很加分）
# # =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "Smart Agriculture API is running"
    })

@app.route("/ui")
def ui():
    return render_template("Index.API.html")
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


