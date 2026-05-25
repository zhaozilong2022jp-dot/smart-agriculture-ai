import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "temperature": 28,
    "humidity": 55,
    "soil_moisture": 35
}

res = requests.post(url, json=data)

print("状态码:", res.status_code)
print("返回:", res.json())