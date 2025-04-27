import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "vector": [ [6.5,3.0,5.2,2.0] ]
}

response = requests.post(url, json=data)
print("Prediction response:", response.json())