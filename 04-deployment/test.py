import requests

ride = {
    "taxi_type": yellow,
    "year": 2023,
    "month": May
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())