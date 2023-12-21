import requests


url = "http://localhost:9696/predict"

patient = {"age": 22, "sex": 0, "alb": 33, "alp": 60, "alt": 23, "ast": 27, "bil": 44, "che": 10,
            "chol": 6, "crea": 77, "ggt": 28, "prot": 80}
print (patient)
response=requests.post(url, json=patient).json()
print(response)


#						