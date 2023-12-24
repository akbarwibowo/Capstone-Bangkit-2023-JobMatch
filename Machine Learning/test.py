import requests

# Send the file content in the request body using the 'data' parameter
resp = requests.post("http://127.0.0.1:5000/", files={'file': open('test.json', 'r', encoding='utf-8')})
# resp = requests.get("http://127.0.0.1:5000/")

# Print the response
print(resp.json())
