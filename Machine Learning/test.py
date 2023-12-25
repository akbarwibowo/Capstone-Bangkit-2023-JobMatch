import requests

# Send the file content in the request body using the 'json' parameter
resp = requests.post("http://127.0.0.1:5000/", json={'degree': 'Sci&Tech',
                                                     'job': 3,
                                                     'key': 'java'})

# resp = requests.get("http://127.0.0.1:5000/")

# Print the response
print(resp.json())
