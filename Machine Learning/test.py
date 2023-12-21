import requests

# Read the content of the text file
with open('input.txt', 'r') as file:
    file_content = file.read()

# Send the file content in the request body using the 'data' parameter
resp = requests.post("http://127.0.0.1:5000", files={'file': open('input.txt', 'r')})

# Print the response
print(resp.json())
