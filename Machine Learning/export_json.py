import json

# Assuming you have a JSON data structure (replace this with your actual JSON data)
# Read JSON content
with open('test.json', 'r') as json_file:
    json_data = json.load(json_file)

# Specify the path for the output text file
output_file_path = "input.txt"

# Export JSON data to a text file
with open(output_file_path, "w") as output_file:
    json.dump(json_data, output_file, indent=2)  # The 'indent' parameter adds indentation for better readability
