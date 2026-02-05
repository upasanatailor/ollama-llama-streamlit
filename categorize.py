import ollama
import os

model = "gemma3"

input_file = "./data/grocery_list.txt" 
output_file = "./data/categorized_grocery_list.txt"

# Read the grocery list from the input file
if not os.path.exists(input_file):
    print(f"Input file {input_file} does not exist.")
    exit(1)


#read the uncategorized grocery list
with open(input_file, "r") as f:
    grocery_list = f.read().strip()

# Define the prompt for categorization
prompt = f"""
1.Categorize the following grocery items into appropriate categories and format the output as a list of categories with items under each category
{grocery_list}
2.short the items in alphabetical order under each category
3.present the categorized list in clean and clear format
"""

#send the prompt to the model and get the response
try:
    response = ollama.generate(
        model=model,
        prompt=prompt,)

    generated_text = response.get("response", "")

# Write the categorized grocery list to the output file
    with open(output_file, "w") as f:
        f.write(generated_text.strip())

        print(f"Categorized grocery list written to {output_file}")   
except Exception as e:
    print(f"An error occurred: {e}")
