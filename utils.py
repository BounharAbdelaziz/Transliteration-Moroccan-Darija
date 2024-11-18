# Function to transform text to lowercase
def lowercase_text(text):
    if isinstance(text, list):
        return [item.lower() for item in text]
    elif isinstance(text, str):
        return text.lower()
    else:
        return text