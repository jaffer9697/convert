from pdf2image import convert_from_path
import os
import shutil
import torch
from transformers import MarianMTModel, MarianTokenizer

# Path to the PDF file
pdf_path = r'C:\Users\USER\syllabus.pdf'

# Directory to save the temporary images
temp_image_folder = 'temp_images'
if not os.path.exists(temp_image_folder):
    os.makedirs(temp_image_folder)

# Convert PDF to a list of PIL images at 300 DPI
images = convert_from_path(pdf_path, dpi=300, output_folder=temp_image_folder)

# Initialize the model and tokenizer for Tamil translation
model_name = "Helsinki-NLP/opus-mt-en-ta"  # English to Tamil translation
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

try:
    # Initialize a list to store the translated text
    tamil_text_list = []

    for i, image in enumerate(images):
        # OCR and translation pipeline
        # Note: You may need to preprocess the image before passing it to the model
        # Here, we assume the text is already extracted from the image

        # Replace this with your OCR code if needed
        extracted_text = "Extracted text from OCR"

        # Translate extracted text from English to Tamil
        inputs = tokenizer(extracted_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            translated_text = model.generate(**inputs)
        tamil_text = tokenizer.batch_decode(translated_text, skip_special_tokens=True)

        # Append translated text to the list
        tamil_text_list.append(tamil_text)

    # Combine all the Tamil text into a single string
    full_tamil_text = '\n'.join(tamil_text_list)

    # Save the translated text to a file
    tamil_text_file_path = 'output_tamil_text_file.txt'
    with open(tamil_text_file_path, 'w', encoding='utf-8') as tamil_text_file:
        tamil_text_file.write(full_tamil_text)

    print(f"Tamil text has been extracted and saved to {tamil_text_file_path}")

finally:
    # Ensure all images are closed and remove the temporary images and directory
    def remove_temp_files(temp_image_folder):
        for file_name in os.listdir(temp_image_folder):
            file_path = os.path.join(temp_image_folder, file_name)
            try:
                os.remove(file_path)
            except PermissionError:
                print(f"Permission error encountered while trying to delete {file_path}")
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")

    remove_temp_files(temp_image_folder)
    try:
        shutil.rmtree(temp_image_folder)
    except Exception as e:
        print(f"Error removing directory {temp_image_folder}: {e}")
