from pdf2image import convert_from_path
import os
import shutil
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageOps
from PIL import ImageFilter
import torch



from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification

# Path to the PDF file
pdf_path = r'C:\Users\USER\syllabus.pdf'

# Directory to save the temporary images
temp_image_folder = 'temp_images'
if not os.path.exists(temp_image_folder):
    os.makedirs(temp_image_folder)

def preprocess_image(image):
    """Preprocess the image for better OCR accuracy."""
    # Convert to grayscale
    image = image.convert('L')
    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    # Binarize the image
    image = ImageOps.invert(image)
    image = image.point(lambda x: 0 if x < 128 else 255, '1')
    # Apply a slight blur to remove noise
    image = image.filter(ImageFilter.MedianFilter())
    return image

try:
    # Convert PDF to a list of PIL images at 300 DPI
    images = convert_from_path(pdf_path, dpi=300, output_folder=temp_image_folder)

    # OCR each image and store the result in a list
    text_list = []
    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
    model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")

    for i, image in enumerate(images):
        preprocessed_image = preprocess_image(image)
        image_path = os.path.join(temp_image_folder, f'page_{i+1}.png')
        preprocessed_image.save(image_path, 'PNG')

        # Preprocess the image and tokenize it
        inputs = tokenizer(image_path, return_tensors="pt", padding=True, truncation=True)

        # Perform OCR using LayoutLM model
        with torch.no_grad():
            outputs = model(**inputs)

        # Decode the model's output into text
        pred_ids = outputs.logits.argmax(dim=-1)
        pred_texts = tokenizer.batch_decode(pred_ids)

        # Append the OCR output to text_list
        text = ' '.join(pred_texts)
        text_list.append(text)

    # Combine all the text into a single string
    full_text = '\n'.join(text_list)

    # Save the text to a file
    text_file_path = 'output_text_file.txt'
    with open(text_file_path, 'w', encoding='utf-8') as text_file:
        text_file.write(full_text)

    print(f"Text has been extracted and saved to {text_file_path}")

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
