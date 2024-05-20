import os
import io
import fitz
from google.cloud import vision
from PIL import ImageEnhance, ImageFilter, ImageOps
import shutil

# Ensure the GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\USER\Desktop\convert\gen-lang-client-0148295645-196d4eacee57.json"

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

def detect_text(image_path):
    """Detects text in the file using Google Cloud Vision API."""
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f'{response.error.message}')

    if texts:
        return texts[0].description
    return ''

try:
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Iterate through each page of the PDF
    text_list = []
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        image_list = page.get_pixmap()

        # Convert the PDF page to a PIL image
        image = Image.frombytes("RGB", [image_list.width, image_list.height], image_list.samples)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Save the preprocessed image
        image_path = os.path.join(temp_image_folder, f'page_{page_number+1}.png')
        preprocessed_image.save(image_path, 'PNG')

        # Perform OCR using Google Cloud Vision API
        text = detect_text(image_path)
        text_list.append(text)
        
    # Combine all the text into a single string
    full_text = '\n'.join(text_list)

    # Save the text to a file
    text_file_path = 'output_text_file.txt'
    with open(text_file_path, 'w', encoding='utf-8') as text_file:
        text_file.write(full_text)

    print(f"Text has been extracted and saved to {text_file_path}")

finally:
    # Close the PDF document
    pdf_document.close()

    # Remove the temporary images and directory
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
