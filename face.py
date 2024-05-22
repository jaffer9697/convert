import os
import time
import json
import multiprocessing
from tempfile import NamedTemporaryFile
from io import BytesIO
from PIL import Image
from pytesseract import image_to_string
from dotenv import load_dotenv
import pypdfium2 as pdfium
from transformers import AutoTokenizer, AutoModelForCausalLM
from jsonformer.main import Jsonformer

load_dotenv()

# 1. Convert PDF file into images via pypdfium2
def convert_pdf_to_images(file_path, scale=300/72):
    pdf_file = pdfium.PdfDocument(file_path)
    page_indices = [i for i in range(len(pdf_file))]
    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        scale=scale,
    )
    final_images = []
    for i, image in zip(page_indices, renderer):
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        final_images.append(dict({i: image_byte_array}))
    return final_images

# 2. Extract text from images via pytesseract
def extract_text_from_img(list_dict_final_images):
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []
    for index, image_bytes in enumerate(image_list):
        image = Image.open(BytesIO(image_bytes))
        raw_text = str(image_to_string(image))
        image_content.append(raw_text)
    return "\n".join(image_content)

def extract_content_from_url(url: str):
    images_list = convert_pdf_to_images(url)
    text_with_pytesseract = extract_text_from_img(images_list)
    return text_with_pytesseract

# 3. Extract structured info from text via LLM
class HuggingFaceLLM:
    def __init__(self, temperature=0, top_k=50, model_name="databricks/dolly-v2-12b"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=True, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_cache=True)
        self.top_k = top_k

    def generate(self, prompt, max_length=1024):
        json_schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "price": {"type": "number"}
                        }
                    }
                },
                "Company_name": {"type": "string"},
                "invoice_date": {"type": "string"},
            }
        }

        builder = Jsonformer(
            model=self.model,
            tokenizer=self.tokenizer,
            json_schema=json_schema,
            prompt=prompt,
            max_string_token_length=20
        )

        print("Generating...")
        output = builder()
        return output

def extract_structured_data(content: str, data_points):
    llm = HuggingFaceLLM(temperature=0)  # Choose the desired Hugging Face model

    template = """
    You are an expert admin people who will extract core information from documents

    {content}

    Above is the content; please try to extract all data points from the content above:
    {data_points}
    """

    # Fill in the placeholders in the template
    formatted_template = template.format(content=content, data_points=data_points)

    # Generate text using the formatted template
    results = llm.generate(formatted_template)

    return results

def main():
    default_data_points = """{
        "item": [{
            "description": "description or name of the item that has been bought",
            "price": "how much does the item cost"
        }],
        "Company_name": "company that issued the invoice",
        "invoice_date": "when was the invoice issued",
    }"""

    folder_path = r'folder_path = r'C:\Users\USER\Desktop\convert\resume.pdf_path'
    # Replace this with your folder path containing PDFs
    output_folder_path = './output'

    pdf_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith('.pdf')]

    results = []

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    if pdf_paths:
        total_start_time = time.time()
        with open("output_results.txt", "w") as output_file:
            for pdf_path in pdf_paths:
                output_file.write(f"PDF Path: {pdf_path}\n")
                start_time = time.time()  # Record the start time
                content = extract_content_from_url(pdf_path)
                data = extract_structured_data(content, default_data_points)
                json_data = json.dumps(data)
                output_filename = os.path.basename(pdf_path).split('.')[0] + '_output.txt'
                output_filepath = os.path.join(output_folder_path, output_filename)
                with open(output_filepath, 'w') as output_file:
                    output_file.write(json_data)
                if isinstance(json_data, list):
                    results.extend(json_data)
                else:
                    results.append(json_data)
                end_time = time.time()  # Record the end time
                elapsed_time = end_time - start_time
                output_file.write(f"Execution time: {elapsed_time:.2f} seconds\n")
                output_file.write(f"Results: {json_data}\n")
                output_file.write("\n")
        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time
        output_file.write(f"Total execution time: {total_elapsed_time:.2f} seconds\n")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
