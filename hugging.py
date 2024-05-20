from ocr_tamil.ocr import OCR

image_path = r"C:\Users\USER\Desktop\convert\image\Capture.JPG" # insert your own path here
ocr = OCR()
text_list = ocr.predict(image_path)
print(text_list[0])

## OUTPUT : நெடுஞ்சாலைத்
