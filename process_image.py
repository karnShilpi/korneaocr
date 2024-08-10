import cv2
import os
import numpy as np
import json
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from typing import List, Sequence
import pandas as pd 
import pytesseract
from pytesseract import Output

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def get_greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    return cv2.threshold(image, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def canny(image):
    return cv2.Canny(image, 100, 200)
    
def remove_noise(image):
    return cv2.medianBlur(image, 3)

def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def google_doc_ai_prediction(image):
   
    header_field = {}
    max_values_per_table = {}
    if image:
         _, extension = os.path.splitext(image)
         
    project_id='562667290269'
    location = 'us'
    processor_id = '949abc6f8357e58a'
    #doc_data_json=process_document_form_sample(project_id, location, processor_id, image,'image/tiff') 
    doc_data_json={'Supplier Invoice #:': '0095674283', 'Ship From:': 'SNIDER FLEET SOLUTIONS\n603 Northpark Dr #100\nHouston, TX 77073', 'C#:': '105457662-105457662', 'Statement Date': '01/31/2024', 'AC#:': 'CPVU-001082', 'Originating Document Number': 'DH00047773', 'Ship To:': 'CIRCLE K CORPORATION - Waycross\n3020 Harris Rd\nWaycross, GA 31503\nC#: 105457662 - 79', 'Due Date': '02/24/2024', 'ABA#': 'Remit To:\nCorcentric, LLC\n62861 Collections Center Drive\nChicago, IL 60693\nPhone:(800) 608-0809\n071000039\nACCT# 8666287292', 'PO No.': '012224PW36126', 'PO Date': '01/24/2024', 'Delivery Information:': 'Ship Date:01/23/2024\nShip Via:Supplier', 'Term of Sale': 'Net 30 Days', 'Settlement Date': '01/25/2024', 'Shipped Date:': '01/23/2024', 'Supplier Invoice Date': '01/24/2024', 'Program Invoice Date': '01/25/2024', 'ACCT#': 
                    '8666287292', 'Received By :': 'L1:\nUK', 'Phone:(': '800) 608-0809', 'Vehicle License Number :': '3366007', 'Remit To:': '- Waycross\nCorcentric, 62861 Chicago, IL Phone:(800ABA#', 'VL:': '36126', 'Vehicle Number :': '36126', 'Odometer Reading :': '00000000', 'Trailer Number :': '36126', 'Name of orderer :': 'PAUL W', 'Casing Sold by :': 'L1:\nNA', 'Total Summary Taxes:': '72.23', 'Tax Type': 'Special Tax', 'Vehicle License State': 'S\nL1:\n:IN', 'Chicago,': 'IL 60693', 'Phone:': '(714)870-3800'}
    search_text_list = list(doc_data_json.values())
    return doc_data_json
    
    

def process_document_form_sample(project_id: str, location: str, processor_id: str, file_path: str, mime_type: str):
    # Online processing request to Document AI
    document = process_document(
        project_id, location, processor_id, file_path, mime_type
    )

    text = document.text

    print(f"There are {len(document.pages)} page(s) in this document.")

    # Read the form fields and tables output from the processor
    i = 0
    json_dict = {}
    for page in document.pages:
        i += 1
        print(f"\n\n**** Page {page.page_number} ****")

        print(f"\nFound {len(page.tables)} table(s):")
        tab_ctr=0
        for table in page.tables:
            tab_ctr+=1
            num_collumns = len(table.header_rows[0].cells)
            num_rows = len(table.body_rows)
            print(f"Table with {num_collumns} columns and {num_rows} rows:")

            # Print header rows
            print("Columns:")
            print_table_rows(table.header_rows, text)
            # Print body rows
            print("Table body data:")
            print_table_rows(table.body_rows, text)


            header_row_values: List[List[str]] = []
            body_row_values: List[List[str]] = []

            header_row_values = get_table_data(table.header_rows, document.text)
            body_row_values = get_table_data(table.body_rows, document.text)

            df = pd.DataFrame(
                data=body_row_values,
                columns=pd.MultiIndex.from_arrays(header_row_values),
            )

            df.to_csv("invoice.csv")
            print(f"\nFound {len(page.form_fields)} form field(s):")


        for field in page.form_fields:
            name = layout_to_text(field.field_name, text)
            value = layout_to_text(field.field_value, text)
            print(f"    * {repr(name.strip())}: {repr(value.strip())}")

        if(i==1):
            for field in page.form_fields:
                name = layout_to_text(field.field_name, text)
                value = layout_to_text(field.field_value, text)
                json_dict[name.strip()]=value.strip()
                print(f"    * {repr(name.strip())}: {repr(value.strip())}")

    jsonString = json.dumps(json_dict, indent=4)

    with open("header.json", "w") as json_file:
        json.dump(json_dict, json_file)
    return json_dict;

def process_document(
    project_id: str, location: str, processor_id: str, file_path: str, mime_type: str
) -> documentai.Document:
    # You must set the api_endpoint if you use a location other than 'us', e.g.:
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    name = client.processor_path(project_id, location, processor_id)

    # Read the file into memory
    with open(file_path, "rb") as image:
        image_content = image.read()

    # Load Binary Data into Document AI RawDocument Object
    raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)

    # Configure the process request
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)

    result = client.process_document(request=request)

    return result.document

def print_table_rows(
    table_rows: Sequence[documentai.Document.Page.Table.TableRow], text: str
) -> None:
    for table_row in table_rows:
        row_text = ""
        for cell in table_row.cells:
            cell_text = layout_to_text(cell.layout, text)
            row_text += f"{repr(cell_text.strip())} | "
        print(row_text)

def layout_to_text(layout: documentai.Document.Page.Layout, text: str) -> str:
    """
    Document AI identifies text in different parts of the document by their
    offsets in the entirety of the document's text. This function converts
    offsets to a string.
    """
    response = ""
    # If a text segment spans several lines, it will
    # be stored in different text segments.
    for segment in layout.text_anchor.text_segments:
        start_index = int(segment.start_index)
        end_index = int(segment.end_index)
        response += text[start_index:end_index]
    return response

def get_table_data(
    rows: Sequence[documentai.Document.Page.Table.TableRow], text: str
) -> List[List[str]]:
    """
    Get Text data from table rows
    """
    all_values: List[List[str]] = []
    for row in rows:
        current_row_values: List[str] = []
        for cell in row.cells:
            current_row_values.append(
                text_anchor_to_text(cell.layout.text_anchor, text)
            )
        all_values.append(current_row_values)
    return all_values

def text_anchor_to_text(text_anchor: documentai.Document.TextAnchor, text: str) -> str:
    """
    Document AI identifies table data by their offsets in the entirity of the
    document's text. This function converts offsets to a string.
    """
    response = ""
    # If a text segment spans several lines, it will
    # be stored in different text segments.
    for segment in text_anchor.text_segments:
        start_index = int(segment.start_index)
        end_index = int(segment.end_index)
        response += text[start_index:end_index]
    return response.strip().replace("\n", " ")
def draw_rectangles_on_words(img_draw_boxes, data_invoice, language='eng'):
        # Perform OCR to get data
        data = pytesseract.image_to_data(img_draw_boxes, output_type=Output.DICT, config=f'-l {language} --oem 1 --psm 3')
        print('data:', data)
        n_boxes = len(data["text"])
        
        # Convert search phrases list to a set for faster lookup
        search_phrases_set = set(data_invoice)

        # Combine detected words into phrases
        detected_phrases = []
        phrase = []
        for i in range(n_boxes):
            if float(data["conf"][i]) > 60:
                word = data["text"][i].strip()
                if word:
                    if not phrase:
                        phrase = [word]
                    else:
                        phrase.append(word)
                    
                    combined_phrase = " ".join(phrase)
                    if combined_phrase in search_phrases_set:
                        detected_phrases.append((combined_phrase, phrase))
                        phrase = []
                    elif len(combined_phrase) > max(len(p) for p in search_phrases_set):
                        phrase = []

        # Draw rectangles around the detected phrases
        for phrase, words in detected_phrases:
            indices = [i for i, w in enumerate(data["text"]) if w.strip() in words]
            if indices:
                x1 = min(data["left"][i] for i in indices)
                y1 = min(data["top"][i] for i in indices)
                x2 = max(data["left"][i] + data["width"][i] for i in indices)
                y2 = max(data["top"][i] + data["height"][i] for i in indices)
                img_draw_boxes = cv2.rectangle(img_draw_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Resize and show the image with detected phrases
        resized_img = resize_with_aspect_ratio(img_draw_boxes, height=900)
        cv2.imshow("Detected Phrases", resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()