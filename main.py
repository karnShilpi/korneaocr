from kivy.config import Config
Config.set("graphics", "width", "900")
Config.set("graphics", "height", "450")
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

import os
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.properties import StringProperty, ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.uix.textinput import TextInput
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.snackbar import Snackbar
from kivymd.uix.screen import MDScreen

from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.app import MDApp

from plyer import filechooser
from pytesseract import Output
from process_image import get_greyscale, google_doc_ai_prediction, draw_rectangles_on_words
import cv2
import csv

class ScreenDisplay(MDScreen):
    output_text = StringProperty("")
    file_path = StringProperty("")
    action_bar_color = ObjectProperty([.6, 4, .2, .6])  # Default red color

    def change_action_bar_color(self, color):
        self.action_bar_color = color

    # Default language is English
    # For more languages download the tesseract language packs
    language = StringProperty("eng")

    def select_image(self):
        try:
            self.file_path = filechooser.open_file(title="Pick an Image..", filters=[("Image", "*.jpeg", "*.jpg", "*.png", "*.tif")])[0]
        except IndexError as e:
            self.file_path = ""

    def detect_words(self):
        if self.file_path != "":
            data_invoice = google_doc_ai_prediction(self.file_path)
            search_text_list = list(data_invoice.values())
            print(search_text_list)
            # Read image
            img = cv2.imread(self.file_path)
            img_draw_boxes = get_greyscale(img)
            img_predict = get_greyscale(img)
            draw_rectangles_on_words(img, data_invoice)

            print('data_invoice', data_invoice)
            self.display_invoice_data(data_invoice)

    def format_invoice_data(self, data_invoice):
        formatted_data = []
        for key, value in data_invoice.items():
            formatted_data.append((key, value))
        return formatted_data

    def display_invoice_data(self, data_invoice):
        formatted_data = self.format_invoice_data(data_invoice)

        # Find the grid layout in the kv file by id
        grid_layout = self.ids.grid_layout
        grid_layout.clear_widgets()  # Clear previous widgets

        for key, value in formatted_data:
            key_label = TextInput(text=key, size_hint_y=None, height=40, multiline=False)
            value_textinput = TextInput(text=value, size_hint_y=None, height=40, multiline=False)

            grid_layout.add_widget(key_label)
            grid_layout.add_widget(value_textinput)

    def download_csv(self):
        grid_layout = self.ids.grid_layout
        formatted_data = []

        # Extract key-value pairs from GridLayout
        for i in range(0, len(grid_layout.children), 2):
            value_textinput = grid_layout.children[i]
            key_label = grid_layout.children[i + 1]
            key = key_label.text
            value = value_textinput.text
            formatted_data.append((key, value))

        file_path = 'output.csv'
        default_path = os.path.join(os.path.expanduser('~'), 'Downloads', file_path)
        print('default_path', default_path)
        self.write_csv(default_path, formatted_data)

    def write_csv(self, file_path, formatted_data):
        try:
            with open(file_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                for key, value in formatted_data:  # Iterate through the list of tuples
                    csv_writer.writerow([key, value])
            print(f'File saved to {file_path}')
            Snackbar(text="File downloaded successfully!").show()
        except Exception as e:
            print(f'Error saving file:')
            Snackbar(text="File not downloaded successfully!").show()
    def show_toast(self, message):
        snackbar = Snackbar(text=message)
        snackbar.open()

class TextRecognitionApp(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Blue"
        return ScreenDisplay()

TextRecognitionApp().run()
