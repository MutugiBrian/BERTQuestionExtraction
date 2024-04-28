import streamlit as st
from transformers import BertTokenizer,BertForSequenceClassification
import torch
import pandas as pd
import re
import pytesseract
import sklearn
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import numpy as np
from scipy.sparse import hstack
import cv2
from PIL import Image
import io
import sweetviz
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import base64
import os


# Define the navigation structure
# pages = {
#     "Training Data": "This is the Training Data page.",
#     "Extract": "This is the default Extraction page.",
#     "Downloads": "This is the Downloads page."
# }

# Define the navigation structure
pages = ["EDA", "Extract", "Downloads"]

def get_current_page_index(pages, current_selection):
    """ Return the index of the current page in the list of pages. """
    return pages.index(current_selection)

# Sidebar navigation
st.sidebar.title("Navigation")
current_selection = st.sidebar.radio("Go to", pages, index=1)  # Default to "Extract"

# Calculate the current page index
current_page_index = get_current_page_index(pages, current_selection)
st.sidebar.write(f"Current page index: {current_page_index}")  # Display the current page index


def clear_cache():
    st.cache_data.clear()
    st.cache_resource.clear()

def get_file_content_as_base64(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode()

def create_download_link(file_path, download_name):
    b64 = get_file_content_as_base64(file_path)
    href = f'<li><a href="data:file/zip;base64,{b64}" download="{download_name}">Download {download_name}</a></li>'
    return href


# Define the page content functions
def load_page(page_name):
    if page_name == "EDA":
        st.title('Train-test Data Analysis Report')
        df = pd.read_csv('downloads/data.csv')

        # Generate the profiling report
        profile = df.profile_report()

        # Use Streamlit's function to display the report
        st_profile_report(profile)
        
        # df.dropna(inplace=True)
        # my_report = sweetviz.analyze([df,"Train"],target_feat='is_question')
        # my_report.show_html()
    elif page_name == "Extract":
        # st.header("Extract")
        # st.write("Use this page to perform text extraction tasks.")
        st.title('Exam Question Extraction with BERT')
        st.sidebar.button("Refresh Program",on_click=clear_cache)
    elif page_name == "Downloads":
        # Streamlit page setup
        st.title('Downloads Page')
        st.write('Please click the links below to download the resources:')
                                
        # List of download links
        download_links = """
        <ul>
            {0}
            {1}
        </ul>
        """.format(
            create_download_link('downloads/data.csv', 'data.csv'),
            create_download_link('downloads/exam_question_extraction.ipynb', 'question_extraction.ipyn'),
            # create_download_link('../models/BERT', 'BERT_Model.zip')
        )

        st.markdown(download_links, unsafe_allow_html=True)

# Display the selected page content
load_page(current_selection)


if current_selection == "Extract":
    # Set the default figure size
    plt.rcParams['figure.figsize'] = [6, 3]
    # Configure Pytesseract path
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    nltk.download('punkt')


    # Set the default figure size
    plt.rcParams['figure.figsize'] = [6, 3]

    # Set the global theme for Seaborn
    sns.set_theme(
        style='whitegrid',  # Sets the background to white grid
        palette='deep',     # Sets the color palette
        font='sans-serif',  # Sets the font to sans-serif
        font_scale=1.25,    # Scales up the font size by 25%
        color_codes=True,   # Allow the use of shorthand color codes
        rc={"figure.figsize": (6, 3)}  # Sets the default figure size
    )

    selected_features  = ['contains_prompt_word','starts_with_prompt_word','ends_with_ks', 'indexed_start', 'lower_start', 'upper_start']


    # @st.cache_data(allow_output_mutation=True)
    def get_model():
        model = BertForSequenceClassification.from_pretrained("MutugiBrian/BERTQuestionLabelling",token='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')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return tokenizer,model



    def extract_attributes(text):
        # List of prompt words
        prompt_words = ["what", "why", "when", "where", "explain", "describe", "give", "how", "discuss", "write", "define","do","analyze","compare"]

        # First word, ignoring leading spaces and converting to lowercase for case-insensitive comparison
        clean_line = text.strip()

        # Split the line into words
        words = clean_line.split()

        second_word = ""
        # Check if there are at least three words to get the word between the first and second space
        if len(words) >= 3:
            second_word = words[1]  # This is the word between the first and second spaces
        else:
            second_word = ""

        first_word = clean_line.split()[0].lower() if text.strip() else ""
        single_word = int(len(text.split()) == 1) if text else 0

        # Check if the first word is a prompt word
        starts_with_prompt_word = int((first_word.lower() in prompt_words) or (second_word.lower() in prompt_words))
        contains_prompt_word = int(any(pw in text.lower() for pw in prompt_words))


        # Refined patterns for 'indexed_start'
        patterns = [
            r"^\(?[a-zA-Z]\)\s",    # Matches patterns like (a) , b) 
            r"^\d+\.\s",            # Matches patterns like 1. 
            r"^\(?[ivxlcdm]+\)\s",  # Matches patterns like (i) , ii) 
        ]
        # Check the pattern against the first 1 to 5 characters of the text
        indexed_start = int(any(re.match(pattern, text.strip()[:7]) for pattern in patterns))

        question_start = int(first_word == "question")
        last_four_chars = text[-4:] if len(text) >= 4 else text
        last_q_chars = ["ks) ","rks) ","mks) ","ks] ","rks] ","mks] ","ks} ","rks} ","mks} ","ks)","rks)","mks)","ks]","rks]","mks]","ks}","rks}","mks}"]
        # ends_with_ks = int(last_four_chars == "ks) " or last_four_chars == "rks)"  or last_four_chars == "mks)")
        ends_with_ks = int(any(text.endswith(q_char) for q_char in last_q_chars))
        contains_ks = int(any(q_char in text for q_char in last_q_chars))
        lower_start = int(text.strip()[0].islower()) if text.strip() else 0
        upper_start = int(text.strip()[0].isupper()) if text.strip() else 0

        return first_word, indexed_start, question_start, last_four_chars, ends_with_ks, contains_ks, lower_start, upper_start, contains_prompt_word, starts_with_prompt_word, single_word

    def prep_text(raw_text):
        # Split the raw text into lines
        lines = raw_text.split('\n')

        # Remove empty lines
        lines = [line for line in lines if line.strip()]

        # Initialize variables
        processed_lines = []
        current_line = ''

        # Iterate over the lines to process them according to the specified conditions
        for i in range(len(lines)):
            line = lines[i].strip()

            # Flag to determine if the current line should be joined with the next
            join_with_next = False

            # If the line does not end with ')'...
            if not line.endswith(')'):
                # Check if the next line should be joined with the current line
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    # If the next line starts with a lowercase letter or doesn't start with a capital letter/parenthesis
                    # Or if the next line is less than 11 characters and ends with ')'
                    if (next_line and (next_line[0].islower() or not re.match(r'^[A-Z(]', next_line[:4]))) or \
                            (len(next_line) < 16 and next_line.endswith(')')):
                        current_line += line + ' '  # Join with the next line
                        join_with_next = True

            # If not joining with the next line, process the current line
            if not join_with_next:
                processed_line = current_line + line
                if processed_line:
                    processed_lines.append(processed_line)
                current_line = ''  # Reset the current line buffer
            else:
                # Skip processing the next line as it has been joined
                continue

        # Add the last line if it's not empty and has not been processed yet
        if current_line:
            processed_lines.append(current_line.strip())

        # Join processed lines with line breaks and return
        refined_text = '\n'.join(processed_lines)
        return refined_text

    def features_from_text_line(text):
        # Assuming extract_attributes is a function you've defined
        # Adapt this part according to your specific implementation
        selected_features  = ['contains_prompt_word','starts_with_prompt_word','ends_with_ks', 'indexed_start', 'lower_start', 'upper_start']
        vectorizer = TfidfVectorizer()
        extracted_features = extract_attributes(text)
        features_df = pd.DataFrame([extracted_features])
        numeric_features = features_df[selected_features]
        line_vectorized = vectorizer.transform([text])
        combined_features = hstack([line_vectorized, np.array(numeric_features).reshape(1, -1)])
        return combined_features


    def classify_text(all_text):
        lines = all_text.split('\n')
        results = []
        all_questions = []

        for line in lines:
            if line.strip():  # Ensure the line is not empty
                # Encode the line using the tokenizer
                encoded_line = tokenizer.encode_plus(
                    line,
                    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                    max_length=64,  # Maximum length for the text, truncate if longer
                    padding='max_length',  # Pad shorter sentences
                    return_attention_mask=True,
                    return_tensors='pt',  # PyTorch tensors format
                )

                # Get input ids and attention mask tensors, and move them to the device
                input_ids = encoded_line['input_ids']
                attention_mask = encoded_line['attention_mask']

                # Predict the label using the BERT model
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)
                    prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
                    confidence = torch.softmax(outputs.logits, dim=1).max().item()

                prediction_label = "Question" if prediction == 1 else "Not Question"
                results.append({'text': line, 'label': prediction_label, 'confidence': confidence})

                # Add to all_questions if it's predicted as a question with high confidence
                if prediction == 1 and confidence > 0.5:
                    all_questions.append(line)

        # Display results and questions in the Streamlit app
        return results, all_questions

        
    # Load your model
    tokenizer,model=get_model()

    # Set maximum number of files
    MAX_FILES = 5
    MAX_FILE_SIZE_MB = 50
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024  # convert MB to bytes

    uploaded_files = st.file_uploader("#### Upload exam question paper pictures/images :",accept_multiple_files = True, type='jpg')
    if uploaded_files:
        if len(uploaded_files) > MAX_FILES:
            st.error(f"Please upload no more than {MAX_FILES} files.")
        else:
            for uploaded_file in uploaded_files:
                for uploaded_file in uploaded_files:
                    # Check the size of each file
                    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
                        st.error(f"{uploaded_file.name} is too large; please upload files up to {MAX_FILE_SIZE_MB} MB.")
                    # else:
                    #     # Process the file if it meets the size requirements
                    #     st.write(f"Uploaded {uploaded_file.name} successfully!")
                    #     # Additional file processing can go here
                    # st.write(uploaded_file.name)



    if uploaded_files:
        if st.button('Extract Text and Exam Questions'):
            all_text = ''
            results = []  # Store tuples of (filename, image, text)
            for uploaded_file in uploaded_files:
                # Display the image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Convert image to grayscale for OCR
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                with st.spinner(f'Extracting text from {uploaded_file.name}...'):
                    # Extract text using Pytesseract
                    # new_text = pytesseract.image_to_string(gray)
                    custom_oem_psm_config = r"--oem 3 --psm 6"
                    new_text = pytesseract.image_to_string(gray, config=custom_oem_psm_config)
                    results.append((uploaded_file.name, image, new_text))

            for filename, img, extracted_text in results:
                all_text += extracted_text + '\n'
            #     st.write(f"##### File: {filename}")
            #     # st.image(img, caption=f"Uploaded Image: {filename}", use_column_width=True)
            #     st.text_area("Extracted Text", extracted_text, height=150)
            st.write(f"### All Text from Image(s)")
            st.text_area("Text from all images", all_text, height=400)

            with st.spinner(f'Extracting Exam Questions using BERT'):
                lines = all_text.split('\n')
                results = []
                all_questions = []

                for line in lines:
                    if line.strip():  # Ensure the line is not empty
                        # Encode the line using the tokenizer
                            # Encode the line using the tokenizer
                        encoded_line = tokenizer.encode_plus(
                            line,
                            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                            max_length=64,  # Maximum length for the text, truncate if longer
                            padding='max_length',  # Pad shorter sentences
                            return_attention_mask=True,
                            return_tensors='pt',  # PyTorch tensors format
                        )

                        # Get input ids and attention mask tensors, and move them to the device
                        input_ids = encoded_line['input_ids']
                        attention_mask = encoded_line['attention_mask']

                        # Predict the label using the BERT model
                        with torch.no_grad():
                            outputs = model(input_ids, attention_mask=attention_mask)
                            prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]  # Convert to numpy and get the label
                            confidence = torch.softmax(outputs.logits, dim=1).max().item()  # Softmax to get probabilities, and then the max for confidence


                        prediction_label = "Question" if prediction == 1 else "Not Question"
                        results.append({'text': line, 'label': prediction_label, 'confidence': confidence})

                        # Add to all_questions if it's predicted as a question with high confidence
                        if prediction == 1 and confidence > 0.5:
                            all_questions.append(line)

                        prediction = "Question" if prediction==1 else "Not Question"
                        # st.write("text: "+str(line)+", label: "+str(prediction)+", confidence: "+str(confidence))


            

            st.write(f"### Extracted Questions")
            for result in results:
                if result['label'] == 'Question':
                    st.write(f"{result['text']}  - {result['label']}, confidence: ({result['confidence']*100:.2f}%)")

    # if uploaded_files:
    #     if st.button('Extract Text'):
    #         ocr_text = ''
    #         for uploaded_file in uploaded_files:
    #             with st.spinner('Extracting...'):
    #                 # Convert to a format that OpenCV can read
    #                 file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    #                 image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    #                 # Convert image to grayscale
    #                 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #                 # Use pytesseract to extract text
    #                 new_text = pytesseract.image_to_string(gray)
    #                 ocr_text += new_text + '\n'
