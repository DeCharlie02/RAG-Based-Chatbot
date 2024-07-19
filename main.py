import io
import faiss
from PyPDF2 import PdfReader 
import numpy as np
import pytesseract
import streamlit as st
import torch
from openai import OpenAI
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# Streamlit configuration
st.set_page_config(page_title="Scientific Literature RAG Chatbot",
                   layout="wide")
st.title("RAG Chatbot for Scientific Literature Analysis")

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'index' not in st.session_state:
    st.session_state.index = faiss.IndexFlatL2(1536)

# Login functionality
if not st.session_state.logged_in:
    st.header("Login")
    email = st.text_input("Enter your email ID")
    if st.button("Login"):
        if email:
            st.session_state.logged_in = True
            st.session_state.email = email
            st.experimental_rerun()
        else:
            st.warning("Please enter your email ID.")

if st.session_state.logged_in and st.sidebar.button("Logout"):
    st.session_state.clear()
    st.experimental_rerun()

if not st.session_state.logged_in:
    st.stop()

# OpenAI API setup
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API key to use the chatbot.")
    st.stop()

client = OpenAI(api_key=api_key)


# Image analysis model initialization
@st.cache_resource
def load_image_model():
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "microsoft/resnet-50")
    model = AutoModelForImageClassification.from_pretrained(
        "microsoft/resnet-50")
    return feature_extractor, model


feature_extractor, image_model = load_image_model()


# Utility functions
def get_embeddings(texts):
    try:
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-ada-002")

        return np.array([item.embedding for item in response.data])

    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return None


def analyze_image(image):
    # Classify image
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = image_model(**inputs)
    predicted_class_id = outputs.logits.argmax(-1).item()
    predicted_class = image_model.config.id2label[predicted_class_id]

    # Extract text from image
    text = pytesseract.image_to_string(image)

    return {"class": predicted_class, "extracted_text": text}


# PDF processing
st.header("Upload a PDF file")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        try:
            pdf_file = io.BytesIO(uploaded_file.read())
            pdf_reader = PdfReader(pdf_file)

            text = ""
            images = []
            image_analyses = []

            for page in pdf_reader.pages:
                text += page.extract_text()

                # Extract images (Note: PyPDF2 doesn't have built-in image extraction)
                # You might need to use a different library for image extraction
                # This is a placeholder for image extraction
                # images.append(extract_images_from_page(page))
                # image_analyses.append(analyze_image(images[-1]))

            if text:
                # Combine text with image analysis results
                full_text = text + "\n\n" + "\n".join([
                    f"Image {i+1}: {analysis['class']}, Text: {analysis['extracted_text']}"
                    for i, analysis in enumerate(image_analyses)
                ])
                embeddings = get_embeddings([full_text])
                if embeddings is not None:
                    if 'index' not in st.session_state:
                        st.session_state.index = faiss.IndexFlatL2(1536)
                    if 'documents' not in st.session_state:
                        st.session_state.documents = []
                    st.session_state.index.add(x=embeddings,n=len(embeddings))
                    st.session_state.documents.append(full_text)
                st.success("PDF content extracted, analyzed, and stored.")

                with st.expander("View Extracted Text"):
                    st.write(full_text)

                if images:
                    st.write("Images extracted:")
                    for i, (img, analysis) in enumerate(zip(images, image_analyses)):
                        st.image(img, caption=f"Image {i+1}: {analysis['class']}")
            else:
                st.warning("No text could be extracted from the PDF.")
        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {e}")

# Question answering functionality
st.header("Ask me a question")
query = st.text_input("Enter your question here:",
                      placeholder="Type your question and hit Enter...")

if query:
    with st.spinner("Finding the answer..."):

        def retrieve_documents(query):
            query_embedding = get_embeddings([query])
            if query_embedding is None:
                return []
            D, I = st.session_state.index.search(query_embedding, k=1)
            return [st.session_state.documents[i] for i in I[0]]

        def generate_response(query):
            relevant_docs = retrieve_documents(query)
            if not relevant_docs:
                return "No relevant documents found."
            context = ' '.join(relevant_docs)
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{
                        "role":
                        "system",
                        "content":
                        "You are a helpful assistant specializing in scientific literature. Provide detailed answers and insights based on the given context."
                    }, {
                        "role":
                        "user",
                        "content":
                        f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
                    }])
                content = response.choices[0].message.content
                return content.strip() if content is not None else "No response content received."
            except Exception as e:
                return f"Error generating response: {e}"

        response = generate_response(query)
        st.write("Response:")
        st.write(response)

# Additional UI elements
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This chatbot analyzes scientific literature using advanced AI. Upload a PDF to extract text and images, then ask questions to get insights."
)

st.markdown("---")
st.markdown("Created by Ashish Kumar sahu")

with st.expander("Provide Feedback"):
    feedback = st.text_area("Your feedback helps us improve the chatbot.",
                            placeholder="Enter your feedback here...")
    if st.button("Submit Feedback"):
        if feedback:
            st.success("Thank you for your feedback!")
            # Here you would typically save the feedback to a database or file
        else:
            st.warning("Please enter your feedback before submitting.")

with st.expander("Help"):
    st.markdown("""
    ### How to Use the Chatbot

    1. **Login**: Enter your email to access the chatbot.
    2. **API Key**: Enter your OpenAI API key in the sidebar.
    3. **Upload a PDF**: Use the file uploader to select a scientific paper in PDF format.
    4. **Process Content**: The chatbot will extract text and images, analyze them, and store the information.
    5. **Ask Questions**: Enter your questions in the text box to get answers based on the extracted content.
    6. **View Results**: See extracted text, images, and their classifications in the expandable sections.
    7. **Provide Feedback**: Help us improve by sharing your experience with the chatbot.
    """)

with st.expander("Contact"):
    st.markdown("""
    For any inquiries or support, please contact us at:

    - Email: [sahuashish22706866@gmail.com]
    """)

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
