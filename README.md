### Project Brief: Scientific Literature RAG Chatbot

**Overview**  
This project is a sophisticated Scientific Literature RAG (Retrieve and Generate) Chatbot designed to analyze and interact with scientific literature using advanced AI technologies. It enables users to upload PDF files, extract and analyze content, and ask questions based on the extracted information.

---

**Login Page**  
![image](https://github.com/user-attachments/assets/0d5905c0-5b7d-425f-b801-d094cdb661c5)

- **Function:** Allows users to log in using their email ID.
- **Technology:** Streamlit is used to create the web interface and manage session states.

**API Key Entry**  
![image](https://github.com/user-attachments/assets/2c4aaf45-8efb-4829-b8ca-060d0330af5d)

- **Function:** Users must enter their OpenAI API key to utilize the chatbot's AI capabilities.
- **Technology:** Streamlit for UI and OpenAI's API for embedding generation and chat completions.

**PDF Upload and Processing**  
![image](https://github.com/user-attachments/assets/11d49920-a5a1-4d34-8405-2c9d928780a1)

- **Function:** Users can upload PDF files. The chatbot extracts text and images from the PDF, analyzes them, and stores the information for further querying.
- **Technologies:** 
  - **PyPDF2:** Extracts text from PDFs.
  - **PIL (Python Imaging Library):** Used for image handling.
  - **pytesseract:** Extracts text from images.
  - **faiss:** Indexes and retrieves document embeddings.
  - **OpenAI:** Generates text embeddings and responses based on extracted content.

**Image Analysis**  
- **Function:** Classifies images and extracts text from them.
- **Technologies:** 
  - **Transformers:** Uses `AutoFeatureExtractor` and `AutoModelForImageClassification` from Hugging Face to classify images.

**Question Answering**  
- **Function:** Allows users to ask questions about the extracted content. The chatbot retrieves relevant documents and generates answers based on the context.
- **Technologies:**
  - **OpenAI GPT-3.5 Turbo:** Generates responses based on the context of the query.

**Feedback and Help**  
- **Function:** Provides users with the ability to give feedback and access help instructions.
- **Technology:** Streamlit for UI elements.



https://github.com/user-attachments/assets/e18d9b88-ad41-45a3-ae36-a10fb10daf69



---

**Technologies Used:**
- **Streamlit:** For building the web interface.
- **PyPDF2:** For PDF text extraction.
- **PIL and pytesseract:** For image handling and text extraction.
- **faiss:** For document embedding and retrieval.
- **OpenAI API:** For embeddings and chat completions.
- **Transformers (Hugging Face):** For image classification.

This chatbot integrates multiple AI components to provide an interactive and intelligent analysis tool for scientific literature.

### Note on Credentials

Please note that to fully utilize the capabilities of the Scientific Literature RAG Chatbot, a paid version of the API credentials is required. Free versions may have limited access and can result in errors indicating that you have exhausted your quota for the day. For uninterrupted access, ensure you use a paid API key.
