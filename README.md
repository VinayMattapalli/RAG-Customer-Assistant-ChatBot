RAG Chatbot
Project Description
The RAG Chatbot (Retrieval-Augmented Generation Chatbot) is an intelligent FAQ chatbot built using retrieval-augmented generation techniques. It retrieves relevant information from a knowledge base and generates responses to user queries. The chatbot is designed to handle queries efficiently and accurately, making it ideal for FAQs or customer support applications.

Live app: RAG Chatbot

Features
Knowledge Base Query: Retrieves and responds to user queries based on a predefined FAQ knowledge base.
Streamlit Interface: Simple and intuitive UI for interacting with the chatbot.
Scalable: Easily extendable to add more FAQs or integrate with external APIs.
Running Locally
Prerequisites
Python 3.8 or above
Required Python libraries (listed in requirements.txt)
Steps to Run Locally
Clone the Repository:

bash
Copy
Edit
git clone https://github.com/VinayMattapalli/rag-chatbot-vinay.git
cd rag-chatbot-vinay
Install Dependencies: Use pip to install the required Python packages:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit App: Launch the Streamlit app using the following command:

bash
Copy
Edit
streamlit run streamlit_app.py
Access the App: Open your web browser and navigate to http://localhost:8501.

Deployment Details
The chatbot is deployed on Streamlit Cloud and is accessible via the following link: https://rag-chatbot-vinay-5uuy83bx9vvsjbmgfhwkng.streamlit.app/

Future Enhancements
Add support for multi-language queries.
Enhance the UI with additional Streamlit widgets.
Allow integration with external APIs or databases for dynamic FAQs.