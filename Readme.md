# Mental Health Chatbot

This project implements a mental health chatbot using FastAPI for the backend and Streamlit for the frontend. The chatbot uses Retrieval-Augmented Generation (RAG) to provide relevant responses based on mental health articles.

## Features

- RAG-based response generation for mental health queries
- Simple classification of mental health issues
- User-friendly Streamlit interface

## Setup and Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/Mental-Health-Chatbot.git

2. Navigate to the project directory:
   ```sh
   cd Mental-Health-Chatbot
4. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

6. Install the required packages:
   ```sh
   pip install -r requirements.txt

## Running the Application

1. Start the FastAPI server:
   ```sh
   uvicorn app.main:app --reload
2. In a new terminal, start the Streamlit app:
   ```sh
   streamlit run app/streamlit_app.py
3. Open your web browser and go to `http://localhost:8501` to interact with the chatbot.

## Docker Support

To run the application using Docker:

1. Build the Docker image:
   ```sh
   docker build -t mental-health-chatbot .
2. Run the Docker container:
   ```sh
   docker run -p 8000:8000 mental-health-chatbot
3. Access the FastAPI documentation at `http://localhost:8000/docs`

Note: The Streamlit app is not included in the Docker setup. To use the full application, run the Streamlit app separately as described in the "Running the Application" section.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

