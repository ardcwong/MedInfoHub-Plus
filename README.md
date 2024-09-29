# MedInfoHub Plus

**MedInfoHub Plus** is a healthcare information platform built using Streamlit. The application leverages Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG) to provide context-aware medical information, offering users easy access to reliable and relevant healthcare data. MedInfoHub Plus aims to combat the proliferation of misinformation by providing accurate and understandable health information sourced from trusted organizations.

## Features

- **Context-Aware Medical Information**: MedInfoHub Plus uses NLP and RAG to retrieve and present relevant medical information based on user queries.
- **FDA and MedQuAD Dataset Integration**: The application aggregates data from trusted sources, including the FDA, ensuring that users have access to accurate drug information and guidelines.
- **Patient and Healthcare Provider Modes**: The app offers both layman-friendly summaries for patients and detailed technical information for healthcare providers.
- **Medical Recommendations**: Users can access recommendations for specialist doctors and teleconsultation websites.
- **Dynamic Information Retrieval**: MedInfoHub Plus uses a knowledge base with real-time data retrieval, powered by a vector store (ChromaDB) and embedding model (`text-embedding-ada-002`).
- **Teleconsultation Integration**: The app encourages telemedicine to improve accessibility to healthcare.

## Technology Stack

- **Streamlit**: Front-end framework for the application interface.
- **ChromaDB**: Vector store for efficient similarity search within the knowledge base.
- **OpenAI's ChatGPT**: Utilized for NLP and generating context-aware responses.
- **FDA and MedQuAD Datasets**: Trusted data sources for drugs and healthcare information.


## Usage
To use MedInfoHub Plus:
1. **Search for Medical Information**: Enter a medical query (e.g., symptoms, treatments, or drug information) and the app will return relevant medical data.
2. **Get Patient-Friendly Summaries**: The platform offers easy-to-understand summaries for patients, providing clear and concise information on medical topics.
3. **Healthcare Provider Mode**: For more in-depth and technical details, use the healthcare provider mode to access detailed drug usage guidelines and other professional resources.
4. **Teleconsultation Recommendations**: The app suggests teleconsultation services and nearby clinics based on your needs.

## Limitations
- **Data Coverage**: The platform's dataset may not include new diseases, treatments, or drugs discovered after 2022.
- **API Token Limits**: The use of OpenAI’s API for NLP and RAG is limited by token constraints, which may affect the complexity and length of the responses.
- **Internet Dependency**: A stable internet connection is required to query the OpenAI API and retrieve external data.

## Acknowledgments
Special thanks to the team behind MedInfoHub Plus:
- Austine Wong
- Japhet Pamonag
- Kate Ponce
- Nicole Barrion
- Sam Nicasio

For more details and to try the live app, visit: [MedInfoHub Plus](https://bit.ly/dsfc13-g1s4-mihplus)
