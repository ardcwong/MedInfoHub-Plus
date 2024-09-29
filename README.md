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
