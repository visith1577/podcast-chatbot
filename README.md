
# Podcast Chatbot

Podcast chatbot for chat based on transcripts.



## Environment Variables

To run this project, you will need to add the following environment variables to your .env file
```bash
QDRANT_CLOUD_API_KEY
OPENAI_API_KEY
GROQ_API_KEY
ZEP_API_KEY
COHERE_API_KEY
```
Where to get API keys\
`QDRANT_CLOUD_API_KEY` from https://qdrant.tech \
`OPENAI_API_KEY` from https://platform.openai.com \
`GROQ_API_KEY` from https://console.groq.com \
`ZEP_API_KEY` from https://www.getzep.com \
`COHERE_API_KEY` from https://dashboard.cohere.com \


## Run Locally

Clone the project

```bash
  git clone https://github.com/visith1577/podcast-chatbot.git
```

Go to the project directory

```bash
  cd podcast-chatbot
```

Install dependencies
if uv package manager is not installed, install it.

```bash
  pip install uv 
```

install dependencies on to virtual environment

```bash
  uv run app.py
```

run data pipeline  \
go to notebooks folder and run `data_injest.ipynb` to populate database and setup user \
ONLY IF .env not shared \

run app 

```bash
  uv run streamlit run app.py
```

run ell studio 

```bash
  ell-studio --storage ./logdir
```

