# FinSight AI
### AI Financial Assistant

- Provides a natural-language interface for interacting with financial data.
- Allows users to ask questions about their transactions and spending.
- Uses available financial tools and retrieved data to generate grounded responses.

### Retrieval-Augmented Generation (RAG)

- Uses vector-based retrieval to find relevant financial information.
- Provides the foundation for grounding AI responses in the user's own transaction data.
- Reduces reliance on general-purpose knowledge when answering personal finance questions.

### Interactive Dashboard

The Streamlit application provides the following sections:

- Dashboard
- Upload
- Transactions
- Analytics
- Forecast
- Anomalies
- Chat
- Settings

---

## System Architecture

```
                     ┌─────────────────────┐
                     │        User         │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Streamlit Frontend  │
                     │                     │
                     │ Dashboard           │
                     │ Upload              │
                     │ Transactions        │
                     │ Analytics           │
                     │ Forecast            │
                     │ Anomalies           │
                     │ Chat                │
                     │ Settings            │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │     FastAPI API     │
                     └──────────┬──────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
   │    Ingestion   │  │  Transaction   │  │   Analytics    │
   │                │  │     Store      │  │                │
   │ CSV Parser     │  │    SQLite      │  │ Categorizer    │
   │ PDF Parser     │  │                │  │ Anomalies      │
   └───────┬────────┘  └────────────────┘  │ Forecasting    │
           │                              └───────┬────────┘
           │                                      │
           ▼                                      ▼
   ┌────────────────┐                    ┌────────────────┐
   │  Vector Store  │◄───────────────────│    AI Agent    │
   │                │                    │                │
   │ Embeddings     │                    │ Tools          │
   │ Retrieval      │                    │ RAG            │
   └────────────────┘                    └───────┬────────┘
                                                 │
                                                 ▼
                                         ┌────────────────┐
                                         │   AI Response  │
                                         └────────────────┘
```

---

## Technology Stack

### Frontend
- **Streamlit** — Interactive Web Dashboard

### Backend
- **Python**
- **FastAPI**
- **Uvicorn**
- **SQLite**

### Data Processing
- **Pandas**
- **NumPy**
- CSV Processing
- PDF Processing

### Machine Learning
- **Scikit-learn**
- Anomaly Detection
- Transaction Categorization
- Time-Series Forecasting

### AI / NLP
- **Sentence Transformers**
- **LangChain**
- Retrieval-Augmented Generation (RAG)
- Tool-Using AI Agent
- Vector Retrieval
- Embeddings
- Vector Store

### Development & Testing
- Git
- GitHub
- Visual Studio Code
- Kiro
- Pytest

---

## Project Structure

```
finsight-ai/
│
├── data/
│   └── processed/
│
├── docs/
│
├── notebooks/
│
├── scripts/
│
├── src/
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── tools.py
│   │
│   ├── anomaly/
│   │   └── anomaly_detector.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── dependencies.py
│   │   └── vector_store.py
│   │
│   ├── categorization/
│   │   └── categorizer.py
│   │
│   ├── forecasting/
│   │   └── forecaster.py
│   │
│   ├── frontend/
│   │   ├── app.py
│   │   └── views/
│   │       ├── analytics.py
│   │       ├── anomalies.py
│   │       ├── chat.py
│   │       ├── forecast.py
│   │       ├── settings.py
│   │       └── transactions.py
│   │
│   └── ingestion/
│       ├── csv_parser.py
│       ├── pdf_parser.py
│       └── transaction_store.py
│
├── tests/
│   ├── frontend/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_sidebar_bug_condition.py
│   │   └── test_sidebar_preservation.py
│   │
│   └── unit/
│       ├── test_agent.py
│       └── test_vector_store.py
│
├── .env.example
├── .gitignore
├── pytest.ini
├── requirements.txt
├── requirements.lock.txt
└── README.md
```

---

## How FinSight AI Works

1. **User Uploads Statement** — The user uploads a bank or credit-card statement through the Streamlit interface.
2. **Statement Ingestion** — The ingestion layer processes the uploaded CSV or PDF statement.
3. **Data Extraction** — Transaction information such as date, merchant, amount, and category is extracted.
4. **Data Normalization** — Different statement formats are converted into a common transaction structure.
5. **Transaction Storage** — Processed transactions are stored in the SQLite database.
6. **Transaction Categorization** — Transactions are automatically assigned appropriate spending categories.
7. **Financial Analytics** — Historical spending patterns and category-level expenses are analyzed.
8. **Anomaly Detection** — The anomaly detection module identifies transactions that differ significantly from normal spending behavior.
9. **Spending Forecasting** — Historical spending information is used to generate future spending predictions.
10. **Vector Retrieval** — Relevant financial information can be converted into embeddings and retrieved when required.
11. **AI Agent** — The AI agent uses available financial tools and retrieved information to process natural-language financial questions.
12. **AI Response** — The system generates a response based on the user's financial information.

---

## Core Components

### Ingestion Layer
Responsible for reading and processing financial statements.

**Supported formats:** CSV, PDF

**Main components:** `csv_parser.py`, `pdf_parser.py`

### Transaction Store
Responsible for storing and retrieving normalized transactions.

**Storage:** SQLite

**Main component:** `transaction_store.py`

### Categorization Engine
Automatically assigns categories to financial transactions.

**Main component:** `categorizer.py`

### Anomaly Detection
Identifies unusual transactions based on historical spending behavior.

**Main component:** `anomaly_detector.py`

### Forecasting Engine
Generates future spending predictions from historical financial data.

**Main component:** `forecaster.py`

### Vector Store
Provides embedding-based storage and retrieval for financial information.

**Main component:** `vector_store.py`

### AI Agent
Provides the reasoning and tool-using layer for natural-language financial questions.

**Main components:** `agent.py`, `tools.py`

### FastAPI Backend
Provides API endpoints used by the frontend and application services.

**Main component:** `src/api/app.py`

### Streamlit Frontend
Provides the user-facing financial dashboard.

**Main component:** `src/frontend/app.py`

---

## Application Modules

| Module | Description |
|---|---|
| Dashboard | Provides an overview of financial activity |
| Upload | Uploads and processes bank statements |
| Transactions | Displays processed financial transactions |
| Analytics | Analyzes spending patterns and categories |
| Forecast | Predicts future spending |
| Anomalies | Identifies unusual transactions |
| Chat | Provides an AI-powered financial assistant |
| Settings | Provides application configuration options |

---

## Running the Project Locally

### 1. Clone the Repository

```
git clone https://github.com/dhiraj1308/finsight-ai.git
cd finsight-ai
```

### 2. Create a Virtual Environment

Windows PowerShell:

```
python -m venv venv
```

### 3. Activate the Virtual Environment

```
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file based on `.env.example` and add the required configuration and API credentials.

> Do not commit the `.env` file to GitHub.

### 6. Start the FastAPI Backend

```
uvicorn src.api.app:app --reload
```

The backend will run at: `http://127.0.0.1:8000`

FastAPI documentation: `http://127.0.0.1:8000/docs`

### 7. Start the Streamlit Frontend

Open another PowerShell terminal in the project directory and activate the virtual environment:

```
.\venv\Scripts\Activate.ps1
```

Then run:

```
streamlit run src/frontend/app.py
```

The FinSight AI application will open in your browser.

---

## Testing

FinSight AI uses Pytest for automated testing.

### Run All Tests

```
pytest
```

### Run Unit Tests

```
pytest tests/unit
```

### Run Frontend Tests

```
pytest tests/frontend
```

---

## Security Considerations

Financial information is sensitive. FinSight AI is intended primarily as a local development and experimentation project.

Users should:

- Never commit real bank statements to GitHub.
- Never commit `.env` files or API keys.
- Use synthetic or anonymized financial data when testing.
- Keep sensitive financial documents outside the Git repository.
- Ensure generated databases and private data are excluded from version control.

---

## Project Status

FinSight AI is an actively developed project.

The current implementation includes:

- Bank statement ingestion
- CSV and PDF processing
- Transaction storage
- Automatic transaction categorization
- Spending analytics
- Anomaly detection
- Spending forecasting
- Vector-based retrieval
- AI financial assistant
- Streamlit dashboard
- FastAPI backend
- Automated testing
- Single-click sidebar navigation

Development is iterative and is not tied to a fixed weekly schedule.

---

## Future Improvements

Potential future improvements include:

- Improved transaction categorization accuracy
- More advanced financial forecasting
- Enhanced anomaly detection
- Improved RAG retrieval quality
- Additional financial analytics
- Better AI agent reasoning and tool selection
- Authentication and user-specific financial profiles
- Production deployment
- Enhanced security and privacy controls
- Support for additional financial statement formats
- Improved financial visualization
- Personalized financial recommendations

---

## Disclaimer

FinSight AI is an educational and experimental software project.

The insights, forecasts, anomaly detections, and AI-generated responses provided by the application should not be considered professional financial advice.

Users should independently verify important financial information before making financial decisions.
