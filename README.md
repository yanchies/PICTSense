An LLM-powered analytical tool for processing open-ended survey responses.

## Features

- 📊 **Sentiment Analysis**: Automatically score responses on a 1-10 scale
- 🏷️ **Topic Classification**: Smart categorization using embedding technology
- 📈 **Visual Analytics**: Interactive dashboards showing sentiment and topic distributions
- 📝 **Response Summaries**: AI-generated summaries of feedback by topic

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Streamlit
- pandas
- scikit-learn

## Project Structure

```
pictsense/
├── Main.py                 # Main Streamlit application
├── helper_functions/       # Utility functions
├── logics/                # Core analysis functions
└── pages/                 # Additional Streamlit pages
```

## Data Format

Input CSV should contain:
- One response per row
- Required column: 'OER' (Open-Ended Response)

## Important Notes

⚠️ **This is a proof-of-concept prototype**
- Not intended for production use
- LLM outputs may contain inaccuracies