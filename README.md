# WhatsApp Chatbot Experiments — RomeoBot

A Python chatbot project exploring conversational responses from ChatterBot, plus separate scripts for preparing conversation data and fine-tuning a small Transformer language model. The current `chat.py` runs an interactive terminal chatbot; the repository does not itself show a connected WhatsApp messaging API integration.

## Components

- **`chat.py`** — launches RomeoBot in the terminal and stores conversation state in a local SQLite database.
- **`train_bot.py`** — prepares conversational pairs from CSV exports and trains a ChatterBot model.
- **`chat_transformers.py`** — Transformer-based chatbot experiment.
- **`prepare_transformer_data.py`** — data preparation for model training.
- **`train_transformers.py`** — fine-tunes a DistilGPT-2 model using a JSON conversation dataset.
- **`requirements.txt`** — Python dependencies.

## Requirements

- Python installed locally.
- Conversation data files in the formats expected by the relevant training scripts.
- Additional compute may be useful for Transformer training.

## Setup

Clone the repository and create a virtual environment:
```bash
git clone https://github.com/SiyandaNduze/Whatsapp-chatbot.git
cd Whatsapp-chatbot
python -m venv .venv
``

Activate it:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
``

**macOS / Linux**
```bash
source .venv/bin/activate
``

Install dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
``

Because several dependencies are pinned to older versions, installation compatibility may depend on your Python version and operating system.

## Run the terminal chatbot

```bash
python chat.py
``

Type `quit`, `exit`, or `bye` to end the session. The first run creates a local SQLite database for the ChatterBot instance.

## Training data

The training scripts expect local datasets that may not be included in the repository. Check the relevant script before running it:

- CSV training data uses fields such as `sender`, `message`, and `timestamp`.
- Transformer training expects a JSON dataset with `prompt` and `response` fields.

Keep chat exports private, obtain permission before using other people's messages, and anonymise personal information before training. Do not commit private conversations, credentials, or generated datasets.

## Built with

Python · ChatterBot · SQLite · pandas · PyTorch · Hugging Face Transformers · Datasets
