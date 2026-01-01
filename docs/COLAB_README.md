# Vox Amelior on Google Colab

This guide explains how to deploy the Vox Amelior (formerly Nemo Server) platform on Google Colab for demonstration and testing.

## Prerequisites

1.  **Google Account**: To access Google Colab.
2.  **Ngrok Account**: To expose the server to the public internet securely (required for webhooks and external access). Sign up at [ngrok.com](https://ngrok.com).
3.  **Hugging Face Token (Optional)**: If you plan to use Gemma or other gated models.

## Setup Instructions

### 1. Upload to Drive (Optional but Recommended)
For persistent storage, it's best to upload this repository to your Google Drive.
1.  Zip the entire `Nemo_Server` folder.
2.  Upload to Google Drive.
3.  In Colab, mount Drive and unzip (the notebook handles this).

Alternatively, you can clone directly from GitHub if the repo is public.

### 2. Configure Secrets (CRITICAL)
Google Colab has a "Secrets" feature (the key icon on the sidebar). You **MUST** Set the following secrets:

| Name | Value | Description |
| :--- | :--- | :--- |
| `NGROK_AUTHTOKEN` | `your_ngrok_token` | Found on your [Ngrok Dashboard](https://dashboard.ngrok.com/get-started/your-authtoken). |
| `HF_TOKEN` | `your_hf_token` | (Optional) For downloading Gemma models. |
| `SESSION_KEY` | `random_string` | A long random string for session security. |
| `USERS_DB_KEY` | `random_string` | A long random string for database encryption. |

**Do NOT hardcode these in the notebook cells.**

### 3. Open the Notebook
1.  Open `setup_colab.ipynb` in Google Colab.
2.  Run the cells in order.

### 4. What the Notebook Does
- **Installs Dependencies**: Installs `ffmpeg`, `node`, `python` packages.
- **Sets up Tunnel**: Uses `pyngrok` to create a public URL (e.g., `https://xxxx.ngrok-free.app`).
- **Starts Services**: Launches the API Gateway and selected backend services in background threads.
- **Displays URL**: Prints the public URL to access the Vox Amelior UI.

## Troubleshooting

- **OOM (Out of Memory)**: If the runtime crashes, try running fewer services. The notebook allows you to comment out services in the startup block.
- **Ngrok Errors**: Ensure your auth token is correct and you haven't exceeded your free tier limits.
