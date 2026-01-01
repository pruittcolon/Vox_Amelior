# Vox Amelior on Google Colab

This guide explains how to deploy Vox Amelior on Google Colab for development and testing.

## Prerequisites

1.  **Google Account**: For Colab access
2.  **Tailscale Account**: For private network access (unlimited bandwidth). Sign up at [tailscale.com](https://tailscale.com)
3.  **Hugging Face Token (Optional)**: For Gemma or other gated models

## Setup Instructions

### 1. Create Tailscale Auth Key
1.  Go to [Tailscale Admin Console](https://login.tailscale.com/admin/settings/keys)
2.  Click "Generate auth key"
3.  Enable "Reusable" for convenience
4.  Copy the key

### 2. Configure Colab Secrets
Click the key icon in Colab sidebar and add:

| Name | Value | Description |
| :--- | :--- | :--- |
| `TAILSCALE_AUTHKEY` | `tskey-auth-xxx` | From Tailscale Admin Console |
| `SESSION_KEY` | `random_string` | Long random string for sessions |
| `USERS_DB_KEY` | `random_string` | Long random string for DB encryption |
| `HF_TOKEN` | `your_hf_token` | (Optional) For Gemma models |

### 3. Development Workflow
The notebook pulls from the `colab-dev` branch by default.

**Local development:**
```bash
git checkout colab-dev
# Make changes
git add . && git commit -m "your message"
git push origin colab-dev
```

**In Colab:** Re-run cell 2 to pull latest changes.

**Merge to main when ready:**
```bash
git checkout main
git merge colab-dev
git push origin main
```

### 4. Run the Notebook
1.  Open `setup_colab.ipynb` in Google Colab
2.  Run cells 1-8 in order
3.  Access via Tailscale IP shown in cell 7

## What the Notebook Does
- **Auto-syncs** with `colab-dev` branch on GitHub
- **Caches dependencies** on Google Drive for fast restarts
- **Starts all 8 services** in background threads
- **Connects to Tailscale** for private network access

## Troubleshooting

| Issue | Solution |
| :--- | :--- |
| OOM crashes | Comment out services in cell 6 |
| Tailscale fails | Verify auth key is valid and reusable |
| Services DOWN | Check cell output for import errors |
| Stale code | Re-run cell 2 to pull latest |
