# HealthBench Evaluation Pipeline: Local Medical Models + Gemini Judge

This project benchmarks open-source medical LLMs (like MedGemma, MedFound) on the HealthBench dataset. It uses **Google Gemini (Vertex AI)** as the "Judge" model instead of GPT-4, while hosting the target models locally on NVIDIA GPUs using **vLLM**.

## Prerequisites

* **Hardware:** Linux server (Ubuntu 22.04) with NVIDIA GPUs (Recommended: 2x RTX 6000 or A100s).
* **Accounts:**
    * Google Cloud Platform (GCP) Project with Billing enabled.
    * Hugging Face Account (with Access Token).
* **Python:** Python 3.10+ installed.
* **Conda:** Alternatively, install the yml file using `conda env create -f environment.yml`
---

## 1. Setup Google Cloud (The Judge)

We use the Google Cloud CLI to authenticate so `simple-evals` can use Gemini Pro/Flash via Vertex AI.

### Install Google Cloud SDK
Run the following commands to install the CLI on Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates gnupg curl

# Add Google Cloud public key
curl [https://packages.cloud.google.com/apt/doc/apt-key.gpg](https://packages.cloud.google.com/apt/doc/apt-key.gpg) | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

# Add the CLI repo
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] [https://packages.cloud.google.com/apt](https://packages.cloud.google.com/apt) cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Install
sudo apt-get update && sudo apt-get install google-cloud-cli
```
#### Authenticate
Initialize the SDK and log in with your Google account.
```bash
gcloud init
# Follow the on-screen prompts to select your Project ID.
```
Create the Application Default Credentials (ADC) file for Python:
```bash
gcloud auth application-default login
```
---
## 2. Setup vLLM (The Inference Engine)
We use vLLM to host large models locally and expose them via an OpenAI-compatible API.
#### Install vLLM
```bash
gcloud auth application-default login
```
#### Hugging Face Authentication
Many medical models (like Gemma) are "gated." You must accept the license on the Hugging Face website and export your token.
```bash
export HF_TOKEN="your_hugging_face_read_token"
```
---
## 3. Running Benchmarks
1. Start the Model server (vLLM)
2. Run the Evaluation script
3. Kill the server to free up GPU memoru

### Step A: Start the Model Server
Choose the command for the model you want to test. Run this in a separate terminal or use & to background it.
#### Examples:
#### Google MedGemma 27B:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model google/medgemma-27b-text-it \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --port 8000 \
    --served-model-name local-model
```
some model requires specific custom chat template
#### MedFound Llama 3 8B (Custom Chat Template):
```bash
python -m vllm.entrypoints.openai.api_server \
    --model medicalai/MedFound-Llama3-8B-finetuned \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --port 8000 \
    --served-model-name local-model \
    --chat-template ./llama3_template.jinja
```
### Step B: Run the Evaluation
In your main terminal, configure the environment to point to your local server and your Google Cloud project.
```bash
# 1. Point the Client to Localhost
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"

# 2. Configure the Judge (Gemini)
export GOOGLE_CLOUD_PROJECT="your-project-id"

# 3. Run the Eval
# --n-threads 20 prevents overloading the local GPU
python -m simple-evals.simple_evals \
    --eval healthbench \
    --model local-model \
    --n-threads 20
```
### Step C: Cleanup (Critical)
After the eval finishes, you must kill the vLLM process to release the VRAM before starting the next model.
```bash
pkill -f vllm
```

## 4. Maintenance & Storage
Medical models are massive. If you run out of disk space, use the Hugging Face CLI to delete cached model weights.
```bash
# Install CLI tools
pip install "huggingface_hub[cli]"

# Interactive delete menu
huggingface-cli delete-cache
```
---
### Troubleshooting
* 403 PERMISSION_DENIED (Gemini):

    * Check that `GOOGLE_CLOUD_PROJECT` is set correctly.

    * Ensure Billing is linked to the project in Google Cloud Console.

    * Ensure "Vertex AI API" is enabled.

* 401 Unauthorized (vLLM):

    * You are trying to download a gated model (like MedGemma).

    * Ensure you ran export `HF_TOKEN` and have accepted the terms on the model's Hugging Face page.

* OOM (Out of Memory):

    * Ensure `--tensor-parallel-size 2` is set if using 2 GPUs.

    * If using older GPUs (non-Ada), add `--quantization bitsandbytes` to the vLLM command.
* Other Errors:
  * Please refer to the [Simple Evals](https://github.com/openai/simple-evals) documentation.
## Acknowledgements & Citations


This project makes use of the following open-source resources:
* **Dataset:** [HealthBench](https://arxiv.org/abs/2505.08775) by [OpenAI].
* **Grader Model:** [Gemini 2.5 Pro](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro) by Google (Vertex AI).
* **Candidate Models:** 
  * [Medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it) 
  * [Medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)
  * [MedFound-Llama3-8B-finetuned](https://huggingface.co/medicalai/MedFound-Llama3-8B-finetuned)
  * [ClinicalGPT-base-zh](https://huggingface.co/medicalai/ClinicalGPT-base-zh)
* **Evaluation Logic:** Heavily inspired by OpenAI's [Simple Evals](https://github.com/openai/simple-evals)
* **Inference:** Powered by [vLLM](https://github.com/vllm-project/vllm)
