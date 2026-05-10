# Old Lanarkshire

A student-facing web portal designed for college and university deployment.

**Old Lanarkshire** is a fictional educational environment built to provide quick access to college resources. At its core is an integrated AI chatbot named **Roko**, built and deployed entirely on local hardware.

# Key Features
- **Local Deployment:** Built to run on local infrastructure or private cloud solutions.
- **Security First:** No external AI API subscriptions required, minimizing external attack surfaces and reducing cybersecurity concerns.
- **Roko Chatbot:** An integrated AI assistant that can be fed custom information documents.
- **Customizable Knowledge Base:** Currently fed with generic educational data (for the fictional college); in a production environment, it can ingest specific unit specs, course details, and application guidelines.
- **Linux Optimized:** Designed for Ubuntu environments, either standalone or deployed on Hyper-V, or WSL2.

# System Requirements
This project is designed to be run on a machine with the following capabilities:
- **OS:** Ubuntu (native or containerized)
- **Environments:** 
  - Hyper-V
  - WSL2 (Windows Subsystem for Linux)
- **Hardware:** Sufficient resources to run a local LLM and webserver.

# Installation

# Step 1: Clone the Repository
```bash
git clone https://github.com/Atomiek/OldLanarkshire.git
Step 2: Navigate and Prepare
cd OldLanarkshire
sudo chmod +x setup.sh
Step 3: Run Setup
./setup.sh
🏃‍♂️ Usage
Once the setup is complete, start the webserver and select your desired model.

sh run.sh
Note: Follow the prompt to choose the AI model you wish to deploy.

Access the Portal
Once the server is running, navigate to the local address:

http://127.0.0.1:8080