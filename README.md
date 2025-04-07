# DeepfakeVideoDetection
# 🧠 DeepFake Video Detection

**DeepFake Video Detection** is a web-based application built to detect whether a given video has been manipulated using deepfake techniques. It utilizes a pre-trained machine learning model hosted on a Flask backend. The application is easy to set up and can be run locally or deployed to the cloud.

---

## ⚙️ Environment Setup

Follow the steps below to set up your local development environment.

### 1. Prerequisites

- Make sure [Anaconda](https://www.anaconda.com/products/distribution) is installed on your machine.
- Open **Anaconda Prompt** and navigate to the root directory of this project.

### 2. Create a Virtual Environment

```bash
conda create -n deepfakedetection python=3.10

3. Activate the Environment
bash
Copy
Edit
conda activate deepfakedetection
4. Install Dependencies
Inside the project directory, install the required Python packages:

bash
Copy
Edit
pip install -r requirements.txt
5. Run the Application
Start the Flask app by executing:

bash
Copy
Edit
python app.py
Once the server starts, open your browser and navigate to:

cpp
Copy
Edit
http://127.0.0.1:5000/
🚀 You're Ready to Detect DeepFakes!
The app should now be running in your browser. Upload any video you'd like to analyze, and the model will provide you with a prediction on whether it’s real or fake.

📁 Project Structure
cpp
Copy
Edit
deepfake-video-detection/
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   └── js/
└── model/
    └── <pretrained_model_files>
For queries get in touch with https://www.linkedin.com/in/srinidhi-sn-876368210/
