Creating a `requirements.txt` file in VS Code for a Python project is straightforward. Here's a step-by-step guide:

1. **Open Your Project in VS Code**:
   - Launch VS Code and open the folder containing your Python project.

2. **Open the Integrated Terminal**:
   - You can open the terminal in VS Code by selecting `Terminal` from the top menu and then `New Terminal`, or by using the shortcut ``Ctrl+` `` (backtick).

3. **Activate Your Virtual Environment** (if you have one):
   - If you’re using a virtual environment, make sure it’s activated. For example:
     - On Windows: `venv\Scripts\activate`
     - On macOS/Linux: `source venv/bin/activate`

4. **Generate `requirements.txt`**:
   - Run the following command in the terminal to generate the `requirements.txt` file based on your current environment's installed packages:
     ```bash
     pip3 freeze > requirements.txt
     ```

5. **Verify `requirements.txt`**:
   - Open the `requirements.txt` file in the VS Code editor to verify that it contains the list of packages and their versions.

6. **Edit `requirements.txt`** (if necessary):
   - You can manually edit `requirements.txt` in VS Code if you need to add or remove specific packages or versions.

That’s it! You’ve created and verified your `requirements.txt` file. This file can now be used to install the required packages in other environments by running:
```bash
pip3 install -r requirements.txt
```

# Chat Bot
1. install
```
pip3 install streamlit pypdf2 langchain faiss-cpu langchain_community langchain_openai python-dotenv
```
streamlit: UI for Chatbot app. upload PDF
pypdf2: PDF file reader
langchain:
faiss-cpu: 

2. add code
3. run
```
python3 chatbot.py
streamlit run chatbot.py
```

# Process
PDF Source -> Chunks -> Embeddings(openAI) -> Vector Store(faiss/vectordb/pinedb) -> ranked results