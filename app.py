# app.py
import streamlit as st
import pandas as pd
import os
import sys
import io
import matplotlib.pyplot as plt
import seaborn as sns
import json
import base64 # To embed plots if needed

# Assuming ml_pipeline_agents.py is in the same directory
from ml_pipeline_agents import run_ml_pipeline, ask_executive_qa, user_proxy, manager, groupchat # Import necessary agents/functions

# --- Streamlit UI Setup ---
st.set_page_config(page_title="AutoGen ML Pipeline", layout="wide")

st.title("AutoGen Multi-Agent ML Pipeline Assistant")

st.write("""
Upload a CSV file and enter a data science task.
AutoGen agents will collaborate to perform the ML pipeline steps,
and the results will be displayed below.
After the pipeline is complete, you can ask questions about the analysis and results.

**Note:** This requires an OpenAI API key set as an environment variable `OPENAI_API_KEY`.
""")

# --- Session State Management ---
if 'pipeline_started' not in st.session_state:
    st.session_state.pipeline_started = False
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None
if 'pipeline_chat_history' not in st.session_state:
    st.session_state.pipeline_chat_history = []
if 'execution_status' not in st.session_state:
    st.session_state.execution_status = "Idle"
if 'status_placeholder' not in st.session_state:
    st.session_state.status_placeholder = st.empty() # Placeholder for status updates
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = {} # Store results from pipeline steps
if 'qa_chat_history' not in st.session_state:
     st.session_state.qa_chat_history = [] # History for Q&A chat
if 'qa_mode' not in st.session_state:
    st.session_state.qa_mode = False # Flag to indicate if in Q&A mode


# --- File Upload ---
uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.dataframe = df
        st.success("CSV file loaded successfully.")
        st.write("Data Preview:")
        st.dataframe(df.head())

        # Display initial data characteristics and statistics
        st.subheader("Data Characteristics & Statistics")
        st.write("Shape:", df.shape)
        st.write("Column Info:", df.info()) # Info printed to console, need to capture for UI
        st.write("Descriptive Statistics:", df.describe())
        st.write("Missing Values:", df.isnull().sum())


    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        st.session_state.dataframe = None # Reset dataframe on error

# --- ML Task Input ---
ml_task_input = st.text_area("Enter your data science task (e.g., 'Analyze the customer churn data, predict churn, compare models, and explain the best one.'):", height=100, disabled=st.session_state.dataframe is None)



# --- Function to process AutoGen output and display in Streamlit ---
def process_autogen_output(output_string: str):
    # This is a simplified parser. A more robust solution would require
    # agents to format their output in a consistent, machine-readable way (e.g., JSON).

    st.subheader("ML Pipeline Execution Steps")
    st.write("Analyzing agent messages and code execution...")

    # Simple parsing based on agent names and code blocks
    messages = output_string.strip().split('\n\n') # Split by double newline

    current_step = "Starting..."
    step_details = []

    for msg_block in messages:
        lines = msg_block.strip().split('\n')
        if not lines:
            continue

        header = lines[0]
        content = "\n".join(lines[1:])

        # Identify speaker and intent (simplified)
        if "Data_Analyst (to user_proxy)" in header:
            current_step = "Data Analysis (EDA)"
            st.subheader(f"Step: {current_step}")
            st.write(f"**Data Analyst:** {content}")
            # Look for indicators of plots or key stats in content

        elif "Feature_Engineer (to user_proxy)" in header:
            current_step = "Feature Engineering"
            st.subheader(f"Step: {current_step}")
            st.write(f"**Feature Engineer:** {content}")
            # Look for descriptions of new features

        elif "Preprocessor (to user_proxy)" in header:
            current_step = "Data Preprocessing"
            st.subheader(f"Step: {current_step}")
            st.write(f"**Preprocessor:** {content}")
            # Look for descriptions of scaling, encoding, imputation

        elif "Model_Trainer (to user_proxy)" in header:
            current_step = "Model Training & Evaluation"
            st.subheader(f"Step: {current_step}")
            st.write(f"**Model Trainer:** {content}")
            # Look for model names and metrics

        elif "Model_Comparer (to user_proxy)" in header:
            current_step = "Model Comparison"
            st.subheader(f"Step: {current_step}")
            st.write(f"**Model Comparer:** {content}")
            # Look for comparison results and best model

        elif "Model_Explainer (to user_proxy)" in header:
            current_step = "Model Explainability (XAI)"
            st.subheader(f"Step: {current_step}")
            st.write(f"**Model Explainer:** {content}")
            # Look for explanations and plot references

        elif "Executive_QA (to user_proxy)" in header:
             # Q&A part will be handled in the separate chat section
             pass

        elif "user_proxy (to" in header:
            # User proxy messages, often code execution requests or simple replies
            st.write(f"**User Proxy:** {content}")
            # Look for code blocks and execution results

        elif "exitcode" in header or "execution succeeded" in header or "execution failed" in header:
             # Code execution output
             st.write(f"**Code Execution Output:**")
             st.text(content) # Display raw output

        else:
            # Other messages or internal thoughts
            st.text(f"**AutoGen:** {msg_block}") # Uncomment to see all messages

    # --- Display Parsed Results (Example) ---
    # This part would be more sophisticated if agents formatted output
    st.subheader("Summary of ML Pipeline Results")

    # Example: Display a placeholder for EDA results (replace with actual plots/tables extracted from agent output)
    if "Data Analysis (EDA)" in output_string: # Check if EDA step occurred
        st.write("EDA performed. Look for generated plots and statistics in the execution steps above.")
        # In a real app, extract plot file paths or data from agent messages
        # and display plots here using st.image or st.pyplot/st.plotly_chart

    # Example: Display a placeholder for Model Comparison results
    if "Model Comparison" in output_string:
        st.write("Model training and comparison complete.")
        st.write("Model performance metrics: (Extract from agent output)")
        # In a real app, extract the comparison table or summary and display using st.dataframe

    # Example: Display a placeholder for Explainability results
    if "Model Explainability (XAI)" in output_string:
        st.write("Model explainability generated.")
        st.write("Key feature importances: (Extract from agent output or display plot)")
        # In a real app, extract plot file paths or data from agent messages
        # and display explainability plots here.
# --- Start Pipeline Button ---
if st.button("Start ML Pipeline", disabled=st.session_state.dataframe is None or not ml_task_input or st.session_state.pipeline_started):
    if st.session_state.dataframe is not None and ml_task_input:
        st.session_state.pipeline_started = True
        st.session_state.pipeline_chat_history = [] # Clear previous chat
        st.session_state.agent_messages = [] # Clear previous agent messages
        st.session_state.ml_results = {} # Clear previous results
        st.session_state.execution_status = "Starting ML pipeline..."
        st.session_state.status_placeholder.info(st.session_state.execution_status)
        st.session_state.qa_mode = False # Ensure not in Q&A mode

        # --- Run AutoGen Pipeline ---
        # Need to run AutoGen in a way that updates Streamlit.
        # A direct call to run_ml_pipeline will block.
        # A common pattern is to use a generator or capture output.

        # Redirect stdout to capture agent messages and code execution
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        try:
            st.session_state.status_placeholder.info("Running AutoGen agents...")
            # This will block the UI until AutoGen finishes or hits a termination
            # For a production app, consider running AutoGen in a separate thread/process
            # and using a queue to send updates to Streamlit.
            final_history = run_ml_pipeline(ml_task_input, st.session_state.dataframe)
            st.session_state.pipeline_chat_history = final_history # Store full history
            st.session_state.execution_status = "ML pipeline finished."
            st.session_state.status_placeholder.success(st.session_state.execution_status)
            st.session_state.qa_mode = True # Enable Q&A after pipeline

        except Exception as e:
            sys.stdout = old_stdout # Restore stdout before reporting error
            st.session_state.execution_status = f"An error occurred: {e}"
            st.session_state.status_placeholder.error(st.session_state.execution_status)
            st.error(f"An error occurred during the AutoGen conversation: {e}")
            st.session_state.qa_mode = False # Disable Q&A on error

        finally:
            sys.stdout = old_stdout # Ensure stdout is restored

        # Process the captured output and chat history to display steps and results
        process_autogen_output(redirected_output.getvalue())
        st.rerun() # Re-run Streamlit to display results

# --- Q&A Section ---
if st.session_state.qa_mode:
    st.subheader("Ask Questions (Executive Q&A)")
    st.write("The ML pipeline is complete. You can now ask questions about the data, analysis, models, or results.")

    qa_question = st.text_input("Your Question:")

    if st.button("Ask Question"):
        if qa_question:
            st.session_state.qa_chat_history.append({"role": "user", "content": qa_question})

            # Call the Q&A function with the question and relevant context
            # Passing the full pipeline chat history can be a lot;
            # a better approach is to extract key findings and pass those.
            qa_response_history = ask_executive_qa(qa_question, st.session_state.pipeline_chat_history) # Pass pipeline history as context

            # Append the QA agent's response to the QA chat history
            st.session_state.qa_chat_history.extend(qa_response_history)
            st.rerun() # Re-run to display the new message

    # Display the Q&A chat history
    for message in st.session_state.qa_chat_history:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.markdown(content)

# --- Initial Status Display ---
st.session_state.status_placeholder.info(st.session_state.execution_status)