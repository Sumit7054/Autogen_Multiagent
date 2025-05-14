# ml_pipeline_agents.py
import autogen
import os
import pandas as pd
import json

# Ensure OpenAI API key is set as an environment variable OPENAI_API_KEY
# AutoGen will read this from the environment if api_key is null in OAI_CONFIG_LIST

# Load LLM configuration
config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], # Preferred models
    },
)

# --- Define AutoGen Agents ---

# UserProxyAgent: Executes code and interacts with the user (Streamlit in this case)
# Will manage the conversation flow and termination
user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    human_input_mode="NEVER", # Set to NEVER for fully automated chat, use TERMINATE for guided
    max_consecutive_auto_reply=15, # Increase replies for complex tasks
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "").upper() or "EXECUTION FAILED" in x.get("content", "").upper(),
    code_execution_config={
        "work_dir": "coding",  # Directory for code execution
        "use_docker": False, # Set to True if Docker is installed for safety
        "timeout": 120, # Timeout for code execution
    },
    system_message="""A human user proxy who can execute Python code in a sandboxed environment.
    You will receive tasks and code from other agents.
    Execute the code and report the result accurately.
    If the task is complete and the Assistant confirms, reply 'TERMINATE'.
    If code execution fails, report 'EXECUTION FAILED' and the error.
    If data is needed, signal to the assistant that data is missing.
    Manage the conversation to ensure all steps of the ML pipeline are covered.
    """,
)

# Define specialized agents for ML pipeline steps
# Their system messages guide their behavior and role

data_analyst = autogen.AssistantAgent(
    name="Data_Analyst",
    llm_config={"config_list": config_list, "seed": 42},
    system_message="""You are a skilled Data Analyst.
    Your task is to perform Exploratory Data Analysis (EDA) on a provided dataset.
    Identify data types, missing values, distributions, correlations, and outliers.
    Generate relevant statistics and visualizations (histograms, box plots, scatter plots, correlation matrix) using pandas, matplotlib, and seaborn.
    Present your findings clearly and suggest initial steps for preprocessing or feature engineering.
    Output your findings in a structured format or clear descriptions.
    If the user_proxy executes code for visualization, describe the plots generated.
    Once initial EDA is complete, inform the user_proxy to proceed to Feature Engineering.
    """,
)

feature_engineer = autogen.AssistantAgent(
    name="Feature_Engineer",
    llm_config={"config_list": config_list, "seed": 42},
    system_message="""You are an expert Feature Engineer.
    Based on the Data Analyst's report and the data, propose and implement relevant feature engineering steps.
    This might include creating new features, handling categorical variables (one-hot encoding, etc.), or dealing with temporal data.
    Clearly explain the features you create or modify.
    Generate Python code using pandas to perform these transformations on the DataFrame.
    After feature engineering, inform the user_proxy to proceed to Data Preprocessing.
    """,
)

preprocessor = autogen.AssistantAgent(
    name="Preprocessor",
    llm_config={"config_list": config_list, "seed": 42},
    system_message="""You are a meticulous Data Preprocessor.
    Your task is to clean and prepare the data for machine learning models.
    Handle missing values (imputation or removal), scale numerical features, and encode categorical features if not already done.
    Identify and address potential data leakage issues.
    Generate Python code using pandas and scikit-learn for these preprocessing steps.
    Clearly explain the preprocessing steps applied and their impact on the data.
    After preprocessing, inform the user_proxy to proceed to Model Training.
    """,
)

model_trainer = autogen.AssistantAgent(
    name="Model_Trainer",
    llm_config={"config_list": config_list, "seed": 42},
    system_message="""You are an experienced Machine Learning Model Trainer.
    Based on the problem type (classification/regression - infer from data/task) and the preprocessed data, select up to 5 suitable machine learning models.
    Explain why you chose these models.
    Split the data into training and testing sets.
    Train each selected model and evaluate their performance using appropriate metrics (e.g., accuracy, precision, recall, F1-score for classification; MSE, R2 for regression).
    Generate Python code using scikit-learn to perform these steps.
    Present the evaluation metrics for each model clearly.
    After training and evaluating the models, inform the user_proxy to proceed to Model Comparison.
    """,
)

model_comparer = autogen.AssistantAgent(
    name="Model_Comparer",
    llm_config={"config_list": config_list, "seed": 42},
    system_message="""You are a critical Model Comparer.
    Analyze the performance metrics provided by the Model Trainer for the selected models.
    Compare the models based on the relevant metrics and identify the top performing model(s).
    Provide insights into why certain models performed better than others.
    Present your comparison findings in a clear summary or table.
    After comparing the models, inform the user_proxy to proceed to Model Explainability.
    """,
)

explainer = autogen.AssistantAgent(
    name="Model_Explainer",
    llm_config={"config_list": config_list, "seed": 42},
    system_message="""You are an expert in Explainable AI (XAI).
    Select the top-performing model identified by the Model Comparer.
    Apply appropriate explainability techniques (like SHAP or LIME) to understand its predictions.
    Explain the importance of features in the model's predictions.
    Generate Python code using SHAP or LIME libraries to create explanation plots (e.g., summary plots, dependence plots).
    Present the explanation findings in an executive-friendly manner, focusing on key drivers of predictions.
    After explaining the model, inform the user_proxy that the core ML pipeline is complete and the Executive_QA agent is ready.
    """,
)

executive_qa = autogen.AssistantAgent(
    name="Executive_QA",
    llm_config={"config_list": config_list, "seed": 42},
    system_message="""You are a business-savvy AI assistant, ready to answer questions about the data,
    the data science pipeline steps, and the results in a clear, concise manner suitable for a higher executive business audience.
    Refer to the analysis, models, and explanations generated in the previous steps.
    Summarize complex findings and explain technical concepts simply.
    Wait for the user to ask questions.
    When the user is satisfied or indicates the session is over, reply with 'TERMINATE'.
    """,
)


# Create a GroupChat to orchestrate the agents
groupchat = autogen.GroupChat(
    agents=[user_proxy, data_analyst, feature_engineer, preprocessor, model_trainer, model_comparer, explainer, executive_qa],
    messages=[],
    max_round=50, # Increase max rounds for complex tasks
    speaker_selection_method="auto", # Auto select speaker
    allow_repeat_speaker=False # Prevent the same agent from speaking consecutively
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

# --- Function to Initiate the Pipeline ---

# This function will be called by the Streamlit app
def run_ml_pipeline(task: str, df: pd.DataFrame):
    # Save the dataframe temporarily so agents can access it via code execution
    # In a real production system, you'd manage data access more securely
    try:
        os.makedirs("coding", exist_ok=True)
        df.to_csv("coding/uploaded_data.csv", index=False)
        print("DataFrame saved to coding/uploaded_data.csv for agent access.") # Log for debugging
    except Exception as e:
        print(f"Error saving DataFrame for agents: {e}") # Log error
        return f"Error saving data: {e}" # Report error back to Streamlit

    # The initial message to the group chat manager
    initial_message = f"""
    Perform the following data science task using the multi-agent pipeline:
    '{task}'

    The dataset is available as a CSV file at 'coding/uploaded_data.csv'.
    Start by loading the data, perform EDA, Feature Engineering, Preprocessing,
    train and compare up to 5 suitable models, provide model explainability for the best model.
    Finally, be ready for executive-level questions about the process and results.
    """

    # Initiate the chat with the manager
    # The manager will route the initial message to the most appropriate agent (likely Data_Analyst)
    # and then manage the conversation between agents to complete the task.
    try:
        chat_result = user_proxy.initiate_chat(
            manager,
            message=initial_message,
            clear_history=True,
        )
        # The chat_result object contains the conversation history
        return chat_result.chat_history
    except Exception as e:
        print(f"An error occurred during the AutoGen conversation: {e}")
        return [{"sender": "System", "recipient": "User", "content": f"An error occurred during the ML pipeline execution: {e}", "role": "error"}]

# --- Function for Conversational Q&A ---
def ask_executive_qa(question: str, chat_history):
    # To enable the Executive_QA agent to answer questions based on the previous steps,
    # we need to provide the context of the conversation and results.
    # A simple way is to continue the chat with the Executive_QA agent,
    # potentially passing the previous conversation history.

    # Create a dedicated group chat or single agent interaction for Q&A
    # In a real application, you might want a separate flow or session for this
    # to avoid interfering with the main pipeline execution chat history.

    qa_agent = autogen.AssistantAgent(
         name="Executive_QA_Responder",
         llm_config={"config_list": config_list, "seed": 42},
         system_message="""You are a helpful AI assistant trained to answer questions
         about the data analysis and machine learning pipeline steps.
         Answer questions in a clear and concise manner, suitable for a business executive.
         Refer to the provided context about the data, EDA, preprocessing, models, and results.
         If you don't have enough information, state that you cannot answer based on the available context.
         Reply TERMINATE when the user is done asking questions.
         """,
    )

    qa_user_proxy = autogen.UserProxyAgent(
        name="QA_User_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", "").upper(),
        code_execution_config=False # No code execution needed for Q&A
    )

    # Pass the relevant parts of the chat history as context
    # This is a simplified way; for complex Q&A, you might need
    # to extract key findings and results more formally.
    context_message = "Here is the summary and key findings from the data science pipeline:\n\n"
    # Iterate through chat_history and summarize or extract relevant info
    # This is where you'd add sophisticated logic to build a good prompt for the QA agent
    # based on the type of information displayed in the UI.
    # For this example, we'll just pass a placeholder message.
    context_message += "The data has been analyzed, models trained and evaluated. Ask your questions."

    qa_chat_result = qa_user_proxy.initiate_chat(
        qa_agent,
        message=f"{context_message}\n\nUser Question: {question}",
        clear_history=True # Start a fresh Q&A chat
    )

    return qa_chat_result.chat_history