import streamlit as st
import sys
import os
import gdown


# --------------------------------------------------
# Function to Download Required Folders into Specific Paths
# --------------------------------------------------
def download_required_folders():
    base_dir = os.path.dirname(__file__)
    # Define folders with their destination paths and Google Drive URLs.
    folders = [
        {
            "name": "chroma_db1",
            "url": "https://drive.google.com/drive/folders/1mZkRw-WKctV6YRrbtriBewRCxwC8nGXD?usp=sharing",
            "destination": os.path.join(base_dir, "rag", "chroma_db1")
        },
        {
            "name": "model1",
            "url": "https://drive.google.com/drive/folders/1MtBuQoEaAgS9WAj1BTLLWBxuVRScLNSJ?usp=sharing",
            "destination": os.path.join(base_dir, "rag", "model1")
        },
        {
            "name": "my_best_xlnet_model",
            "url": "https://drive.google.com/drive/folders/1E_26qBwYyq23t0u_dfPZRUGsvSVCrDXN?usp=sharing",
            "destination": os.path.join(base_dir, "sentiment", "my_best_xlnet_model")
        }
    ]

    st.info("Checking for required folders...")
    progress_bar = st.progress(0)
    total = len(folders)

    for i, folder_info in enumerate(folders):
        dest = folder_info["destination"]
        if not os.path.exists(dest):
            # Ensure the destination parent folder exists.
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            st.write(f"Downloading **{folder_info['name']}** into {os.path.dirname(dest)}...")
            # Download the folder into the specified destination.
            gdown.download_folder(url=folder_info["url"], output=dest, quiet=False)
        progress_bar.progress((i + 1) / total)
    st.success("All required folders are available!")


# --------------------------------------------------
# Download folders on startup if not present
# --------------------------------------------------
download_required_folders()


# --------------------------------------------------
# Utility: Add Folder to Path (for module imports)
# --------------------------------------------------
def add_to_path(folder_name):
    folder_path = os.path.join(os.path.dirname(__file__), folder_name)
    if folder_path not in sys.path:
        sys.path.append(folder_path)
    return folder_path


# --------------------------------------------------
# Load Modules from Their Respective Folders
# --------------------------------------------------
# RAG (Chatbot) module from deployment folder
deployment_dir = add_to_path("rag")
import rag  # from deployment/rag.py

# DCGAN module
gan_dir = add_to_path("gan")
import dcgan  # from gan/dcgan.py

# GLIDE module
glide_dir = add_to_path("glide")
import glide  # from glide/glide.py

# Sentiment Analysis module from sentiment folder
sentiment_dir = add_to_path("sentiment")
import sentiment  # from sentiment/sentiment.py

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose the App Section:",
    ("Chatbot", "DCGAN Image Generator", "GLIDE Image Generator", "Sentiment Analysis")
)


# --------------------------------------------------
# Chatbot (RAG) Section
# --------------------------------------------------
def run_chatbot():
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = rag.create_qa_chain()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("Chatbot with Direct Text Generation")
    st.write("Enter your message below and get an answer based on the stored context.")

    user_input = st.text_input("Your message:", key="chat_input")
    if st.button("Send") and user_input:
        answer, qa_chain = rag.get_answer(user_input, st.session_state.qa_chain)
        st.session_state.qa_chain = qa_chain
        st.session_state.chat_history.append(("User", user_input))
        st.session_state.chat_history.append(("Bot", answer))

    if st.button("Clear Conversation"):
        st.session_state.chat_history = []

    st.markdown("## Conversation History")
    for speaker, message in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {message}")


# --------------------------------------------------
# DCGAN Image Generator Section
# --------------------------------------------------
def run_dcgan():
    st.title("DCGAN Image Generator")
    option = st.radio("Choose Generation Mode:", ("Generate Random Image", "Generate from Uploaded Image"))

    if option == "Generate Random Image":
        if st.button("Generate Random DCGAN Image"):
            generated_img = dcgan.generate_random()
            st.image(generated_img, caption="DCGAN Generated Image", use_column_width=True)
    else:
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            if st.button("Generate DCGAN Image from Uploaded"):
                generated_img = dcgan.generate_from_image(uploaded_file)
                st.image(generated_img, caption="DCGAN Generated Image", use_column_width=True)


# --------------------------------------------------
# GLIDE Image Generator Section
# --------------------------------------------------
def run_glide():
    st.title("GLIDE Image Generator")
    glide_caption = st.text_input("Enter caption for GLIDE image generation:", key="glide_caption")

    if st.button("Generate GLIDE Image") and glide_caption:
        with st.spinner("Generating GLIDE image..."):
            model_path = os.path.join(glide_dir, "glide_diffusion_model.pth")
            generated_glide_img = glide.generate_image_from_caption(glide_caption, model_path, device="cpu")
        st.image(generated_glide_img, caption=f"GLIDE Generated Image for: {glide_caption}", use_column_width=True)


# --------------------------------------------------
# Sentiment Analysis Section
# --------------------------------------------------
def run_sentiment():
    st.title("Sentiment Analysis")
    sentiment_input = st.text_input("Enter text for sentiment analysis:", key="sentiment_input")

    top_n = st.slider("Select number of top contributing tokens:", min_value=1, max_value=20, value=3)

    if st.button("Analyze Sentiment"):
        if sentiment_input:
            predicted_sentiment, top_tokens_df = sentiment.analyze_text_with_shap(sentiment_input, top_n=top_n)
            st.write(f"**Predicted Sentiment:** {predicted_sentiment}")
            st.write("### Top Contributing Tokens")
            st.table(top_tokens_df)
        else:
            st.write("Please enter some text to analyze.")


# --------------------------------------------------
# Main App Execution Based on Sidebar Selection
# --------------------------------------------------
if app_mode == "Chatbot":
    run_chatbot()
elif app_mode == "DCGAN Image Generator":
    run_dcgan()
elif app_mode == "GLIDE Image Generator":
    run_glide()
elif app_mode == "Sentiment Analysis":
    run_sentiment()
