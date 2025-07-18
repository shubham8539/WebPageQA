import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader

# ------------------ Load Environment ------------------
load_dotenv()
st.secrets["OPENAI_API_KEY"]

# ------------------ LLM Setup ------------------
model = ChatOpenAI()
parser = StrOutputParser()

# Prompt Template
prompt = PromptTemplate(
    template='Answer the following question:\n\n{question}\n\nbased on the webpage content:\n\n{text}',
    input_variables=['question', 'text']
)

# LangChain pipeline
chain = prompt | model | parser

# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="Web Page QA", page_icon="🌐", layout="centered")

# ------------------ Custom Styling ------------------
st.markdown("""
    <style>
    /* Background */
    body {
        background: linear-gradient(to right, #e0ecf6, #f7f9fc);
    }

    /* Input fields */
    .stTextInput>div>div>input, .stTextArea>div>textarea {
        font-size: 16px;
        border-radius: 8px;
        padding: 10px;
    }

    /* Fancy Button */
    .stButton>button {
        background: linear-gradient(90deg, #4b6cb7, #182848);
        color: white;
        font-weight: bold;
        padding: 0.6em 1.2em;
        border: none;
        border-radius: 10px;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #43cea2, #185a9d);
        transform: scale(1.03);
    }

    /* Card-style Answer box */
    .answer-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        margin-top: 1.5rem;
        font-family: monospace;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #222;
    }

    /* Title styling */
    h1 {
        color: #182848;
        text-align: center;
        font-size: 2.4rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ App Title ------------------
st.markdown("<h1>🌐 Smart Web Page Q&A</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size: 1.1rem;'>Ask intelligent questions on any public webpage using AI.</p>", unsafe_allow_html=True)

# ------------------ User Input ------------------
url = st.text_input("🔗 Web Page URL", placeholder="https://example.com")
question = st.text_area("❓ Your Question", placeholder="What is the summary of this page?")

# ------------------ Process Request ------------------
if st.button("✨ Get Answer"):
    if not url or not question:
        st.warning("⚠️ Please enter both a URL and your question.")
    else:
        try:
            with st.spinner("🔄 Loading and analyzing..."):
                # Load content from URL
                loader = WebBaseLoader(url)
                docs = loader.load()

                # Extract text
                page_text = docs[0].page_content

                # Run LLM
                response = chain.invoke({'question': question, 'text': page_text})

            # Show result in styled card
            st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
            st.markdown("✅ **Answer**")
            st.markdown(response)
            st.markdown("</div>", unsafe_allow_html=True)

            

        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")
