import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time
import re
import requests
from io import BytesIO
from pathlib import Path

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()
genai.configure(api_key=api_key)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False
if "user_question" not in st.session_state:
    st.session_state.user_question = ""

# Set this to True to use local files, False to use URLs
USE_LOCAL_FILES = True

# Local file paths - Use your actual file paths
LOCAL_PDF_PATHS = [
    "/Users/mostafa_gommed/Desktop/fiss/test.pdf",
    # Add more file paths as needed
]

# PDF URLs - Only used if USE_LOCAL_FILES is False
PDF_URLS = [
    "https://example.com/document1.pdf",
    "https://example.com/document2.pdf",
    # Add more URLs as needed
]

def load_pdf_file(file_path):
    """Load a PDF from a local file path"""
    try:
        if os.path.exists(file_path):
            return open(file_path, 'rb')
        else:
            st.error(f"File not found: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error loading PDF from {file_path}: {str(e)}")
        return None

def download_pdf_from_url(url):
    """Download a PDF from a URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return BytesIO(response.content)
    except Exception as e:
        st.error(f"Error downloading PDF from {url}: {str(e)}")
        return None

def get_pdf_text(pdf_files):
    """Extract text from multiple PDF files (from files or BytesIO objects)"""
    text = ""
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:  # Only add if text was successfully extracted
                    text += page_text + "\n"
        except Exception as e:
            pdf_name = getattr(pdf, 'name', 'Unknown PDF')
            st.error(f"Error processing {pdf_name}: {str(e)}")
    return text

def clean_text(text):
    """Clean and preprocess text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove special characters that might interfere with processing
    text = re.sub(r'[^\w\s.,;:?!()\[\]{}\'""\-----]', '', text)
    return text

def get_text_chunks(text):
    """Split text into manageable chunks with improved parameters"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, source_names):
    """Create and save vector store with metadata"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Add metadata to chunks to track source documents
        texts_with_metadata = []
        for i, chunk in enumerate(text_chunks):
            # Determine which source this chunk came from (simplified approach)
            chunk_index = i % len(source_names) if source_names else 0
            source = source_names[chunk_index] if source_names else "Unknown"
            
            texts_with_metadata.append({
                "text": chunk,
                "metadata": {"source": source, "chunk_id": i}
            })
        
        # Create vector store with texts and metadata
        vector_store = FAISS.from_texts(
            [item["text"] for item in texts_with_metadata],
            embedding=embeddings,
            metadatas=[item["metadata"] for item in texts_with_metadata]
        )
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False

def get_conversational_chain():
    """Create an enhanced conversational chain with a better prompt"""
    prompt_template = """
    أنت مساعد محترف متخصص بالإجابة على الأسئلة باللغة العربية. استخدم المعلومات التالية للإجابة عن السؤال بطريقة مفصلة وواضحة.
    
    السياق:
    {context}
    
    السؤال الحالي:
    {question}
    
    تاريخ المحادثة السابق (إذا وجد):
    {chat_history}
    
    قواعد للإجابة:
    1. قدم إجابات مفصلة وشاملة باللغة العربية الفصحى
    2. إذا كانت المعلومة غير موجودة في السياق، قل "هذه المعلومة غير متوفرة في النص المتاح"
    3. استشهد بالمصدر إذا كان ذلك مناسباً
    4. نظم إجابتك بترتيب منطقي ومتماسك
    5. راعي المحادثة السابقة عند الإجابة على السؤال الحالي
    
    الإجابة:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        top_p=0.95,
        max_tokens=2048
    )

    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question", "chat_history"]
    )
    
    chain = load_qa_chain(
        model, 
        chain_type="stuff", 
        prompt=prompt,
        verbose=True
    )

    return chain

def format_chat_history(chat_history):
    """Format chat history for the prompt"""
    if not chat_history:
        return "لا توجد محادثة سابقة."
    
    history_text = ""
    for exchange in chat_history[-3:]:  # Use only the last 3 exchanges to avoid token limits
        history_text += f"س: {exchange['question']}\n"
        history_text += f"ج: {exchange['answer']}\n\n"
    
    return history_text

def process_question(user_question):
    """Process user questions and retrieve answers"""
    start_time = time.time()
    
    # Show thinking indicator
    with st.spinner("جاري البحث في المستندات..."):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            # Load vector store
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            
            # Search with higher k for better recall
            docs = new_db.similarity_search(user_question, k=4)
            
            # Extract sources for citation
            sources = [doc.metadata.get("source", "Unknown") for doc in docs]
            unique_sources = list(set(sources))
            
            chain = get_conversational_chain()
            
            # Format chat history for context
            chat_history_text = format_chat_history(st.session_state.chat_history)
            
            # Get response
            response = chain(
                {
                    "input_documents": docs, 
                    "question": user_question, 
                    "chat_history": chat_history_text
                },
                return_only_outputs=True
            )
            
            # Process elapsed time
            elapsed_time = time.time() - start_time
            
            # Add to chat history
            st.session_state.chat_history.append({"question": user_question, "answer": response["output_text"], "sources": unique_sources})
            
            return {
                "answer": response["output_text"],
                "sources": unique_sources,
                "elapsed_time": elapsed_time
            }
            
        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة سؤالك: {str(e)}")
            return {
                "answer": "عذراً، حدث خطأ أثناء معالجة سؤالك. يرجى المحاولة مرة أخرى.",
                "sources": [],
                "elapsed_time": 0
            }

def preprocess_documents():
    """Load and process documents from predefined sources"""
    with st.spinner("جاري تحميل ومعالجة المستندات..."):
        # Get PDFs from either local files or URLs
        pdf_files = []
        source_names = []
        
        if USE_LOCAL_FILES:
            # Load PDFs from local files
            for file_path in LOCAL_PDF_PATHS:
                pdf = load_pdf_file(file_path)
                if pdf:
                    pdf_files.append(pdf)
                    # Use just the filename part as the source name
                    source_names.append(os.path.basename(file_path))
        else:
            # Download PDFs from URLs
            for url in PDF_URLS:
                pdf = download_pdf_from_url(url)
                if pdf:
                    pdf_files.append(pdf)
                    source_names.append(url.split('/')[-1])  # Use filename from URL as source name
        
        if not pdf_files:
            st.error("لم نتمكن من تحميل أي ملفات PDF من المصادر المحددة.")
            return False
            
        # Process documents
        raw_text = get_pdf_text(pdf_files)
        
        # Close any open file handles
        for pdf in pdf_files:
            if hasattr(pdf, 'close'):
                pdf.close()
                
        if not raw_text:
            st.error("لم نتمكن من استخراج أي نص من المستندات المحملة.")
            return False
            
        st.session_state.raw_text_length = len(raw_text)
        
        # Clean text
        cleaned_text = clean_text(raw_text)
        
        # Split into chunks
        text_chunks = get_text_chunks(cleaned_text)
        st.session_state.chunk_count = len(text_chunks)
        
        # Create vector store
        success = get_vector_store(text_chunks, source_names)
        
        if success:
            st.session_state.vector_store_ready = True
            st.session_state.processing_done = True
            return True
        else:
            st.error("حدث خطأ أثناء إنشاء قاعدة البيانات الشعاعية.")
            return False

def handle_send():
    """Handle sending the question"""
    if st.session_state.user_input and st.session_state.vector_store_ready:
        # Get the input value
        user_question = st.session_state.user_input
        
        # Set a flag to process this question
        st.session_state.user_question = user_question
        
        # Clear the input field for next run
        st.session_state.user_input = ""

def main():
    st.set_page_config(
        page_title="محادثة الوثائق الذكية",
        page_icon="📚",
        layout="wide"
    )
    
    # Custom CSS for a modern and responsive UI
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    
    * {
        font-family: 'Tajawal', sans-serif;
    }
    
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #2c3e50;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 100%;
        margin: 0 auto;
        padding: 10px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        height: calc(100vh - 300px);
        overflow-y: auto;
        display: flex;
        flex-direction: column-reverse;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        position: relative;
        clear: both;
        overflow: hidden;
        max-width: 80%;
    }
    
    .user-message {
        background-color: #DCF8C6;
        border-right: 5px solid #4CAF50;
        float: left;
        text-align: left;
    }
    
    .bot-message {
        background-color: #F1F0F0;
        border-right: 5px solid #2196F3;
        float: right;
        text-align: right;
    }
    
    /* Input area */
    .input-area {
        display: flex;
        gap: 10px;
        margin-top: 15px;
    }
    
    .stTextInput>div>div>input {
        padding: 15px;
        border-radius: 20px;
        border: 1px solid #ddd;
        font-size: 16px;
    }
    
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 12px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Sources and metadata */
    .source-info {
        font-size: 12px;
        color: #666;
        margin-top: 5px;
        text-align: left;
    }
    
    /* Loading spinner */
    .stSpinner {
        text-align: center;
        margin: 20px auto;
    }
    
    /* Status indicators */
    .status-indicator {
        padding: 8px 15px;
        border-radius: 20px;
        display: inline-block;
        font-size: 14px;
        margin: 5px 0;
    }
    
    .status-ready {
        background-color: #e3f2fd;
        color: #2196F3;
        border: 1px solid #2196F3;
    }
    
    .status-processing {
        background-color: #fff8e1;
        color: #ffa000;
        border: 1px solid #ffa000;
    }
    
    /* RTL Support */
    .rtl-text {
        direction: rtl;
        text-align: right;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .chat-message {
            max-width: 90%;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Page header
    st.markdown("<h1 class='rtl-text'>📚 محادثة الوثائق الذكية باستخدام Gemini</h1>", unsafe_allow_html=True)
    st.markdown("<p class='rtl-text'>اطرح أسئلة حول محتوى المستندات المضمنة واحصل على إجابات ذكية</p>", unsafe_allow_html=True)
    
    # Process documents at startup if not already done
    if not st.session_state.processing_done:
        with st.spinner("جاري تجهيز النظام وتحميل المستندات..."):
            success = preprocess_documents()
            if success:
                st.success("تم تجهيز النظام بنجاح! يمكنك الآن طرح الأسئلة.")
            else:
                st.error("حدث خطأ أثناء تجهيز النظام. يرجى التحقق من مصادر الملفات وإعادة المحاولة.")
                st.session_state.vector_store_ready = False
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Input field for questions
        with st.container():
            user_input = st.text_input("", placeholder="اكتب سؤالك هنا...", key="user_input", on_change=handle_send)
            send_btn = st.button("إرسال", key="send_button", on_click=handle_send)
            
            # Process the pending question if exists
            if st.session_state.user_question and st.session_state.vector_store_ready:
                # Process the question and get response
                result = process_question(st.session_state.user_question)
                # Clear the pending question
                st.session_state.user_question = ""
        
        # Display chat history
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        
        for exchange in st.session_state.chat_history[::-1]:
            # Bot message
            st.markdown(f"""
            <div class='chat-message bot-message rtl-text'>
                <div><strong>الإجابة:</strong></div>
                <div>{exchange["answer"]}</div>
                <div class='source-info'>
                    {f"المصادر: {', '.join(exchange.get('sources', []))}" if exchange.get('sources') and exchange['sources'][0] != "Unknown" else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # User message
            st.markdown(f"""
            <div class='chat-message user-message rtl-text'>
                <div><strong>السؤال:</strong></div>
                <div>{exchange["question"]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # System status and information
        st.markdown("<h3 class='rtl-text'>حالة النظام</h3>", unsafe_allow_html=True)
        
        if st.session_state.vector_store_ready:
            st.markdown("<div class='status-indicator status-ready rtl-text'>النظام جاهز للاستخدام ✓</div>", unsafe_allow_html=True)
            
            if "chunk_count" in st.session_state:
                st.markdown(f"<p class='rtl-text'>عدد المقتطفات النصية: {st.session_state.chunk_count}</p>", unsafe_allow_html=True)
            
            if "raw_text_length" in st.session_state:
                text_size_kb = st.session_state.raw_text_length / 1024
                st.markdown(f"<p class='rtl-text'>حجم النص: {text_size_kb:.1f} كيلوبايت</p>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-indicator status-processing rtl-text'>جاري تجهيز النظام...</div>", unsafe_allow_html=True)
        
        # System information
        st.markdown("<h3 class='rtl-text'>معلومات النظام</h3>", unsafe_allow_html=True)
        st.markdown("<p class='rtl-text'>هذا النظام يستخدم تقنية Gemini من Google لتحليل المستندات والإجابة على الأسئلة باللغة العربية بطريقة ذكية.</p>", unsafe_allow_html=True)
        
        # Usage instructions
        st.markdown("<h3 class='rtl-text'>تعليمات الاستخدام</h3>", unsafe_allow_html=True)
        st.markdown("""
        <ul class='rtl-text'>
            <li>اكتب سؤالك في مربع النص بأعلى الصفحة</li>
            <li>انقر على زر "إرسال" أو اضغط Enter</li>
            <li>انتظر بضع ثوان للحصول على الإجابة</li>
            <li>يمكنك طرح أسئلة متتابعة والنظام سيراعي سياق المحادثة</li>
        </ul>
        """, unsafe_allow_html=True)
        
        # Show source file information if using local files
        if USE_LOCAL_FILES and st.session_state.vector_store_ready:
            st.markdown("<h3 class='rtl-text'>المستندات المُحملة</h3>", unsafe_allow_html=True)
            for path in LOCAL_PDF_PATHS:
                filename = os.path.basename(path)
                st.markdown(f"<p class='rtl-text'>• {filename}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
