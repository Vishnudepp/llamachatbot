from lib2to3.fixes.fix_input import context

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st



model = OllamaLLM(model="llama3.2")

template= """
    Answer the question below with what yow know. Give clear, brief and concise replies.
    here is the context: {context}
    
    Here is the question: {question}
    
    """
prompt= ChatPromptTemplate.from_template(template)
chain = prompt | model
def handle_conversation(user_input,context):
    context = ""

    result=chain.invoke({"context": context, "question": user_input})

    context += f"\nUser: {user_input}\nAI: {result}"

    return result



st.set_page_config(page_title="Conversational Q&A Bot")
st.title("🤖 Conversational Bot")
st.write("Let's chat! Ask me anything.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
    st.session_state.context = ""  # For storing conversation history

# Input form for user message
with st.form("chat_form"):
    user_input = st.text_input("Type your message here:")
    submitted = st.form_submit_button("Send")

# Handle user input
if submitted and user_input:
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate AI response
    response = handle_conversation(user_input,context)

    # Update context and add AI response
    st.session_state.context += f"\nUser: {user_input}\nAI: {response}"
    st.session_state.messages.append({"role": "assistant", "content": response})



# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.markdown(f"**🤖 AI:** {msg['content']}")
    else:
        st.markdown(f"**🧑 You:** {msg['content']}")