import streamlit as st
from agent import app, CapstoneState
import uuid


st.set_page_config(
    page_title="Physics Study Buddy",
    page_icon=":atom_symbol:",
    layout="centered"
)


@st.cache_resource
def load_agent():
    return app

agent_app = load_agent()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "user_name" not in st.session_state:
    st.session_state.user_name = ""


with st.sidebar:
    st.title(" Physics Study Buddy")
    st.markdown("**Domain:** Physics Education")
    st.markdown("**For:** B.Tech / Engineering Students")
    st.markdown("---")
    st.markdown("** Topics Covered:**")
    st.markdown("""
- Newton's Laws of Motion
- Kinematics & Equations of Motion
- Work, Energy & Power
- Gravitation
- Thermodynamics
- Waves & Oscillations
- Electrostatics
- Current Electricity
- Optics — Ray & Wave
- Modern Physics
    """)
    st.markdown("---")
    st.markdown("** Calculator Tool:** Ask numerical problems and the bot will compute them for you!")
    st.markdown("---")

    if st.button(" New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.user_name = ""
        st.rerun()


st.title(" Physics Study Buddy")
st.caption("Your 24/7 AI assistant for B.Tech Physics — powered by LangGraph + Groq")
st.markdown("---")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("Ask me anything about Physics..."):

    
    with st.chat_message("user"):
        st.markdown(prompt)

    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            result = agent_app.invoke(
                {
                    "question": prompt,
                    "messages": [],
                    "route": "",
                    "retrieved": "",
                    "sources": [],
                    "tool_result": "",
                    "answer": "",
                    "faithfulness": 0.0,
                    "eval_retries": 0,
                    "user_name": st.session_state.user_name
                },
                config=config
            )

            answer = result["answer"]
            route = result["route"]
            faithfulness = result["faithfulness"]

            
            if result.get("user_name"):
                st.session_state.user_name = result["user_name"]

            
            st.markdown(answer)

            
            sources = result.get("sources", [])
            if sources:
                st.caption(f"Sources: {', '.join(sources)}")
            if route == "tool":
                st.caption(" Calculator used")
            if route == "retrieve":
                st.caption(f" Faithfulness score: {faithfulness:.2f}")

    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": answer})