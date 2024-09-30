import streamlit as st
from agents.entry import chatbot_entry
from agents.memory_db import get_memory, delete_session, add_memory, create_session, summarize_conversation, SESSION_ID
from agents.podcast_agent import execute_rag_response
from ell import Message


st.set_page_config(
    page_title="Assesment",
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title("podcast chat")

if "session_active" not in st.session_state:
    st.session_state.session_active = False

def handle_create_session():
    create_session(user_id="user_1", session_id=SESSION_ID)
    st.session_state.session_active = True
    st.session_state.history = []
    st.session_state.messages = [
        Message(role="assistant", content="Hello! I'm a podcast host. Ask me anything about the podcast.")
    ]

def handle_delete_session():
    delete_session(SESSION_ID)
    st.session_state.session_active = False
    st.session_state.history = []
    st.session_state.messages = []

with st.expander("session"):
    st.button("Clear session", on_click=handle_delete_session)
    st.button("create session", on_click=handle_create_session)



if st.session_state.session_active:
    async def main():

        for message in st.session_state.messages:
            with st.chat_message(message.role):
                st.markdown(message.content[-1].text)
        
        if prompt := st.chat_input(placeholder="Ask me anything about the podcast"):
            st.session_state.messages.append(Message(role="user", content=prompt))
            st.session_state.history.append(Message(role="user", content=prompt))

            with st.chat_message("user"):
                st.markdown(prompt)
        
        try:
            facts = get_memory(session_id=SESSION_ID)
            fact_content = ("\n ").join([fact for fact in facts.facts])
        except Exception as e:
            fact_content = "No facts available"

        if len(st.session_state.messages) > 20:
            combined_history = st.session_state.history[:-20] + st.session_state.messages[-20:]
        else:
            combined_history = st.session_state.messages

        if st.session_state.messages[-1].role != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chatbot_entry(
                        query=prompt,
                        history=combined_history,
                        facts=fact_content,
                    )
                    if str(response.content[-1].parsed.use_rag).lower() == "true":
                        rag_response = execute_rag_response(
                            query=prompt,
                            user_intent=response.content[-1].parsed.user_intent,
                            output_emotion=response.content[-1].parsed.output_emotion,
                            history=fact_content,
                        )
            if str(response.content[-1].parsed.use_rag).lower() != "true":
                st.markdown(response.content[-1].parsed.answer)
                add_memory(
                    session_id=SESSION_ID,
                    user_content=str(prompt),
                    assistent_content=str(response.content[-1].parsed.answer),
                )
                st.session_state.messages.append(Message(role="assistant", content=response.content[-1].parsed.answer))
            else:
                st.markdown(rag_response)
                add_memory(
                    session_id=SESSION_ID,
                    user_content=str(prompt),
                    assistent_content=str(rag_response),
                )
                st.session_state.messages.append(Message(role="assistant", content=rag_response))
            assistant_message = st.session_state.messages[-1].content[-1].text
            summary = summarize_conversation(conversation=assistant_message)
            st.session_state.history.append(Message(role="assistant", content=summary))

else:
    async def main():
        st.warning("Please create a session to start chatting")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
