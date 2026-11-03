from pathlib import Path
import sys

import streamlit as st

from agent import make_responder


st.set_page_config(page_title="Empório da Música | Léo", page_icon="🎸")
st.title("Empório da Música")
st.caption("Oi! Sou o Léo, assistente virtual de IA da loja. Como posso ajudar?")

if len(sys.argv) != 3:
    st.error("Usage: streamlit run web.py -- DATA_DIR MANUAL_PDF")
    st.stop()

if "respond" not in st.session_state:
    try:
        st.session_state.respond = make_responder(Path(sys.argv[1]), Path(sys.argv[2]))
    except (KeyError, OSError):
        st.error("Confira os caminhos dos dados e a configuração da OpenRouter.")
        st.stop()
    st.session_state.transcript = []

if st.sidebar.button("Nova conversa"):
    del st.session_state.respond
    st.rerun()

for role, content in st.session_state.transcript:
    with st.chat_message(role):
        st.markdown(content)

if message := st.chat_input("Pergunte sobre instrumentos ou políticas da loja"):
    with st.chat_message("user"):
        st.markdown(message)
    with st.chat_message("assistant"):
        with st.spinner("Léo está respondendo…"):
            answer = st.session_state.respond(message)
        st.markdown(answer)
    st.session_state.transcript.extend([("user", message), ("assistant", answer)])
