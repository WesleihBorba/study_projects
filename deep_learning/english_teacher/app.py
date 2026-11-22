import streamlit as st
from pathlib import Path
import sys
import logging
import base64
import os
from deep_learning.english_teacher.ai_teacher.conversation import ConversationAI
from deep_learning.english_teacher.speech.speech_to_text import TranscribeAudio
from logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_base64_of_file(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


st.set_page_config(
    page_title="English Tutor",
    page_icon="🎧",
    layout="wide"
)

background_image = "ai_teacher\\flaming_horse_head.jpg"
if os.path.exists(background_image):
    img_base64 = get_base64_of_file(background_image)
    st.markdown(
        f"""
          <style>
          .stApp {{
              background-image: url("data:image/jpg;base64,{img_base64}");
              background-attachment: fixed;
              background-size: cover;
              background-position: center;
              color: #000000 !important;
          }}
          
          h1, h2, h3, p, span, label, .stMarkdown {{
              color: #000000 !important;
          }}

          .stChatMessage {{
              background-color: rgba(255, 255, 255, 0.90) !important;
              border-radius: 10px;
              padding: 10px;
              margin-bottom: 10px;
          }}

          .stChatMessage .stMarkdown p {{
              color: #000000 !important;
          }}
          </style>
          """,
        unsafe_allow_html=True
    )
else:
    st.warning("File not found.")


col_left, col_center, col_right = st.columns([2, 4, 2])

with col_left:
    first_image = "ai_teacher\\nvidia.JPG"
    if os.path.exists(first_image):
        st.image(first_image, width='stretch')
    else:
        st.write("Image not found.")

with col_center:
    st.title("Talk to Mary without your wife")
    st.divider()

    if "conversation" not in st.session_state:
        st.session_state.conversation = ConversationAI()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.divider()

    st.subheader("Speak in English")
    st.write("Record your message using the microphone.")

    audio = st.audio_input("Record your message")

    if audio is not None:
        st.audio(audio)

        if st.button("Send audio", type="primary"):
            try:
                with st.spinner("Transcribing your speech..."):
                    transcriber = TranscribeAudio()
                    user_text = transcriber.transcribe_file(audio.getvalue())

                if not user_text or not user_text.strip():
                    st.warning("No speech was detected. Please try again.")
                else:
                    user_text = user_text.strip()

                    with st.chat_message("user"):
                        st.markdown(user_text)

                    st.session_state.messages.append({
                        "role": "user",
                        "content": user_text
                    })

                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            answer = (st.session_state.conversation.ai_return(user_text))

                        if answer:
                            st.markdown(answer)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer
                            })

            except Exception as error:
                logger.exception(f"Error processing audio {error}.")
                st.error("An error occurred while processing your audio. Check the terminal for details.")

with col_right:
    second_image = "ai_teacher\\men_image.jfif"
    if os.path.exists(second_image):
        st.image(second_image, width='stretch')
    else:
        st.write("Image not found.")