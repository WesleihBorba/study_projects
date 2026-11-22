from openai import OpenAI
import os
import logging
from deep_learning.english_teacher.database.sql_base_history import InsertDatabase
from deep_learning.english_teacher.speech.text_to_speech import TTSPiperLocal
from pathlib import Path

PROMPT_PATH = Path(__file__).resolve().parent / "prompts.txt"
logger = logging.getLogger(__name__)

with PROMPT_PATH.open("r", encoding="utf-8") as txt_file:
    prompt_file = txt_file.read()

API = os.environ.get('API_GPT_LUNA')


class ConversationAI:
    def __init__(self):
        self.api_client = OpenAI(api_key=API)
        self.prompt_text = prompt_file

        self.database = InsertDatabase()
        self.ai_speech = TTSPiperLocal()
        self.ai_message, self.new_message = [None] * 2

    def load_history(self):
        data_history = self.database.getting_data()

        history = []

        for row in data_history:
            role = row[0]
            content = row[1]

            if role in ("user", "assistant"):
                history.append({
                    "role": role,
                    "content": content
                })

        return history

    def ai_return(self, user_text):
        if not user_text or not user_text.strip():
            logger.warning("No speech was transcribed.")
            return

        self.new_message = user_text.strip()

        logger.info("Loading conversation history")
        historic_message = self.load_history()

        logger.info('Save User text in database')
        self.database.insert_history_data(
            user_categoric="user",
            text_information=self.new_message
        )

        logger.info('Return of AI \n')

        payload_messages = [
            {
                "role": "system",
                "content": self.prompt_text
            },
            {
                "role": "system",
                "content": (
                    "The following conversation messages are historical context only. "
                    "They are provided to help you understand the user. "
                    "Do not answer or continue historical questions unless the "
                    "latest user message explicitly refers to them. "
                    "Always prioritize the latest user message."
                )
            },
            *historic_message,
            {
                "role": "user",
                "content": self.new_message
            }
        ]

        full_answer = self.api_client.chat.completions.create(
            model="gpt-6-luna",
            messages=payload_messages,
            max_completion_tokens=600
        )

        self.ai_message = full_answer.choices[0].message.content

        if not self.ai_message:
            logger.warning("The AI returned an empty response.")
            return

        logger.info('Save AI text in database')
        self.database.insert_history_data(
            user_categoric="assistant",
            text_information=self.ai_message
        )

        logger.info('Deleting data in database')
        self.database.deleting_data()

        self.ai_speech.speak(self.ai_message)

        logger.info("AI response: %s", self.ai_message)