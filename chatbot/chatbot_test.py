import unittest
from unittest.mock import Mock, patch

from chatbot.chatbot import ChatBot, SYSTEM_PROMPT
from rag import llm_client

UNKNOWN_INFO = "I don’t know based on the given information."


class ChatBotTest(unittest.TestCase):
    def test_model_name(self):
        with patch('chatbot.chatbot.llm_client.get_llm_client') as mock_get_llm, \
             patch('chatbot.chatbot.llm_client.generate_response_with_context') as mock_generate:
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm
            mock_generate.return_value = llm_client.MODEL_FILENAME

            response = ChatBot().ask_question_with_context(
                f"Model is using ${llm_client.MODEL_FILENAME}",
                "What model is this chatbot using?",
            )

        self.assertIn(llm_client.MODEL_FILENAME, response)

    def test_model_should_not_hallucinate(self):
        with patch('chatbot.chatbot.llm_client.get_llm_client') as mock_get_llm, \
             patch('chatbot.chatbot.llm_client.generate_response_with_context') as mock_generate:
            mock_get_llm.return_value = Mock()
            mock_generate.return_value = UNKNOWN_INFO

            response = ChatBot().ask_question_with_context(
                f"Model is using ${llm_client.MODEL_FILENAME}",
                "What GPU is the model running on?",
            )

        self.assertIn(UNKNOWN_INFO, response)

    def test_context_lookup_positive(self):
        with patch('chatbot.chatbot.llm_client.get_llm_client') as mock_get_llm, \
             patch('chatbot.chatbot.llm_client.generate_response_with_context') as mock_generate:
            mock_get_llm.return_value = Mock()
            mock_generate.return_value = "blue"

            response = ChatBot().ask_question_with_context("The sky is blue", "What color is the sky?")

        self.assertIn("blue", response)

    def test_context_lookup_negative(self):
        with patch('chatbot.chatbot.llm_client.get_llm_client') as mock_get_llm, \
             patch('chatbot.chatbot.llm_client.generate_response_with_context') as mock_generate:
            mock_get_llm.return_value = Mock()
            mock_generate.return_value = UNKNOWN_INFO

            response = ChatBot().ask_question_with_context("", "What model is this chatbot using?")

        self.assertIn(UNKNOWN_INFO, response)

    def test_lookup_using_rag(self):
        with patch('chatbot.chatbot.llm_client.get_llm_client') as mock_get_llm, \
             patch('chatbot.chatbot.llm_client.generate_response_using_rag') as mock_generate:
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm
            mock_generate.return_value = "mock-rag-response"

            response = ChatBot().ask_question_using_rag("What blood pressure levels are considered elevated?")

        self.assertEqual(response, "mock-rag-response")
        mock_generate.assert_called_once_with(
            mock_llm,
            SYSTEM_PROMPT,
            "What blood pressure levels are considered elevated?",
        )

    def test_lazy_loads_model_on_first_use(self):
        with patch('chatbot.chatbot.llm_client.get_llm_client') as mock_get_llm, \
             patch('chatbot.chatbot.llm_client.generate_response_without_context') as mock_generate:
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm
            mock_generate.return_value = 'mock-response'

            bot = ChatBot()
            self.assertIsNone(bot.llm)
            response = bot.ask_question_without_context('hello')

            self.assertEqual(response, 'mock-response')
            self.assertIs(bot.llm, mock_llm)
            mock_get_llm.assert_called_once()
            mock_generate.assert_called_once_with(mock_llm, SYSTEM_PROMPT, 'hello')

if __name__ == '__main__':
    unittest.main()
