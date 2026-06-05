import unittest
from unittest.mock import Mock, patch

from chatbot.chatbot import ChatBot, SYSTEM_PROMPT
from rag import llm_client

UNKNOWN_INFO = "I don’t know based on the given information."


class ChatBotTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.chatbot = ChatBot()

    def test_model_name(self):
        response = self.chatbot.ask_question_with_context(f"Model is using ${llm_client.MODEL_FILENAME}", "What model is this chatbot using?")
        # We need a method looking for a substring within a larger one. self.assertIn doesn't do that.
        self.assertTrue(llm_client.MODEL_FILENAME in response, f"Chatbot should know what model it's using since it's included in the context: {response}")

    def test_model_should_not_hallucinate(self):
        response = self.chatbot.ask_question_with_context(f"Model is using ${llm_client.MODEL_FILENAME}", "What GPU is the model running on?")
        self.assertTrue(UNKNOWN_INFO in response, f"Chatbot should not hallucinate information it doesn't know: {response}")

    def test_context_lookup_positive(self):
        response = self.chatbot.ask_question_with_context("The sky is blue", "What color is the sky?")
        self.assertTrue("blue" in response, f"The color of the sky is included in the context {response}")

    def test_context_lookup_negative(self):
        response = self.chatbot.ask_question_with_context(f"", "What model is this chatbot using?")
        self.assertTrue(UNKNOWN_INFO in response, f"The model name isn't included in the context: {response}")

    def test_lookup_using_rag(self):
        response = self.chatbot.ask_question_using_rag( "What blood pressure levels are considered elevated?")
        print(response)

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
