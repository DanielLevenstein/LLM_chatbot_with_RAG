from rag import llm_client

SYSTEM_PROMPT = "Answer the following questions using simple straight forward language. "
SYSTEM_PROMPT2 = '''SYSTEM: If the answer is not contained in the provided context, respond with:
"I don’t know based on the given information, do not use information found outside of the system context."'''
SYSTEM_PROMPT3 = '''
You are a strict RAG assistant.
- Use ONLY the information provided in the retrieved context.
- If the context contains the answer, provide it concisely.
- If the context does NOT contain the answer, respond exactly:
  "I don’t know based on the given information."
- Do not use prior knowledge or guess.
'''

class ChatBot:
    def __init__(self):
        self.llm = None

    def _get_llm(self):
        if self.llm is None:
            self.llm = llm_client.get_llm_client()
        return self.llm

    def ask_question_without_context(self, message: str) -> str:
        print(message)
        return llm_client.generate_response_without_context(self._get_llm(), SYSTEM_PROMPT, message)

    def ask_question_using_rag(self, message: str) -> str:
        print(message)
        return llm_client.generate_response_using_rag(self._get_llm(), SYSTEM_PROMPT, message)
    def ask_question_with_context(self, context: str, message: str) -> str:
        print(message)
        return llm_client.generate_response_with_context(self._get_llm(), SYSTEM_PROMPT3, context, message)

    def close_model(self):
        if self.llm is not None:
            self.llm.close()

if __name__ == '__main__':
    chatbot = ChatBot()
    print("Asking question without context")
    response = chatbot.ask_question_without_context("What model is this chatbot using?")
    print(response)
    print("Asking questions with context")
    response2 = chatbot.ask_question_with_context(f"Model is using {llm_client.MODEL_FILENAME}", "What model is this chatbot using?")
    print(response2)
    response3 = chatbot.ask_question_with_context(f"Model is using {llm_client.MODEL_FILENAME}", "What GPU is the model running on?")
    print(response3)
    response4 = chatbot.ask_question_with_context("The sky is blue", "What color is the sky?")
    print(response4)
    print("When context is given model will not use prior knowledge.")
    response5 = chatbot.ask_question_with_context("", "What is the square root of 2?")
    print(response5)
    print("Asking questions using RAG")
    response6 = chatbot.ask_question_using_rag("What are the symptoms of high blood pressure?")
    print(response6)
    response7 = chatbot.ask_question_using_rag("What blood pressure levels become medically dangerous?")
    print(response7)
    response8 = chatbot.ask_question_using_rag("What are the best ways to lower blood pressure?")
    print(response8)
    chatbot.close_model()