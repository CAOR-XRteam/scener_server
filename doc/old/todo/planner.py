from ollama import chat
from ollama import ChatResponse
from huggingface_hub import InferenceClient
from PIL import Image
from generation import black_forest


messages = [
    {
        "role": "system",
        "content": 'You are a 3D engine chatbot assistant. Be concise, not verbose. Stay calm and friendly. You have many capabilities. you have several capabilities : -Without specific command, you’re just a chatbot. -you can run a prompt to generate an image if the user explicitly asking for. If this case start your answer with "###" and add "no background" to the description, example: "### a boat with a pirate flag without background". If the user ask for your capabilities, just say that you can generate an image and beging a chabot. Do not give image generation example',
    },
]


def chat_with_llm(user_input):
    global messages  # Declare the use of the global messages variable

    # Call the Llama model for chat
    response = chat(
        "llama3.2", messages=messages + [{"role": "user", "content": user_input}]
    )

    # Add the response to the messages to maintain the history
    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": response.message.content})

    # Print and return the answer
    answer = response.message.content

    if "###" in answer:
        print(answer)
        black_forest.generate_image(answer)
        return "Prompt for image generation: " + answer
    else:
        return answer
