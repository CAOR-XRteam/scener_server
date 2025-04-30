from .llm import chat
import os


def main():
    agent = chat.create()
    chat.run(agent)

if __name__ == "__main__":
    os.system("clear")
    main()
