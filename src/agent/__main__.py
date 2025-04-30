from .agent import mediator
import os
import sys

# Add the parent directory to sys.path so we can import the 'library' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def main():
    agent = mediator.create()
    mediator.run(agent)

if __name__ == "__main__":
    os.system("clear")
    main()
