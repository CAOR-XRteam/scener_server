from colorama import init, Fore, Back
import re
from langchain_core.messages import AIMessage

# Initialize colorama
init(autoreset=True)


def print_answer(raw_response):
    # Extract AI message from list
    messages = raw_response.get("messages", [])
    ai_msg = next((m for m in messages if isinstance(m, AIMessage)), None)

    if not ai_msg:
        print(Fore.RED + "No AI message found.")
        return

    content = ai_msg.content

    # Extract <think> and the main response
    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    think = think_match.group(1).strip() if think_match else ''
    answer = content[think_match.end():].strip() if think_match else content

    print()
    print(Fore.YELLOW + "[THINKING]")
    print(think + "\n")
    print(Fore.YELLOW + "[RESPONSE]")
    print(answer + "\n")

if __name__ == "__main__":
    # Simulated raw response from LangChain
    from langchain_core.messages import HumanMessage

    raw_response = "{'messages': [HumanMessage(content='ho', additional_kwargs={}, response_metadata={}, id='dc3eab67-91bb-4df1-b647-e10ad288e300'), AIMessage(content='<think>\nOkay, the user just said 'ho'. That\'s pretty short. I need to figure out what they mean. Maybe it\'s a typo or a mistake. Let me check the available functions to see if any can help.\n\nThe functions available are calculator, date, browsing, and library. Since the user\'s message is just 'ho', there\'s no clear math expression, date request, or search query. The library function is for checking the asset library, but there\'s no indication they need that. \n\nMaybe the user intended to ask something else but didn\'t type it correctly. Without more context, it\'s hard to tell. I should respond by asking them to clarify their question so I can assist better. Let me make sure I\'m not missing any possible interpretations. Nope, 'ho' doesn\'t match any function parameters. So the best move is to prompt for more information.\n</think>\n\nIt seems like your message might be incomplete or contain a typo. Could you please clarify your question or provide more details so I can assist you better?', additional_kwargs={}, response_metadata={'model': 'qwen3:4b', 'created_at': '2025-05-02T10:11:04.809717569Z', 'done': True, 'done_reason': 'stop', 'total_duration': 4666485889, 'load_duration': 15149677, 'prompt_eval_count': 282, 'prompt_eval_duration': 22742573, 'eval_count': 213, 'eval_duration': 4627671154, 'message': Message(role='assistant', content='<think>\nOkay, the user just said 'ho'. That\'s pretty short. I need to figure out what they mean. Maybe it\'s a typo or a mistake. Let me check the available functions to see if any can help.\n\nThe functions available are calculator, date, browsing, and library. Since the user\'s message is just 'ho', there\'s no clear math expression, date request, or search query. The library function is for checking the asset library, but there\'s no indication they need that. \n\nMaybe the user intended to ask something else but didn\'t type it correctly. Without more context, it\'s hard to tell. I should respond by asking them to clarify their question so I can assist better. Let me make sure I\'m not missing any possible interpretations. Nope, 'ho' doesn\'t match any function parameters. So the best move is to prompt for more information.\n</think>\n\nIt seems like your message might be incomplete or contain a typo. Could you please clarify your question or provide more details so I can assist you better?', images=None, tool_calls=None)}, id='run-c3ab3a86-149e-4d8b-83cd-08bb46eb31b2-0', usage_metadata={'input_tokens': 282, 'output_tokens': 213, 'total_tokens': 495})]}"

    format_answer(raw_response)
