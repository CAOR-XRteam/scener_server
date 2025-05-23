from langchain.callbacks.base import BaseCallbackHandler


""" WIP : make a custom tool tracker for functionnal tests """
class ToolTrackerCallback(BaseCallbackHandler):
    def __init__(self):
        self.used_tools = []

    def on_tool_start(self, serialized_tool_input, **kwargs):
        tool_name = serialized_tool_input.get("name")
        if tool_name and tool_name not in self.used_tools:
            self.used_tools.append(tool_name)

"""
# Usage:
tracker = ToolTrackerCallback()
agent_executor.run("Your prompt here", callbacks=[tracker])

print("Used tools:", tracker.used_tools)
"""
