from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from diagram_analysis.agent import get_agent_executor


class ChatInput(BaseModel):
    message: str


if __name__ == "__main__":
    agent_executor = get_agent_executor()
    while True:
        message = input("> ")

        chat_input = ChatInput(message=message)

        result = agent_executor.invoke(
            {"messages": [HumanMessage(content=chat_input.message)]},
            config={
                "configurable": {"thread_id": "abc123"},
                "recursion_limit": 100,
            },
        )

        messages = result.get("messages", [])

        new_messages = []
        for i in range(len(messages)):
            if isinstance(messages[::-1][i], HumanMessage):
                new_messages = messages[-i:]
                break

        final_message = messages[-1].content

        print(final_message)
