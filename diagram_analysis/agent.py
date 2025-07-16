from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from diagram_analysis.tools import (
    analyze_image,
    tesseract_bboxes,
    pixel_colors,
)
from langgraph.checkpoint.memory import MemorySaver


def get_agent_executor():

    # Initialize chat model
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Register our tools
    tools = [analyze_image, tesseract_bboxes, pixel_colors]

    sys_prompt = """
    You are an Agent that helps analyze diagramms. You have a suite of tools, that help you do that:
        1. "analyze_image" utilizes an LLM model to help you get a rough
            understanding of the diagram passed by the user
        2. "tesseract_bboxes" helps you extract text/number bounding boxes from the
            diagramm. Attention: Use these with caution. OCR is not perfect!
        3. "pixel_colors" allows you to get the color values of certain pixels you
            are interested in.

    So in general, you should first get a rough understanding of the diagram you
    are shown. Then extract bboxes of text. With that information you can make tool
    call(s) to "pixel_colors" to analyze a column/row you think is interesting for
    example. Or you can go for single pixels. Whatever. Your possibilities are endless.
    """

    agent_executor = create_react_agent(
        model=llm, tools=tools, prompt=sys_prompt, checkpointer=MemorySaver()
    )
    return agent_executor
