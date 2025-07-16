from PIL import Image
import base64
from pydantic import BaseModel, Field
import pytesseract
import numpy as np
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage


# 1. Image analyzer via OpenAI
@tool(
    "analyze_image",
    description="Given an image and a question, returns JSON with label-boxes and column‑x ranges.",
)
def analyze_image(image_path: str, question: str) -> str:
    """Ask GPT to describe what it “sees” and what numbers/axes/columns to look for."""
    # Pass to LLM
    llm = init_chat_model("openai:gpt-4o-mini")
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")
    system_prompt = """You are a diagram analyst.
        Answer the given question. Don't miss out on details in the diagram
        (bytes) that are relevant for understanding the image. Your output is
        given to a parent LLM Node, that has no ability to see the image.
        Be VERY precise, when describing the diagram. Describe every color you see, the rough positioning of all elements, how the elements look (e.g. columns in a column diagram).
        How they are aligned, etc... The parent LLM REALLY needs to understand
        whats going on in the diagram, because it will later search for
        specific data in the diagram, using pixel colors, and it has to get a
        feeling for where things are positioned.
        """
    sys_message = SystemMessage(system_prompt)
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": question,
            },
            {
                "type": "image",
                "source_type": "base64",
                "data": img_data,
                "mime_type": "image/jpeg",
            },
        ],
    }
    response = llm.invoke([sys_message, user_message])

    # llm = ChatOpenAI(model="gpt-4o-mini")
    # This uses OpenAI’s vision-enabled endpoint under the hood.
    # resp = llm.invoke(images=[img_bytes], question=prompt + "\nQuestion: " + question)
    return response.text()


# 2. Tesseract bounding‑box extractor
@tool("tesseract_bboxes", description="Returns all word bounding‑boxes in the image.")
def tesseract_bboxes(image_path: str) -> list[dict]:
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    boxes = []
    for i, txt in enumerate(data["text"]):
        if txt.strip():
            boxes.append(
                {
                    "text": txt,
                    "left": data["left"][i],
                    "top": data["top"][i],
                    "width": data["width"][i],
                    "height": data["height"][i],
                }
            )
    return boxes


# 3. Pixel‑color probe
class PixelInput(BaseModel):
    coords: list[list[int]] = Field(
        ..., description="A list of (x, y) integer pixel coordinates"
    )
    image_path: str = Field(..., description="Path to the image file")


@tool(
    "pixel_colors",
    description="Given a list of (x,y) pixels and an image, returns their RGB color triples.",
    args_schema=PixelInput,
)
def pixel_colors(
    coords: list[tuple[int, int]], image_path: str
) -> list[tuple[int, int, int]]:
    img = np.array(Image.open(image_path))
    h, w = img.shape[:2]
    out = []
    for x, y in coords:
        if 0 <= x < w and 0 <= y < h:
            out.append(tuple(int(c) for c in img[y, x]))
        else:
            out.append(None)
    return out
