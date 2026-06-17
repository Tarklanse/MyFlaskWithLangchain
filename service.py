from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import base64
from pathlib import Path
import os
import llmTools
import inspect
import yaml
import re
from datetime import datetime
import random
from typing import TypedDict, Annotated
from io import BytesIO
from PIL import Image  # 用於圖片格式轉換

config = ""
model = ""
app = ""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def graph_init():
    global app
    workflow = StateGraph(AgentState)
    tools = get_dynamic_tools()
    tool_node = ToolNode(tools)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)


def call_model(state: AgentState):
    global model
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def model_init_opanai():
    global model
    os.environ["OPENAI_API_KEY"] = config["api_keys"]["openai_api_key"]
    llm = ChatOpenAI(model="gpt4o")
    tools = get_dynamic_tools()
    model = llm.bind_tools(tools)


def model_init_api():
    global model
    llm = ChatOpenAI(
        model="model",
        openai_api_base="http://192.168.35.131:5000/",
        openai_api_key="sk-no-key-required",
    )
    tools = get_dynamic_tools()
    model = llm.bind_tools(tools)


def get_dynamic_tools():
    """Dynamically fetch all tools from llmTools module."""
    tools = []
    for name, obj in inspect.getmembers(llmTools):
        if isinstance(obj, BaseTool):
            tools.append(obj)
    return tools


def load_config():
    global config
    with open("config.yaml", "r", encoding="utf-8") as f:
        # 使用 safe_load 避免執行 YAML 檔中可能存在的惡意程式碼
        config = yaml.safe_load(f)
    return config


def model_predict(input_text):
    global app
    config = {"configurable": {"thread_id": f"{generate_serial()}"}}
    messages = []
    sysprompts = {"role": "system", "content": ""}
    userprompts = {"role": "user", "content": input_text}
    messages.append(sysprompts)
    messages.append(userprompts)
    output = app.invoke({"messages": messages}, config=config)
    last_message = output["messages"][-1].content
    if "<think>" in last_message and "</think>" in last_message:
        last_message = remove_think(last_message)
    return last_message


def encode_image(image_path: str) -> tuple[str, str]:
    """將圖片編碼為 base64，並回傳 (base64_data, media_type)"""
    suffix = Path(image_path).suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")
    data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def remove_think(message):
    return re.sub(r"<think>[\s\S]*?</think>", "", str(message))


def model_predict_image(
    input_text: str, image_b64: str | None = None, media_type: str | None = None
):
    global app
    config = {"configurable": {"thread_id": f"{generate_serial()}"}}
    messages = []
    if image_b64 and media_type:
        # 【在此調用轉換函數】
        image_b64, media_type, is_video = convert_webp_to_compatible_image(
            image_b64, media_type
        )

        if is_video:
            return "不支援上傳webp影片"

    sysprompts = {"role": "system", "content": ""}

    if image_b64 and media_type:
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_b64,
                },
            },
            {"type": "text", "text": input_text},
        ]
        userprompts = {"role": "user", "content": content}
    else:
        userprompts = {"role": "user", "content": input_text}

    messages.append(sysprompts)
    messages.append(userprompts)

    output = app.invoke({"messages": messages}, config=config)
    last_message = output["messages"][-1].content
    if "<think>" in last_message and "</think>" in last_message:
        last_message = remove_think(last_message)
    return last_message


def generate_serial():
    # 格式化時間: yyyymmddHHMMSS
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # 加上 4 位隨機數防止同一秒重複 (例如: 202310271430051234)
    random_suffix = f"{random.randint(1000, 9999)}"

    return timestamp + random_suffix


def convert_webp_to_compatible_image(
    image_b64: str, original_media_type: str
) -> tuple[str, str, bool]:
    """
    檢查 base64 圖片是否為 WebP 影片。
    回傳 (新的 base64 字串, 新的 media_type, 是否為影片 is_video)
    """
    try:
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(BytesIO(img_bytes))

        # 檢查是否為 WebP 格式
        if img.format and img.format.upper() == "WEBP":
            # Pillow 的 n_frames > 1 代表是動畫/影片
            is_video = getattr(img, "n_frames", 1) > 1

            if is_video:
                return image_b64, original_media_type, True  # 是影片，原樣回傳
            else:
                # 靜態 WebP 轉為 PNG (llama.cpp 最穩定的格式)
                img = img.convert("RGBA")
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                buffer.seek(0)
                new_img_bytes = buffer.read()
                return (
                    base64.b64encode(new_img_bytes).decode("utf-8"),
                    "image/png",
                    False,
                )

        # 非 WebP 圖片，直接回傳
        return image_b64, original_media_type, False
    except Exception as e:
        print(f"[Warning] 圖片處理失敗: {e}")
        return image_b64, original_media_type, False
