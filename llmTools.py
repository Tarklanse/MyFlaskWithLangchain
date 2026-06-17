from langchain_core.tools import tool

detection_model = None


def init_detection_model():
    global detection_model
    # Initialize the detection model here
    pass


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# @tool
def detect_objects(image_b64: str, media_type: str) -> list:
    """Detect objects in an image."""
    global detection_model
    if detection_model is None:
        init_detection_model()
    return [{"object": "example_object", "confidence": 0.9}]
