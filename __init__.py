from .nodes import JoyCaptioning

version_code = [0, 1]
version_str = f"V{version_code[0]}.{version_code[1]}" + (f'.{version_code[2]}' if len(version_code) > 2 else '')
print(f"### Loading: ComfyUI Joycaption({version_str})")

NODE_CLASS_MAPPINGS = {
    "Joycaption": JoyCaptioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Joycaption": "Joycaption",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
