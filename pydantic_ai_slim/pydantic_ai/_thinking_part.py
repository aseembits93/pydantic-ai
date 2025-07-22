from __future__ import annotations as _annotations

from pydantic_ai.messages import TextPart, ThinkingPart

START_THINK_TAG = '<think>'
END_THINK_TAG = '</think>'


def split_content_into_text_and_thinking(content: str) -> list[ThinkingPart | TextPart]:
    """Split a string into text and thinking parts.

    Some models don't return the thinking part as a separate part, but rather as a tag in the content.
    This function splits the content into text and thinking parts.

    We use the `<think>` tag because that's how Groq uses it in the `raw` format, so instead of using `<Thinking>` or
    something else, we just match the tag to make it easier for other models that don't support the `ThinkingPart`.
    """
    parts: list[ThinkingPart | TextPart] = []
    
    start_index = 0
    while True:
        start = content.find('<think>', start_index)
        if start < 0:
            if start_index < len(content):
                parts.append(TextPart(content=content[start_index:]))
            break
        if start > start_index:
            parts.append(TextPart(content=content[start_index:start]))
        think_start = start + 7  # len('<think>')
        end = content.find('</think>', think_start)
        if end < 0:
            parts.append(TextPart(content=content[think_start:]))
            break
        if end > think_start:
            parts.append(ThinkingPart(content=content[think_start:end]))
        start_index = end + 8  # len('</think>')
    return parts
