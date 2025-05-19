"""
This module defines the llama_chat function for interacting with a local Ollama LLM model.
"""
import ollama

# llama
def llama_chat(model_name, prompt_text, b64_image=None):
    """
    Send a chat message to the specified Ollama model.
    """
    try:
        messages_input = {
            'role': 'user',
                    'content': prompt_text,
        }

        if b64_image:
            messages_input['images'] = [b64_image]

        response = ollama.chat(
            model=model_name,
            messages=[messages_input]
        )

        # 取出模型回覆
        result = response['message']['content']
        return result
    except ollama.ResponseError as e:
        print(f"Ollama response error: {e}")
        return None
