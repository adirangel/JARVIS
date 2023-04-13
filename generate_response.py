import openai

def generate_response(api_key, conversation_history, context_info, extra_context=None):
    openai.api_key = api_key

    assistant_marker = "Assistant:"
    user_marker = "User:"

    formatted_history = ""
    for idx, message in enumerate(conversation_history):
        if idx % 2 == 0:
            formatted_history += f"{user_marker} {message}\n"
        else:
            formatted_history += f"{assistant_marker} {message}\n"

    prompt = f"{context_info}\n\n{formatted_history}{assistant_marker} "

    # Include extra_context in the prompt if available
    if extra_context:
        prompt += f"{extra_context}\n"

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = response.choices[0].text.strip()
    return message
