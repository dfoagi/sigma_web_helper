from google import genai
from openai import OpenAI
import anthropic
import asyncio

from app.services.llm_clients import openai_client, genai_client, anthropic_client


def ask_chatgpt(client: OpenAI, context: str, user_question: str, model: str):

    my_messages = [
        {"role": "system", "content": "Ты помощник по продукту. Отвечай дружелюбно, понятно и по делу, как опытный специалист техподдержки."
        "Используй простой язык, избегай лишних сложностей. Всегда отвечай на языке, на котором задан вопрос."},
        {"role": "user", "content": f"""
        Вот часть руководства, в которой нужно смотреть:

        {context}

        Вопрос пользователя: {user_question}

        Ответь максимально точно, используя только предоставленный текст.
        Если ответ найден, укажи в какой главе руководства пользователя можно посмотреть подробнее. 
        Если ответа точно нет в тексте, напиши: "К сожалению, мне не удалось найти ответ на ваш вопрос. Переформулируйте вопрос либо
        обратитесь к службе технической поддержки."""}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=my_messages
    )

    answer = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens
    response_tokens = response.usage.completion_tokens
    used_model = response.model

    return answer, prompt_tokens, response_tokens, used_model


def ask_anthropic(client: anthropic.Anthropic, context: str, user_question: str, model: str):

    response = client.messages.create(
        model=model,
        max_tokens=1000,
        system="Ты помощник по продукту. Отвечай ясно и по делу, как специалист техподдержки. На языке, котором задали вопрос",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
                            Вот часть руководства, в которой нужно смотреть:

                            {context}

                            Вопрос пользователя: {user_question}

                            Ответь максимально точно, используя только предоставленный текст. Если возможно, то дай пользователю инструкцию
                            с конкретными действиями.
                            Если ответ найден, укажи в какой главе руководства пользователя можно посмотреть подробнее.
                            Если ответа точно нет в тексте, скажи честно, что ответ ты не нашел и попроси переформулировать вопрос либо
                            обратиться к службе технической поддержки.
                            В ответе используй только стандартные символы ASCII или символы, совместимые с Windows-1251. Не используй
                            эмодзи, стрелки, математические символы и другие спецсимволы."""
                    }
                ]
            }
        ]
    )

    answer = response.content[0].text
    prompt_tokens = response.usage.input_tokens
    response_tokens = response.usage.output_tokens
    used_model = response.model

    return answer, prompt_tokens, response_tokens, used_model


def ask_gemini(client: genai.Client, context: str, user_question: str, model: str):

    response = client.models.generate_content(
        model=model,
        config=genai.types.GenerateContentConfig(
            system_instruction="Ты помощник по продукту. Отвечай ясно и по делу, как специалист техподдержки. На языке, котором задали вопрос"),
        contents=f"""
            Вот часть руководства и отдельных ответов, в которых нужно смотреть:

            {context}

            Вопрос пользователя: {user_question}

            Ответь максимально точно на вопрос пользователя, используя только предоставленный текст. Не придумывай расшифровок аббревиатур, если их нет
            в предоставленной части руководства.

            Если пользователь спрашивает, где можно найти руководство пользователя - скажи ему, что он может найти его в главном окне программы, нажав на
            кнопку "Руководство пользователя", либо на странице программы в разделе "Техническая документация и публичная оферта"; так же отправь ему
            ссылку https://3ksigma.ru/wp-content/uploads/2025/02/manual_sigmapb_v7-1.pdf. Лишней информации не отправляй.

            Если ответ на вопрос пользователя найден и в начале фрагмента указан номер главы и её название, укажи в какой главе руководства пользователя
            можно посмотреть подробнее. Если название главы не указано, то просто дай ответ на вопрос. Если есть ссылка на видео,
            в котором есть ответ на вопрос пользователя, поделись с ним этой ссылкой.

            Если ответа точно нет в тексте, напиши: «К сожалению, мне не удалось найти ответ на ваш вопрос. \n\nПереформулируйте вопрос либо обратитесь
            к службе технической поддержки. Задавайте вопрос так, как будто общаетесь с человеком. Чем полнее и корректнее будет вопрос, тем больше
            вероятность, что я смогу вам помочь. \n\nСпасибо, что пользуетесь ботом. Ваш вопрос передан команде, скоро ответ на него появится».

            В ответе используй только стандартные символы ASCII или символы, совместимые с Windows-1251. Не используй
            эмодзи, стрелки, математические символы и другие спецсимволы.
            Для выделения текста можешь использовать следующие теги: <b>bold</b>, <i>italic</i>, <u>underline</u>. Другие теги не
            поддерживаются!

            Перед отправкой еще раз прочитай вопрос и свой ответ. Убедись, что ответ соответствует вопросу.
            """
    )

    answer = response.candidates[0].content.parts[0].text
    prompt_tokens = response.usage_metadata.prompt_token_count
    response_tokens = response.usage_metadata.candidates_token_count
    used_model = response.model_version

    return answer, prompt_tokens, response_tokens, used_model


async def ask_llm(context: str, user_question: str, model: str):
    model = model.lower()

    if model.startswith("gpt") or model == "o4-mini-2025-04-16":
        return await asyncio.to_thread(ask_chatgpt, openai_client, context, user_question, model)

    elif model.startswith("claude"):
        return await asyncio.to_thread(ask_anthropic, anthropic_client, context, user_question, model)

    elif model.startswith("gemini"):
        return await asyncio.to_thread(ask_gemini, genai_client, context, user_question, model)

    raise ValueError(f"Неизвестная модель: {model}")
