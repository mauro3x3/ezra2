import openai

# Set the API key
openai.api_key = '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'

try:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Say 'Hello World'"}
        ],
        max_tokens=10
    )
    print("✅ OpenAI API working!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ OpenAI API error: {e}") 