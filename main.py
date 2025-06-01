import asyncio

import streamlit as st
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

st.title("Automic")

# Set OpenAI API key from Streamlit secrets
# client = openai.OpenAI(
#     base_url="https://lukasmosser--ollama-server-ollamaserver-serve.modal.run/v1",
#     api_key="not-needed",  # Ollama doesn't require API keys
# )


model = OpenAIModel(
    "llama3.1:8b",
    provider=OpenAIProvider(
        base_url="https://lukasmosser--ollama-server-ollamaserver-serve.modal.run/v1", api_key="not-needed"
    ),
)
server = MCPServerHTTP(url="http://0.0.0.0:9000/atomic")
agent = Agent(model, mcp_servers=[server])


async def run_agent(prompt: str):
    async with agent.run_mcp_servers():
        result = await agent.run(prompt)
    return result.output


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # stream = client.chat.completions.create(
        #     model=st.session_state["openai_model"],
        #     messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
        #     stream=True,
        # )

        stream = asyncio.run(run_agent(prompt=prompt))

        response = st.write(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
