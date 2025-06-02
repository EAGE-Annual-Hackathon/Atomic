import asyncio

import streamlit as st
from pydantic_ai import Agent
from pydantic_ai._agent_graph import ModelRequestNode
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai.messages import BinaryContent, UserPromptPart
from pydantic_graph import End

st.title("Automic")

# create mcp client
server = MCPServerHTTP(url="http://0.0.0.0:9000/atomic")
# create llm agent with mcp server
agent = Agent(
    "openai:gpt-4.1-mini",
    mcp_servers=[server],
    system_prompt="Use the mcp server to handle seismic data.",
    verbose=True,
    instrument=True,
)


async def agent_iter(prompt: str):
    """Iterate over the intermediary results of the llm agent.

    This function prints the results and all intermediary images to streamlit and debugs all other intermediary results.

    Args:
        prompt: Prompt to give to the model.
    """
    async with agent.iter(user_prompt=prompt) as agent_run:
        async for node in agent_run:
            print(f"{type(node)=}, {node=}")

            if isinstance(node, End):
                response = node.data.output
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            if isinstance(node, ModelRequestNode):
                print(f"{type(node.request)=}, {node.request=}")
                for part in node.request.parts:
                    if isinstance(part, UserPromptPart):
                        content = part.content
                        for content_part in content:
                            if isinstance(content_part, BinaryContent):
                                if content_part.is_image:
                                    st.image(content_part.data)
                                    st.session_state.messages.append({"role": "assistant", "image": content_part.data})


async def run_agent(prompt: str):
    """Run agent with given prompt.

    Args:
        prompt: Prompt to give to the model.
    """
    async with agent.run_mcp_servers():
        await agent_iter(prompt)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages and images from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "content" in message:
            st.markdown(message["content"])
        elif "image" in message:
            st.image(message["image"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # run llm agent
        asyncio.run(run_agent(prompt=prompt))
