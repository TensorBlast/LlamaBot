import streamlit as st
import replicate
import os
import math

DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being concise. Please ensure that your responses are socially unbiased and positive in nature. Please also make the response as concise as possible. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
MAX_TOKENS = 4096

def llama_v2_prompt(
    messages: list[dict]
):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being concise. Please ensure that your responses are socially unbiased and positive in nature. Please also make the response as concise as possible. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)

def prepare_prompt(messagelist: list[dict], system_prompt: str = None):
    prompt = "\n".join([f"[INST] {message['content']} [/INST]" if message['role']=='User' else message['content'] for message in messagelist])
    if system_prompt:
        prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n[\INST] {prompt}"
    return prompt

def approx_tokens(prompt: str):
    return math.ceil(len(prompt)* 0.4)

def handleSubmit(messagelist : list[dict], system_prompt: str = None):
    prompt = prepare_prompt(messagelist, system_prompt)
    tokens = approx_tokens(prompt)
    if tokens>=MAX_TOKENS:
        if len(messagelist)<3:
            return "Please enter a shorter message"
        messagelist.pop(2)
        return messagelist, prepare_prompt(messagelist, system_prompt)
    return messagelist, prompt


st.title("ChatBot")
def clear_all():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    st.subheader('Model Parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-13B', 'Llama2-70B', 'CodeLlama-34B'], key='selected_model')
    if selected_model == 'Llama2-13B':
        llm = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
        clear_all()
    elif selected_model == 'Llama2-70B':
        llm = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
        clear_all()
    elif selected_model == 'CodeLlama-34B':
        llm = "meta/codellama-34b-instruct:b17fdb44c843000741367ae3d73e2bb710d7428a662238ddebbf4302db2b5422"
        clear_all()
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.75, step=0.01)
    top_k = st.sidebar.slider('top_k', min_value=1, max_value=10000, value=50, step=10)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_new_tokens = st.sidebar.slider('max_new_tokens', min_value=32, max_value=4096, value=2048, step=8)
    st.button('Clear all', on_click=clear_all)



if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({'role': 'User', 'content': prompt})
    with st.chat_message("User"):
        st.markdown(prompt)
    with st.chat_message("Assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # message_history = prepare_prompt(st.session_state.messages, system_prompt=DEFAULT_SYSTEM_PROMPT)
        st.session_state.messages, message_history = handleSubmit(st.session_state.messages, system_prompt=DEFAULT_SYSTEM_PROMPT)

        for output in replicate.run(llm
                               , input={"prompt": message_history, "max_new_tokens": max_new_tokens, "temperature": temperature, "top_k": top_k, "top_p": top_p}):
            full_response += output
            message_placeholder.markdown(full_response+"▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({'role': 'Assistant', 'content': full_response})
