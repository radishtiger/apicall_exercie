import os, json, base64
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from openai import OpenAI


"""
key.json

{
    "OpenAI_API_KEY": "YOUR_API_KEY"
}

"""
with open('key.json') as f:
    keys = json.load(f)
    
for key, val in keys.items():
    os.environ[key] = val


def api_call_3p5(messages,  return_onlycontent = True):
        
    client = OpenAI()
    message_content = []

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
        
    messages=messages
    )
    if return_onlycontent:
        return completion.choices[0].message
    else:
        return completion

def api_call_vision_text(human_message, image, system_message = '', ai_message = '', ModelName = "gpt-4-vision-preview", max_token=4096, retun_onlycontent = True):
        
    if type(image) == str:
        image = encode_image(image)
    chat = ChatOpenAI(model=ModelName, max_tokens=max_token)

    msg = chat.invoke(
        [
            AIMessage(
                content=ai_message
            ),
            SystemMessage(
                content = system_message
            ),
            HumanMessage(
                content=[
                    
                    {"type": "text", 
                        "text": human_message},
                    
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image}",
                            "detail": "auto",
                        },
                    },
                ]
            )
        ]
    )
    if retun_onlycontent:
        return msg.content
    else:
        return msg



def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')