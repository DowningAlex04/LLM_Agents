import os 
from uuid import uuid4
from pprint import pprint
import requests
import rich
from rich.markdown import Markdown
from langchain.agents import create_agent
from langchain.messages import HumanMessage 
from langgraph.checkpoint.memory import InMemorySaver 

from plant_details_rag import plant_rag

import logging
logging.getLogger("Langchain_google_vertixai.functions_utils").addFilter(
    lambda record: "'additionalProperties' is not supported in schema" not in record.getMessage()
)

def get_order_status(order_number: int ) -> dict:
    """ Fetch the order details and status for the given order_number
    Returns a dictionary with the products ordered, the price and quantity of each, order date, and order status. 
    If the order has been delivered the dictionary will include the delivery date
    If there's an error getting the order details, a dictionary with one key "error" and an error message will be returned. 
    """

    # The part above is a docstrung or documentation string and is used by humans to understand
    # What a function does, and also used by the agent to understand what it can use this function for. 
    
    print(f'Calling Get Order Status for Order{order_number}')

    try:
        ORDER_STATUS_KEY = os.environ.get('ORDER_STATUS_KEY')
        url = f'https://mock-order-status.uc.r.appspot.com/orders/status/{order_number}?API_KEY={ORDER_STATUS_KEY}'
        response = requests.get(url)
        if response.status_code == 403:
            return {'error': 'Missing or incorrect API key'}
        elif response.status_code != 200:
            return {'error': 'Error calling order status API'}
        else:
            return response.json()
    except:
        return{'error': 'Error connecting to API'}


output = get_order_status(1234)
print(output)

def search_plants(query: dict) -> list:
    """
    Provide a dictionary of query parameters, min_price, max_price, care_level
    care can be easy, medium or difficult. can have multiple care levels, seprated by ;
    Return a JSON list of matching plants 
    """
    print(f'Calling Search  Plants with query {query}')

    url = 'https://strong-province-113523.appspot.com/search'
    try:
        response = requests.get(url,query)
        if response.status_code != 200:
            return 'Error calling plant search API'
        else: return response.json()
    except:
        return 'Error calling plant search API'
    
query ={'min_price':20,'max_price': 30, 'care_level': 'difficult'}
results = search_plants(query)
print(results)

agent = create_agent(
    model='gemini-2.5-flash',
    tools=[get_order_status, search_plants, plant_rag],
    system_prompt=""" You are a friendly helpful assistant for houseplant store.
    if the user asks about other types of plants, or anything that isnt plant-related, dont answer
    but remind them what you can do. 
    Dont include any technical information in the response.
    """,
    checkpointer= InMemorySaver()
)

thread_id = uuid4()
print(thread_id)
config = {'configgurable': {'thread_id': thread_id}}

print('Welcome to the houseplant store! How can I help?')
while True:

    user_message = input('> ')
    if not user_message: # empty string
        print('Thanks for chatting today, goodbye!')
        break
    human_message = HumanMessage(user_message)
    response = agent.invoke({'messages': [human_message]}, config=config)


    messages = response['messages']
    for message in messages:
        message.pretty_print()
    ai_message = messages[-1] # -1 is the last item in a list
    ai_message_text = ai_message.content
    # rich.print(Markdown(ai_message_text))
