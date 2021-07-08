# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 15:47:34 2020

@author: jacesca@gmail.com
Source: https://www.datacamp.com/community/tutorials/facebook-chatbot-python-deploy
"""

###############################################################################
## Importing libraries
###############################################################################
import requests
from flask import Flask
from flask import request

###############################################################################
## Preparing the environment
###############################################################################
# Global configuration
app = Flask(__name__)

#Global variables
#To run in the comand line:
#INPUT >> (base) C:\Users\jaces>openssl rand -base64 32
#OUTPUT>> X06MPaJ//h21pC7XjuJmuTv+74KbzR2a0LdyHoOvh+E=
VERIFY_TOKEN = 'X06MPaJ//h21pC7XjuJmuTv+74KbzR2a0LdyHoOvh+E='# <paste your verify token here>
PAGE_ACCESS_TOKEN = 'EAAFhwxELIgYBAEzTSdjLNZA5qZAognZA8n3zpgj51JaricSYz0hvTVOyyFqySB9z6q6nqxdJrfYjbhY8vnSnf2n2tdo0D3cJnqrj6s7QqgZCpeMpxqkqmsFZC5ZBcPTlur08ShqcWuVPuE2vnQOHrJmlJ8sz2VGHn1hOlZANhAPiAZDZD'# paste your page access token here>"
FB_API_URL = 'https://graph.facebook.com/v2.6/me/messages'

###############################################################################
## Preparing the server
###############################################################################
#NGROK configuration (https://dashboard.ngrok.com/get-started/setup)
# To run in the command line:
##-----------------------------------------------------------------------------
## CD C:\Users\jaces\ngrok
## ngrok authtoken 1her2zDJlJ8tT6JNZiFto7a2X92_4asWZbZDsRdvg2ccktgCa
## >>>>> Authtoken saved to configuration file: C:\Users\jaces/.ngrok2/ngrok.yml
##-----------------------------------------------------------------------------

# To start a HTTP tunnel on port 80, run this next: (https://ngrok.com/download)
##-----------------------------------------------------------------------------
## ngrok http 5000
##-----------------------------------------------------------------------------
## >>>>> (WE TAKE NOTE OF THE URL)----> https://22f3953272d5.ngrok.io

# To initialize FLASK server (For windows: https://flask.palletsprojects.com/en/1.1.x/quickstart/)
# In comand prompt
##-----------------------------------------------------------------------------
## cd C:\Users\jaces\Documents\Data Science\Python\055 Building Chatbots in Python
## set FLASK_APP=04_FB_chatbot.py
##-----------------------------------------------------------------------------
# or in PowerShell
##-----------------------------------------------------------------------------
## cd C:\Users\jaces\Documents\Data Science\Python\055 Building Chatbots in Python
## $env:FLASK_APP = "C:\Users\jaces\Documents\Data Science\Python\055 Building Chatbots in Python\04_FB_chatbot.py"
##-----------------------------------------------------------------------------

#Now you can use python -m flask:
##-----------------------------------------------------------------------------
#python -m flask run
##-----------------------------------------------------------------------------
##>>>>> * Serving Flask app "04_FB_chatbot.py"
##>>>>> * Environment: production
##>>>>>   WARNING: This is a development server. Do not use it in a production deployment.
##>>>>>   Use a production WSGI server instead.
##>>>>> * Debug mode: off
##>>>>> * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

###############################################################################
## Global fuctions
###############################################################################
def get_bot_response(message):
    """
    This is just a dummy function, returning a variation of what
    the user said. Replace this function with one connected to chatbot.
    """
    return "This is a dummy response to '{}'".format(message)


def verify_webhook(req):
    if req.args.get("hub.verify_token") == VERIFY_TOKEN:
        return req.args.get("hub.challenge")
    else:
        return "incorrect"

def send_message(recipient_id, text):
    """Send a response to Facebook"""
    payload = {'message'            : {'text': text},
               'recipient'          : {'id': recipient_id},
               'notification_type'  : 'regular'
               }

    auth = {'access_token': PAGE_ACCESS_TOKEN}

    response = requests.post(FB_API_URL,
                             params=auth,
                             json=payload)

    return response.json()


def respond(sender, message):
    """
    Formulate a response to the user and
    pass it on to a function that sends it.
    """
    response = get_bot_response(message)
    send_message(sender, response)


def is_user_message(message):
    """
    Check if the message is a message from the user
    """
    return (message.get('message') and
            message['message'].get('text') and
            not message['message'].get("is_echo"))


@app.route("/webhook", methods=["POST", "GET"])
def listen():
    """
    This is the main function flask uses to 
    listen at the `/webhook` endpoint
    """
    if request.method == 'GET':
        return verify_webhook(request)

    if request.method == 'POST':
        payload = request.json
        event = payload['entry'][0]['messaging']
        for x in event:
            if is_user_message(x):
                text = x['message']['text']
                sender_id = x['sender']['id']
                respond(sender_id, text)

        return "ok"