from openai import OpenAI
client = OpenAI()

response = client.responses.retrieve("resp_67fd1791ed5c8191a57a84e2ca5baf9e0686a98c3fdeed86")
print(response)



