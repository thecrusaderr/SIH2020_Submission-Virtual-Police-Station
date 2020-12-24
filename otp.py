otp=""
def send_message(number):   
    global otp 
    from twilio.rest import Client

    account_sid="AC208fd81769b08cd5c4489033f6726d4b"
    auth_token="19e4aa4bcfde166ff5317e925f4bfa81"

    client=Client(account_sid,auth_token)
    otp=generateOTP()
    number=str(number)
    otp=generateOTP()
    client.messages.create(
        to=number,
        from_="+13143264810",
        body=otp
    )
    return otp

def generateOTP():
    global otp
    import random

    otp=random.randint(100000,999999)
    
    return otp

def verify(userInput):
    global otp
    userInput=str(userInput)
    
    if(userInput==otp):
        return True
    else:
        return False


