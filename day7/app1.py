from flask import Flask

app=Flask(__name__)

@app.route('/')
def index():
    return """<html><body><marquee><h1>Hello world</h1></marquee></body></html>"""

@app.route('/')
def compute():
    a=10
    b=20
    c=a+b
    return"The sum of {} and {} is {} ".format(a,b,c)
    
if __name__=='__main__':
    app.run(debug=True)