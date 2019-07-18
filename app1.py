from flask import Flask,request,render_template

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('template1.html')

#@app.route('/')
#def compute():
#    a=10
#    b=20
#   c=a+b
#   return"The sum of {} and {} is {} ".format(a,b,c)

@app.route('/display',methods=['POST'])
def display():
    a =request.form.get('name')
    b=request.form.get('pword')
    c=float(a)+float(b)
    #name =request.form.get('name')
    #print(name)
    #return "Hello "+ name
    return"the result is "+str(c)
    
if __name__=='__main__':
    app.run(debug=True , port=5001)