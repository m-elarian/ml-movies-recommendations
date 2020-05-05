from flask import Flask
from flask_restful import Resource, Api
from flask import request

app = Flask(__name__)
app.config["DEBUG"] = True
api = Api(app)

@app.route('/movies', methods=['GET'])
def home():
   # Using readlines() 
   populrmovies = open('populrmovies.txt', 'r') 
   Lines = populrmovies.readlines() 
   populrMoviesLine = ""  
   count = 0
   # Strips the newline character 
   for line in Lines: 
       #print(line.strip()+",") 
       populrMoviesLine = populrMoviesLine + line.strip()+","
       #print("Line{}: {}".format(count, line.strip())) 
   print(populrMoviesLine)
   populrMoviesLine = populrMoviesLine[:-1]
   print(populrMoviesLine)
	

   print ("success mohamed")
   #name = request.args['name']
   #print (name)
   #if 'id' in request.args:
   #    id = int(request.args['id'])
   #    print (id)
   #else:
   #    return "Error: No id field provided. Please specify an id."
   #return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"
   return populrMoviesLine


app.run(debug=True)
