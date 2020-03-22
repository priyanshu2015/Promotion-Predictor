#render_template to return html page 
from flask import Flask,render_template,request
app = Flask(__name__)
import pickle

# open a file, where you stored the pickled data
file=open('model.pkl','rb')

clf=pickle.load(file)
file.close()

@app.route('/',methods=["GET","POST"])
def hello_world():
    if request.method == "POST":
        # all the entered values in form get transfered as a dictionary into myDict
        myDict=request.form
        education=int(myDict['education'])
        jobLevel=int(myDict['jobLevel'])
        workLifeBalance=int(myDict['workLifeBalance'])
        performance=int(myDict['performance'])
        worktime=int(myDict['worktime'])
        project=int(myDict['project'])
        # Code for inference
        inputFeatures=[education,jobLevel,workLifeBalance,performance,worktime,project]
        promotionProb=clf.predict_proba([inputFeatures])[0][1]
        print(promotionProb)
        return render_template('show.html',inf=round(promotionProb*100))
    # to return html page saved as index.html
    return render_template('index.html')
    # return 'Hello, World!' + str(promotionProb)
    
if __name__ == "__main__":
    app.run(debug=True)