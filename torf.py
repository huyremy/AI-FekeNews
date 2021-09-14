import pickle
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.filterwarnings('ignore')

var = input("Enter the news: ")
print("You entered: " + str(var))
def detecting(var):    
#retrieving the best model for prediction call
    load_model = pickle.load(open('check.h5', 'rb'))
    prediction = load_model.predict([var])
    prob = load_model.predict_proba([var])
    return (print("The given statement is ",prediction[0]),
        print("The truth probability score is ",prob[0][1]))
if __name__ == '__main__':
    detecting(var)
