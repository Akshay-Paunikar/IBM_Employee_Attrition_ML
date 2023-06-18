import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from prediction_pipeline import CustomData, PredictPipeline
application = Flask(__name__)

app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    
    else:
        data = CustomData(
            Age = float(request.form.get('Age')),
            BusinessTravel = request.form.get('BusinessTravel'),
            Department = request.form.get('Department'),
            DistanceFromHome = float(request.form.get('DistanceFromHome')),
            EducationField = request.form.get('EducationField'),
            EnvironmentSatisfaction = float(request.form.get('EnvironmentSatisfaction')),
            Gender = request.form.get('Gender'),
            JobLevel = float(request.form.get('JobLevel')),
            JobRole = request.form.get('JobRole'),
            JobSatisfaction = float(request.form.get('JobSatisfaction')),
            MaritalStatus = request.form.get('MaritalStatus'),
            MonthlyIncome = float(request.form.get('MonthlyIncome')),
            NumCompaniesWorked = float(request.form.get('NumCompaniesWorked')),
            OverTime = request.form.get('OverTime'),
            PercentSalaryHike = float(request.form.get('PercentSalaryHike')),
            PerformanceRating = float(request.form.get('PerformanceRating')),
            RelationshipSatisfaction = float(request.form.get('RelationshipSatisfaction')),
            StockOptionLevel = float(request.form.get('StockOptionLevel')),
            TotalWorkingYears = float(request.form.get('TotalWorkingYears')),
            TrainingTimesLastYear = float(request.form.get('TrainingTimesLastYear')),
            WorkLifeBalance = float(request.form.get('WorkLifeBalance')),
            YearsAtCompany = float(request.form.get('YearsAtCompany')),
            YearsInCurrentRole = float(request.form.get('YearsInCurrentRole')),
            YearsSinceLastPromotion = float(request.form.get('YearsSinceLastPromotion')),
            YearsWithCurrManager = float(request.form.get('YearsWithCurrManager'))            
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('index.html', results=results[0])
    
if __name__ == "__main__":
    app.run(debug=True)