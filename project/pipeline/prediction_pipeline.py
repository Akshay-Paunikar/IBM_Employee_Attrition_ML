import sys
import os

import numpy as np
import pandas as pd

from project.exception import CustomException
from project.logger import logging
from project.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            
            return preds            
            
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 Age:int,
                 BusinessTravel:str,
                 Department:str,
                 DistanceFromHome:int,
                 EducationField:str,
                 EnvironmentSatisfaction:int,
                 Gender:str,
                 JobLevel:int,
                 JobRole:str,
                 JobSatisfaction:int,
                 MaritalStatus:str,
                 MonthlyIncome:int,
                 NumCompaniesWorked:int,
                 OverTime:str,
                 PercentSalaryHike:int,
                 PerformanceRating:int,
                 RelationshipSatisfaction:int,
                 StockOptionLevel:int,
                 TotalWorkingYears:int,
                 TrainingTimesLastYear:int,
                 WorkLifeBalance:int,
                 YearsAtCompany:int,
                 YearsInCurrentRole:int,
                 YearsSinceLastPromotion:int,
                 YearsWithCurrManager:int):
        
        self.Age = Age
        self.BusinessTravel = BusinessTravel
        self.Department = Department
        self.DistanceFromHome = DistanceFromHome
        self.EducationField = EducationField
        self.EnvironmentSatisfaction = EnvironmentSatisfaction
        self.Gender = Gender
        self.JobLevel = JobLevel
        self.JobRole = JobRole
        self.JobSatisfaction = JobSatisfaction
        self.MaritalStatus = MaritalStatus
        self.MonthlyIncome = MonthlyIncome
        self.NumCompaniesWorked = NumCompaniesWorked
        self.OverTime = OverTime
        self.PercentSalaryHike = PercentSalaryHike
        self.PerformanceRating = PerformanceRating
        self.RelationshipSatisfaction = RelationshipSatisfaction
        self.StockOptionLevel = StockOptionLevel
        self.TotalWorkingYears = TotalWorkingYears
        self.TrainingTimesLastYear = TrainingTimesLastYear
        self.WorkLifeBalance = WorkLifeBalance
        self.YearsAtCompany = YearsAtCompany
        self.YearsInCurrentRole = YearsInCurrentRole
        self.YearsSinceLastPromotion = YearsSinceLastPromotion
        self.YearsWithCurrManager = YearsWithCurrManager
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "BusinessTravel" :[self.BusinessTravel],
                "Department": [self.Department],
                "DistanceFromHome": [self.DistanceFromHome],
                "EducationField": [self.EducationField],
                "EnvironmentSatisfaction": [self.EnvironmentSatisfaction],
                "Gender": [self.Gender],
                "JobLevel": [self.JobLevel],
                "JobRole": [self.JobRole],
                "JobSatisfaction": [self.JobSatisfaction],
                "MaritalStatus": [self.MaritalStatus],
                "MonthlyIncome": [self.MonthlyIncome],
                "NumCompaniesWorked": [self.NumCompaniesWorked],
                "OverTime": [self.OverTime],
                "PercentSalaryHike": [self.PercentSalaryHike],
                "PerformanceRating": [self.PerformanceRating],
                "RelationshipSatisfaction": [self.RelationshipSatisfaction],
                "StockOptionLevel": [self.StockOptionLevel],
                "TotalWorkingYears": [self.TotalWorkingYears],
                "TrainingTimesLastYear": [self.TrainingTimesLastYear],
                "WorkLifeBalance": [self.WorkLifeBalance],
                "YearsAtCompany": [self.YearsAtCompany],
                "YearsInCurrentRole": [self.YearsInCurrentRole],
                "YearsSinceLastPromotion": [self.YearsSinceLastPromotion],
                "YearsWithCurrManager": [self.YearsWithCurrManager]                
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
              
        