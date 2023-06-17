import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from project.pipeline.prediction_pipeline import CustomData, PredictPipeline