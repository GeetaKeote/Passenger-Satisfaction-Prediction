from flask import Flask, render_template, request, jsonify
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomClass
from prediction.batch import  batch_prediction
import os
from src.logger import logging
from src.components.data_transformation import DataTransfromartionConfigs
from src.config.configuration import MODEL_FILE_PATH, FEATURE_ENG_OBJ_PATH, PREPROCESSING_OBJ_PATH
from src.pipeline.training_pipeline import Train
from werkzeug.utils import  secure_filename


app = Flask(__name__,template_folder='template')


@app.route("/",methods = ["GET", "POST"])
def prediction_data():
    if request.method == "GET":
        return render_template("form.html")
    
    else:
        data = CustomClass(
            Age = int(request.form.get("Age")),
            Flight_Distance = int(request.form.get("Flight_Distance")),
            Inflight_wifi_service = int(request.form.get("Inflight_wifi_service")),
            Departure_Arrival_time_convenient = int(request.form.get("Departure_Arrival_time_convenient")),
            Ease_of_Online_booking = int(request.form.get("Ease_of_Online_booking")),            
            Food_and_drink = int(request.form.get("Food_and_drink")),
            Online_boarding = int(request.form.get("Online_boarding")),
            Seat_comfort = int(request.form.get("Seat_comfort")),
            Inflight_entertainment = int(request.form.get("Inflight_entertainment")),
            On_board_service = int(request.form.get("On_board_service")),
            Leg_room_service = int(request.form.get("Leg_room_service")),
            Baggage_handling = int(request.form.get("Baggage_handling")),
            Checkin_service = int(request.form.get("Checkin_service")),
            Inflight_service = int(request.form.get("Inflight_service")),
            Cleanliness = int(request.form.get("Cleanliness")),           
            Gender = str(request.form.get("Gender")),
            Customer_Type = str(request.form.get("Customer_Type")),
            Type_of_Travel = str(request.form.get("Type_of_Travel")),
            Class = str(request.form.get("Class")),
            
        )

    final_data = data.get_data_DataFrame()
    pipeline_prediction = PredictionPipeline()
    pred = pipeline_prediction.predict(final_data)

    result = pred

    if result == 0:
        return render_template("results.html", final_result = "Survey Opinion of the customer is satisfied:{}".format(result) )

    elif result == 1:
            return render_template("results.html", final_result = "Survey Opinion of the customer is dissatisfied or neutral:{}".format(result) )
    
if __name__ == "__main__":
     app.run(host = "0.0.0.0", debug = True)