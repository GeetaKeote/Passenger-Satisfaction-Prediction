import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        preprocessro_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")
        model_path = os.path.join("artifacts/model_trainer", "model.pkl")

        processor = load_object(preprocessro_path)
        model = load_object(model_path)

        scaled = processor.transform(features)
        pred = model.predict(scaled)

        return pred


class CustomClass:
    def __init__(self, 
                  column_0: int,
                  gender: str,
                  customer_type: str,
                  age: int,
                  type_of_travel: str,
                  class_type: str,
                  flight_distance: int,
                  inflight_wifi_service: int,
                  departure_arrival_time_convenient: int,
                  ease_of_online_booking: int,
                  gate_location: int,
                  food_and_drink: int,
                  online_boarding: int,
                  seat_comfort: int,
                  inflight_entertainment: int,
                  onboard_service: int,
                  leg_room_service: int,
                  baggage_handling: int,
                  checkin_service: int,
                  inflight_service: int,
                  cleanliness: int,
                  departure_delay_minutes: int,
                  arrival_delay_minutes: float):
        self.column_0 = column_0
        self.gender = gender
        self.customer_type = customer_type
        self.age = age
        self.type_of_travel = type_of_travel
        self.class_type = class_type
        self.flight_distance = flight_distance
        self.inflight_wifi_service = inflight_wifi_service
        self.departure_arrival_time_convenient = departure_arrival_time_convenient
        self.ease_of_online_booking = ease_of_online_booking
        self.gate_location = gate_location
        self.food_and_drink = food_and_drink
        self.online_boarding = online_boarding
        self.seat_comfort = seat_comfort
        self.inflight_entertainment = inflight_entertainment
        self.onboard_service = onboard_service
        self.leg_room_service = leg_room_service
        self.baggage_handling = baggage_handling
        self.checkin_service = checkin_service
        self.inflight_service = inflight_service
        self.cleanliness = cleanliness
        self.departure_delay_minutes = departure_delay_minutes
        self.arrival_delay_minutes = arrival_delay_minutes

    def get_data_DataFrame(self):
        try:
            custom_input = {
                "column_0": [self.column_0],
                "Gender": [self.gender],
                "Customer Type": [self.customer_type],
                "Age": [self.age],
                "Type of Travel": [self.type_of_travel],
                "Class": [self.class_type],
                "Flight Distance": [self.flight_distance],
                "Inflight wifi service": [self.inflight_wifi_service],
                "Departure/Arrival time convenient": [self.departure_arrival_time_convenient],
                "Ease of Online booking": [self.ease_of_online_booking],
                "Gate location": [self.gate_location],
                "Food and drink": [self.food_and_drink],
                "Online boarding": [self.online_boarding],
                "Seat comfort": [self.seat_comfort],
                "Inflight entertainment": [self.inflight_entertainment],
                "On-board service": [self.onboard_service],
                "Leg room service": [self.leg_room_service],
                "Baggage handling": [self.baggage_handling],
                "Checkin service": [self.checkin_service],
                "Inflight service": [self.inflight_service],
                "Cleanliness": [self.cleanliness],
                "Departure Delay in Minutes": [self.departure_delay_minutes],
                "Arrival Delay in Minutes": [self.arrival_delay_minutes]
            }

            data = pd.DataFrame(custom_input)

            return data
        except Exception as e:
            raise CustomException(e, sys)
