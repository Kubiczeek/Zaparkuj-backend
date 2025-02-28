from math import ceil
import random
import pickle
import secrets
import os
import asyncio
from datetime import datetime
from typing import Annotated
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware

from lib.vehicleDetection import count_vehicle

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(periodic_task())
    yield
app = FastAPI(lifespan=lifespan)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()

PARKING_LOTS_DIR = "parking-lots"

def is_authorized(credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = b"kubiczeek" # This is safe (no it's not)
    is_correct_username = secrets.compare_digest(current_username_bytes, correct_username_bytes)
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = b"xvylUhi&]%,WH@1" # This is safe (no it's not)
    is_correct_password = secrets.compare_digest(current_password_bytes, correct_password_bytes)
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials

def get_newest_file(folder) -> str:
    path = f"{PARKING_LOTS_DIR}/{folder}"
    files = os.listdir(path)
    files.sort(key=lambda x: os.path.getmtime(f"{path}/{x}"))
    return files[-1]

def load_database(filename='database.pkl') -> dict:
    # Load the database from a file with pickle
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

def save_database(data, filename='database.pkl'):
    # Save the database to a file with pickle
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def get_current_time_to_quarter_hour() -> str:
    now = datetime.now()
    return now.replace(minute=15*(now.minute // 15), second=0, microsecond=0).strftime("%H:%M")

def get_day_of_week() -> int:
    # Monday is 0 and Sunday is 6
    return datetime.now().weekday()

def generate_csv():
    # Generate a CSV file with data from the database
    data = load_database() or {}
    with open('database.csv', 'w') as file:
        file.write("identifier,day,time,occupancy\n")
        for identifier, days in data.items():
            for day, times in days.items():
                for time, occupancy in times.items():
                    for count in occupancy:
                        file.write(f"{identifier},{day},{time},{count}\n")

    return "database.csv"

def get_nearest_quarters(time_str: str) -> tuple[str, str]:
    time = datetime.strptime(time_str, "%H:%M")
    minutes = time.hour * 60 + time.minute
    lower_quarter = minutes - (minutes % 15)
    upper_quarter = lower_quarter + 15 if lower_quarter + 15 < 24 * 60 else lower_quarter

    lower_time = f"{lower_quarter // 60:02d}:{lower_quarter % 60:02d}"
    upper_time = f"{upper_quarter // 60:02d}:{upper_quarter % 60:02d}"
    return lower_time, upper_time

def calculate_expected_occupancy(data, identifier, day, time, image_path) -> int:
    if identifier not in data or day not in data[identifier]:
        return -1

    lower_time, upper_time = get_nearest_quarters(time)

    # Get values for both time points
    lower_values = data[identifier][day].get(lower_time, [])
    upper_values = data[identifier][day].get(upper_time, [])

    if not lower_values or not upper_values:
        return -1

    # Get current occupancy from the image
    current_occupancy = count_vehicle(image_path)

    # Calculate weighted averages for both time points
    lower_historical = sum(lower_values) / len(lower_values) if lower_values else current_occupancy
    lower_occupancy = 0.6 * current_occupancy + 0.4 * lower_historical

    upper_historical = sum(upper_values) / len(upper_values) if upper_values else current_occupancy
    upper_occupancy = 0.6 * current_occupancy + 0.4 * upper_historical

    # Calculate time progression between quarter hours (0 to 1)
    time_parts = [int(x) for x in time.split(":")]
    minutes = time_parts[0] * 60 + time_parts[1]
    progress = (minutes % 15) / 15

    # Linear interpolation between lower and upper occupancy
    weighted_occupancy = lower_occupancy + (upper_occupancy - lower_occupancy) * progress

    return ceil(weighted_occupancy)

async def periodic_task():
    while True:
        # Loop through all parking lots and count vehicles
        # Save the count in a database
        time = get_current_time_to_quarter_hour()
        day = get_day_of_week()
        data = load_database() or {}
        for directory in os.listdir(PARKING_LOTS_DIR):
            if not os.path.isdir(f"{PARKING_LOTS_DIR}/{directory}"):
                continue
            image = get_newest_file(directory)
            count = count_vehicle(f"{PARKING_LOTS_DIR}/{directory}/{image}")
            identifier = int(directory.split("-")[-1])
            if identifier not in data:
                data[identifier] = {}
            if day not in data[identifier]:
                data[identifier][day] = {}
            if time not in data[identifier][day]:
                data[identifier][day][time] = []
            # Push to the end of the table
            data[identifier][day][time].append(count)
            # Ensure the table length does not exceed 4 (FIFO)
            if len(data[identifier][day][time]) > 4:
                data[identifier][day][time].pop(0)

        save_database(data)

        await asyncio.sleep(15*60) # 15 minutes


def database_set_negative_to_zero():
    data = load_database() or {}

    # Iterate through all parking lots
    for identifier in data:
        # Iterate through all days
        for day in data[identifier]:
            # Iterate through all times
            for time in data[identifier][day]:
                # Iterate through all occupancy values
                for i in range(len(data[identifier][day][time])):
                    # Set negative values to zero
                    if data[identifier][day][time][i] < 0:
                        data[identifier][day][time][i] = 0

    # Save the updated database
    save_database(data)
    return data


@app.get("/api/v1")
async def root():
    database_set_negative_to_zero()
    return JSONResponse(content=jsonable_encoder({"message": "Welcome to the parking occupancy API"}), status_code=status.HTTP_200_OK)

@app.get("/api/v1/parking-occupancy/{id}")
async def say_hello(id: int, credentials: Annotated[HTTPBasicCredentials, Depends(is_authorized)], conf: float = 0.3):
    directory = f"parking-lot-{id}"
    if not os.path.exists(f"{PARKING_LOTS_DIR}/{directory}"):
        return JSONResponse(content=jsonable_encoder({"error": "Parking lot not found"}), status_code=status.HTTP_404_NOT_FOUND)
    if conf < 0 or conf > 1:
        return JSONResponse(content=jsonable_encoder({"error": "Confidence threshold must be between 0 and 1"}), status_code=status.HTTP_400_BAD_REQUEST)
    image = get_newest_file(directory)
    count = count_vehicle(f"{PARKING_LOTS_DIR}/{directory}/{image}", conf)
    # if (id == 1):
    #    count = 90
    time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return JSONResponse(content=jsonable_encoder({"vehicle_count": count, "conf": conf, "time_stamp": time_stamp}), status_code=status.HTTP_200_OK)

@app.get("/api/v1/parking-occupancy/{id}/history")
async def get_history(id: int, credentials: Annotated[HTTPBasicCredentials, Depends(is_authorized)], time_arrival: str = None):
    data = load_database() or {}
    if id not in data:
        return JSONResponse(content=jsonable_encoder({"error": "Parking lot not found"}), status_code=status.HTTP_404_NOT_FOUND)
    day = get_day_of_week()
    time = get_current_time_to_quarter_hour()
    if time_arrival:
        time = time_arrival
    expected_occupancy = calculate_expected_occupancy(data, id, day, time, f"{PARKING_LOTS_DIR}/parking-lot-{id}/{get_newest_file(f'parking-lot-{id}')}")
    if expected_occupancy == -1:
        return JSONResponse(content=jsonable_encoder({"error": "No data available"}), status_code=status.HTTP_404_NOT_FOUND)
    return JSONResponse(content=jsonable_encoder({"expected_occupancy": expected_occupancy}), status_code=status.HTTP_200_OK)

@app.get("/api/v1/admin/csv")
async def get_csv(credentials: Annotated[HTTPBasicCredentials, Depends(is_authorized)]):
    file_path = generate_csv()
    return FileResponse(
        path=file_path,
        filename="database.csv",
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=database.csv"}
    )