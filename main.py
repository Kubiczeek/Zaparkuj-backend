from math import ceil, exp
import random
import pickle
import secrets
import os
import asyncio
import json
import requests
from datetime import datetime
from typing import Annotated
from contextlib import asynccontextmanager
import numpy as np
from io import BytesIO

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
PARKING_DATA_FILE = f"{PARKING_LOTS_DIR}/data.json"


def load_parking_data():
    """Načte data o parkovištích z JSON souboru."""
    if os.path.exists(PARKING_DATA_FILE):
        with open(PARKING_DATA_FILE, 'r') as file:
            return json.load(file)
    return []


def get_image_from_camera(url):
    """Získá obrázek z IP kamery."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return BytesIO(response.content)
        else:
            print(f"Chyba při získávání obrázku z URL {url}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Chyba při získávání obrázku z URL {url}: {str(e)}")
        return None


def is_authorized(credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = b"kubiczeek"  # This is safe (no it's not)
    is_correct_username = secrets.compare_digest(current_username_bytes, correct_username_bytes)
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = b"xvylUhi&]%,WH@1"  # This is safe (no it's not)
    is_correct_password = secrets.compare_digest(current_password_bytes, correct_password_bytes)
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials


def load_database(filename='database.pkl') -> dict:
    # Load the database from a file with pickle
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    return {}


def save_database(data, filename='database.pkl'):
    # Save the database to a file with pickle
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def get_current_time_to_quarter_hour() -> str:
    now = datetime.now()
    return now.replace(minute=15 * (now.minute // 15), second=0, microsecond=0).strftime("%H:%M")


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


def get_nearest_quarters(time_str: str) -> int:
    """Najde nejbližší čtvrthodinové intervaly k zadanému času."""
    time = datetime.strptime(time_str, "%H:%M")
    minutes = time.hour * 60 + time.minute
    quarter = (minutes // 15) * 15
    return quarter


def get_historical_occupancy(data, identifier, day, target_time, past_days=5, decay_factor=0.3) -> float:
    """Spočítá váženou historickou obsazenost pro daný čas."""
    # Získání nejbližšího 15minutového intervalu
    target_time_quarter = get_nearest_quarters(target_time)

    # Převod času zpět na formát HH:MM
    target_time_str = f"{target_time_quarter // 60:02d}:{target_time_quarter % 60:02d}"

    weights = np.exp(-decay_factor * np.arange(past_days))
    weights /= weights.sum()

    # Pro každý z minulých dnů získáme data o obsazenosti
    occupancy_values = []

    for i in range(past_days):
        # Výpočet dne v týdnu (0-6) pro daný minulý den
        past_day = (day - i) % 7

        # Získání dat o obsazenosti pro daný den a čas
        if identifier in data and past_day in data[identifier] and target_time_str in data[identifier][past_day]:
            # Použijeme průměr z dostupných hodnot
            occupancy_data = data[identifier][past_day][target_time_str]
            # Filter out None values before summing
            valid_data = [x for x in occupancy_data if x is not None]
            avg_occupancy = sum(valid_data) / len(valid_data) if valid_data else 0
            occupancy_values.append(avg_occupancy)
        else:
            # Pokud data nejsou k dispozici, použijeme 0
            occupancy_values.append(0)

    # Výpočet váženého průměru
    if occupancy_values:
        return np.dot(weights[:len(occupancy_values)], occupancy_values)
    return 0


def get_deltaO(data, identifier, day, target_time, past_days=5) -> float:
    """Spočítá deltaO (změnu obsazenosti mezi dolním a horním intervalem)."""
    # Převod času na minuty
    time_parts = [int(x) for x in target_time.split(":")]
    minutes = time_parts[0] * 60 + time_parts[1]

    # Výpočet dolního a horního času
    lower_minutes = minutes - 15
    upper_minutes = minutes + 15

    # Převod zpět na formát HH:MM
    lower_time = f"{lower_minutes // 60:02d}:{lower_minutes % 60:02d}"
    upper_time = f"{upper_minutes // 60:02d}:{upper_minutes % 60:02d}"

    lower_occupancy = get_historical_occupancy(data, identifier, day, lower_time, past_days, 0.3)
    upper_occupancy = get_historical_occupancy(data, identifier, day, upper_time, past_days, 0.3)

    return upper_occupancy - lower_occupancy


def calculate_expected_occupancy(data, identifier, day, time) -> int:
    """Predikuje budoucí obsazenost parkoviště."""
    if identifier not in data or day not in data[identifier]:
        return -1

    # Získání aktuální obsazenosti z IP kamery
    parking_data = load_parking_data()
    parking_info = next((p for p in parking_data if p["id"] == identifier), None)

    if not parking_info or not parking_info.get("url"):
        return -1

    # Získání obrázku z IP kamery
    image_data = get_image_from_camera(parking_info["url"])
    if not image_data:
        return -1

    # Získání aktuální obsazenosti z obrázku
    current_occupancy = count_vehicle(image_data)

    # Aktuální čas
    current_time = get_current_time_to_quarter_hour()

    # Výpočet rozdílu v minutách mezi aktuálním časem a cílovým časem
    current_minutes = sum(int(x) * (60 if i == 0 else 1) for i, x in enumerate(current_time.split(":")))
    target_minutes = sum(int(x) * (60 if i == 0 else 1) for i, x in enumerate(time.split(":")))
    time_difference = target_minutes - current_minutes

    # Pokud je cílový čas před aktuálním, předpokládáme, že je to další den
    if time_difference < 0:
        time_difference += 24 * 60

    # Parametry pro predikci
    DECAY_FACTOR = 1.25
    PAST_DAYS = 5

    # Výpočet váhy pro aktuální obsazenost
    weight = exp(-DECAY_FACTOR * (time_difference / 60))

    # Získání historické obsazenosti
    historical_occupancy = get_historical_occupancy(data, identifier, day, time, PAST_DAYS, 0.3)

    # Výpočet změny obsazenosti
    deltaO = get_deltaO(data, identifier, day, time, PAST_DAYS)
    deltaO_adjusted = (time_difference / 15) * deltaO

    # Predikce obsazenosti
    predicted_occupancy = (weight * (current_occupancy + deltaO_adjusted)) + ((1 - weight) * historical_occupancy)

    return ceil(predicted_occupancy)


async def periodic_task():
    while True:
        # Loop through all parking lots and count vehicles from IP cameras
        time = get_current_time_to_quarter_hour()
        day = get_day_of_week()
        data = load_database() or {}

        # Načtení dat o parkovištích
        parking_data = load_parking_data()

        for parking in parking_data:
            identifier = parking["id"]
            url = parking.get("url")

            if not url:
                continue

            # Získání obrázku z IP kamery
            image_data = get_image_from_camera(url)
            if not image_data:
                continue

            # Detekce vozidel
            count = count_vehicle(image_data)

            # Uložení dat do databáze
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

        await asyncio.sleep(15 * 60)  # 15 minutes


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
    return JSONResponse(content=jsonable_encoder({"message": "Welcome to the parking occupancy API"}),
                        status_code=status.HTTP_200_OK)


@app.get("/api/v1/parking-occupancy/{id}")
async def say_hello(id: int, credentials: Annotated[HTTPBasicCredentials, Depends(is_authorized)], conf: float = 0.3):
    # Načtení dat o parkovištích
    parking_data = load_parking_data()
    parking_info = next((p for p in parking_data if p["id"] == id), None)

    if not parking_info:
        return JSONResponse(content=jsonable_encoder({"error": "Parking lot not found"}),
                            status_code=status.HTTP_404_NOT_FOUND)

    if not parking_info.get("url"):
        return JSONResponse(content=jsonable_encoder({"error": "Parking lot camera URL not configured"}),
                            status_code=status.HTTP_404_NOT_FOUND)

    if conf < 0 or conf > 1:
        return JSONResponse(content=jsonable_encoder({"error": "Confidence threshold must be between 0 and 1"}),
                            status_code=status.HTTP_400_BAD_REQUEST)

    # Získání obrázku z IP kamery
    image_data = get_image_from_camera(parking_info["url"])
    if not image_data:
        return JSONResponse(content=jsonable_encoder({"error": "Failed to get image from camera"}),
                            status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

    count = count_vehicle(image_data, conf)
    time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return JSONResponse(content=jsonable_encoder({
        "vehicle_count": count,
        "conf": conf,
        "time_stamp": time_stamp,
        "max_spaces": parking_info.get("maxParkingSpaces", 0),
        "max_handicapped": parking_info.get("maxHandicappedSpaces", 0)
    }), status_code=status.HTTP_200_OK)


@app.get("/api/v1/parking-occupancy/{id}/history")
async def get_history(id: int, credentials: Annotated[HTTPBasicCredentials, Depends(is_authorized)],
                      time_arrival: str = None):
    data = load_database() or {}
    if id not in data:
        return JSONResponse(content=jsonable_encoder({"error": "Parking lot not found"}),
                            status_code=status.HTTP_404_NOT_FOUND)

    # Načtení dat o parkovištích
    parking_data = load_parking_data()
    parking_info = next((p for p in parking_data if p["id"] == id), None)

    if not parking_info:
        return JSONResponse(content=jsonable_encoder({"error": "Parking lot not found"}),
                            status_code=status.HTTP_404_NOT_FOUND)

    day = get_day_of_week()
    time = get_current_time_to_quarter_hour()
    if time_arrival:
        time = time_arrival

    expected_occupancy = calculate_expected_occupancy(data, id, day, time)

    if expected_occupancy == -1:
        return JSONResponse(content=jsonable_encoder({"error": "No data available"}),
                            status_code=status.HTTP_404_NOT_FOUND)

    return JSONResponse(content=jsonable_encoder({
        "expected_occupancy": expected_occupancy,
        "max_spaces": parking_info.get("maxParkingSpaces", 0),
        "max_handicapped": parking_info.get("maxHandicappedSpaces", 0)
    }), status_code=status.HTTP_200_OK)


@app.get("/api/v1/admin/csv")
async def get_csv(credentials: Annotated[HTTPBasicCredentials, Depends(is_authorized)]):
    file_path = generate_csv()
    return FileResponse(
        path=file_path,
        filename="database.csv",
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=database.csv"}
    )
