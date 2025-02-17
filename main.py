import secrets
import os
from datetime import datetime
from typing import Annotated

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware

from lib.vehicleDetection import count_vehicle

app = FastAPI()

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

def get_newest_file(folder):
    path = f"{PARKING_LOTS_DIR}/{folder}"
    files = os.listdir(path)
    files.sort(key=lambda x: os.path.getmtime(f"{path}/{x}"))
    return files[-1]

@app.get("/api/v1")
async def root():
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
