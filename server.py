from fastapi import FastAPI, APIRouter, HTTPException, Depends, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
from enum import Enum
import hashlib
import random
import math

# -------------------------
# Environment & constants
# -------------------------
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "5555")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

# -------------------------
# Database
# -------------------------
client = AsyncIOMotorClient(MONGO_URI)
db = client["project"]

# Optional: ensure indexes (id unique)
# Note: creating indexes is async; leaving as comment for future improvement.
# await db.users.create_index("id", unique=True)
# await db.users.create_index("telegram_id", unique=True)

# -------------------------
# FastAPI app & router
# -------------------------
app = FastAPI(title="Kremenchuk Taxi API")
api_router = APIRouter(prefix="/api")

# Single CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Helpers
# -------------------------
def hash_password(password: str) -> str:
    """Simple SHA256 hashing. Replace with bcrypt for production."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def calculate_price(tariff: "TariffType", distance_km: float, surge_multiplier: float = 1.0) -> float:
    base = TARIFF_PRICES[tariff]
    km_price = distance_km * PRICE_PER_KM[tariff]
    total = (base + km_price) * surge_multiplier
    return round(total, 2)

def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Haversine formula approximate distance in km."""
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def calculate_ride_duration(distance_km: float) -> int:
    duration = max(1, min(120, int(distance_km * 2)))  # more realistic upper bound
    return duration

def now_utc():
    return datetime.now(timezone.utc)

# -------------------------
# Enums & models
# -------------------------
class TariffType(str, Enum):
    ECONOMY = "economy"
    COMFORT = "comfort"
    BUSINESS = "business"

class RideStatus(str, Enum):
    PENDING = "pending"
    SEARCHING = "searching"
    DRIVER_FOUND = "driver_found"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class UserRole(str, Enum):
    PASSENGER = "passenger"
    ADMIN = "admin"

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    telegram_id: str
    first_name: str
    last_name: str
    phone: str
    role: UserRole = UserRole.PASSENGER
    created_at: datetime = Field(default_factory=now_utc)

class UserCreate(BaseModel):
    telegram_id: str
    first_name: str
    last_name: str
    phone: str

class UserLogin(BaseModel):
    telegram_id: str

class AdminLogin(BaseModel):
    password: str

class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None

class Driver(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    phone: str
    car_model: str
    car_number: str
    car_color: str
    photo_url: Optional[str] = None
    rating: float = 5.0
    total_rides: int = 0
    is_active: bool = True
    created_at: datetime = Field(default_factory=now_utc)

class DriverCreate(BaseModel):
    name: str
    phone: str
    car_model: str
    car_number: str
    car_color: str
    photo_url: Optional[str] = None

class Ride(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    driver_id: Optional[str] = None
    pickup_address: str
    pickup_lat: float
    pickup_lng: float
    destination_address: str
    destination_lat: float
    destination_lng: float
    tariff: TariffType
    price: float
    original_price: float = 0
    surge_multiplier: float = 1.0
    distance_km: float = 0
    estimated_duration_min: int = 5
    status: RideStatus = RideStatus.PENDING
    created_at: datetime = Field(default_factory=now_utc)
    completed_at: Optional[datetime] = None

class RideCreate(BaseModel):
    user_id: str
    pickup_address: str
    pickup_lat: float
    pickup_lng: float
    destination_address: str
    destination_lat: float
    destination_lng: float
    tariff: TariffType
    accept_surge: bool = False

class Review(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ride_id: str
    user_id: str
    driver_id: str
    rating: int  # 1-5
    comment: Optional[str] = None
    admin_response: Optional[str] = None
    created_at: datetime = Field(default_factory=now_utc)

class ReviewCreate(BaseModel):
    ride_id: str
    user_id: str
    driver_id: str
    rating: int
    comment: Optional[str] = None

class AdminStats(BaseModel):
    total_rides: int
    rides_today: int
    rides_week: int
    total_revenue: float
    average_rating: float
    total_complaints: int
    top_drivers: List[dict]

class AvailabilityResponse(BaseModel):
    available: bool
    available_drivers: int
    surge_multiplier: float
    estimated_wait_min: int
    message: str

# -------------------------
# Pricing
# -------------------------
TARIFF_PRICES = {
    TariffType.ECONOMY: 45,
    TariffType.COMFORT: 65,
    TariffType.BUSINESS: 95,
}

PRICE_PER_KM = {
    TariffType.ECONOMY: 8,
    TariffType.COMFORT: 12,
    TariffType.BUSINESS: 18,
}

# -------------------------
# Surge & helpers using DB
# -------------------------
async def get_surge_multiplier():
    active_drivers = await db.drivers.count_documents({"is_active": True})
    pending_rides = await db.rides.count_documents({"status": {"$in": [RideStatus.PENDING, RideStatus.SEARCHING]}})
    if active_drivers == 0:
        return 1.0, 0, pending_rides  # default multiplier 1.0, no drivers
    demand_ratio = pending_rides / max(1, active_drivers)
    if demand_ratio <= 0.5:
        surge = 1.0
    elif demand_ratio <= 1.0:
        surge = 1.2
    elif demand_ratio <= 2.0:
        surge = 1.5
    else:
        surge = 2.0
    return surge, active_drivers, pending_rides

# -------------------------
# User endpoints
# -------------------------
@api_router.post("/users/register", response_model=User)
async def register_user(user_data: UserCreate):
    existing = await db.users.find_one({"telegram_id": user_data.telegram_id})
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")
    if not user_data.phone or not user_data.phone.startswith("+380"):
        raise HTTPException(status_code=400, detail="Phone must start with +380")
    user = User(**user_data.model_dump(), role=UserRole.PASSENGER)
    doc = user.model_dump()
    # store datetime objects directly (Mongo can store datetimes)
    await db.users.insert_one(doc)
    return user

@api_router.post("/users/login", response_model=User)
async def login_user(login_data: UserLogin):
    user = await db.users.find_one({"telegram_id": login_data.telegram_id}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # Ensure created_at is datetime object
    if isinstance(user.get("created_at"), str):
        user["created_at"] = datetime.fromisoformat(user["created_at"])
    return User(**user)

@api_router.post("/users/admin-login", response_model=User)
async def admin_login(login_data: AdminLogin):
    if login_data.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    admin = await db.users.find_one({"role": UserRole.ADMIN.value}, {"_id": 0})
    if not admin:
        admin_user = User(
            telegram_id="admin",
            first_name="Адміністратор",
            last_name="",
            phone="+380000000000",
            role=UserRole.ADMIN
        )
        doc = admin_user.model_dump()
        await db.users.insert_one(doc)
        return admin_user
    if isinstance(admin.get("created_at"), str):
        admin["created_at"] = datetime.fromisoformat(admin["created_at"])
    return User(**admin)

@api_router.get("/users/{telegram_id}", response_model=User)
async def get_user(telegram_id: str):
    user = await db.users.find_one({"telegram_id": telegram_id}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if isinstance(user.get("created_at"), str):
        user["created_at"] = datetime.fromisoformat(user["created_at"])
    return User(**user)

@api_router.put("/users/{user_id}")
async def update_user(user_id: str, user_data: UserUpdate):
    update_dict = {k: v for k, v in user_data.model_dump().items() if v is not None}
    if 'phone' in update_dict and not update_dict['phone'].startswith("+380"):
        raise HTTPException(status_code=400, detail="Phone must start with +380")
    result = await db.users.update_one({"id": user_id}, {"$set": update_dict})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "updated"}

@api_router.get("/users/check/{telegram_id}")
async def check_user_exists(telegram_id: str):
    user = await db.users.find_one({"telegram_id": telegram_id})
    return {"exists": user is not None, "is_admin": user.get('role') == UserRole.ADMIN.value if user else False}

# -------------------------
# Driver endpoints
# -------------------------
@api_router.post("/drivers", response_model=Driver)
async def create_driver(driver_data: DriverCreate):
    driver = Driver(**driver_data.model_dump())
    doc = driver.model_dump()
    await db.drivers.insert_one(doc)
    return driver

@api_router.get("/drivers", response_model=List[Driver])
async def get_drivers(active_only: bool = Query(False)):
    query = {"is_active": True} if active_only else {}
    drivers = await db.drivers.find(query, {"_id": 0}).to_list(100)
    # convert possible string datetimes
    for d in drivers:
        if isinstance(d.get('created_at'), str):
            d['created_at'] = datetime.fromisoformat(d['created_at'])
    return drivers

@api_router.get("/drivers/{driver_id}", response_model=Driver)
async def get_driver(driver_id: str):
    driver = await db.drivers.find_one({"id": driver_id}, {"_id": 0})
    if not driver:
        raise HTTPException(status_code=404, detail="Driver not found")
    if isinstance(driver.get('created_at'), str):
        driver['created_at'] = datetime.fromisoformat(driver['created_at'])
    return Driver(**driver)

@api_router.put("/drivers/{driver_id}")
async def update_driver(driver_id: str, driver_data: DriverCreate):
    result = await db.drivers.update_one({"id": driver_id}, {"$set": driver_data.model_dump()})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Driver not found")
    return {"status": "updated"}

@api_router.delete("/drivers/{driver_id}")
async def delete_driver(driver_id: str):
    result = await db.drivers.delete_one({"id": driver_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Driver not found")
    return {"status": "deleted"}

@api_router.patch("/drivers/{driver_id}/toggle")
async def toggle_driver_status(driver_id: str):
    driver = await db.drivers.find_one({"id": driver_id})
    if not driver:
        raise HTTPException(status_code=404, detail="Driver not found")
    new_status = not driver.get('is_active', True)
    await db.drivers.update_one({"id": driver_id}, {"$set": {"is_active": new_status}})
    return {"status": "toggled", "is_active": new_status}

# -------------------------
# Availability
# -------------------------
@api_router.get("/availability", response_model=AvailabilityResponse)
async def check_availability():
    surge, active_drivers, pending_rides = await get_surge_multiplier()
    if active_drivers == 0:
        return AvailabilityResponse(
            available=False,
            available_drivers=0,
            surge_multiplier=1.0,
            estimated_wait_min=0,
            message="На жаль, зараз немає доступних водіїв. Спробуйте пізніше."
        )
    if pending_rides > active_drivers * 3:
        return AvailabilityResponse(
            available=False,
            available_drivers=active_drivers,
            surge_multiplier=surge,
            estimated_wait_min=15,
            message="Занадто багато замовлень. Спробуйте через кілька хвилин."
        )
    estimated_wait = max(1, min(10, int(pending_rides / max(1, active_drivers) * 3)))
    message = "Водії доступні"
    if surge > 1.0:
        message = f"Підвищений попит! Ціни збільшено в {surge}x разів."
    return AvailabilityResponse(
        available=True,
        available_drivers=active_drivers,
        surge_multiplier=surge,
        estimated_wait_min=estimated_wait,
        message=message
    )

# -------------------------
# Rides
# -------------------------
@api_router.post("/rides", response_model=Ride)
async def create_ride(ride_data: RideCreate):
    surge, active_drivers, pending_rides = await get_surge_multiplier()
    if active_drivers == 0:
        raise HTTPException(status_code=503, detail="Немає доступних водіїв")
    distance = calculate_distance(
        ride_data.pickup_lat, ride_data.pickup_lng,
        ride_data.destination_lat, ride_data.destination_lng
    )
    original_price = calculate_price(ride_data.tariff, distance, 1.0)
    final_price = calculate_price(ride_data.tariff, distance, surge)
    duration = calculate_ride_duration(distance)
    ride = Ride(
        **ride_data.model_dump(exclude={'accept_surge'}),
        price=final_price,
        original_price=original_price,
        surge_multiplier=surge,
        distance_km=round(distance, 2),
        estimated_duration_min=duration,
        status=RideStatus.SEARCHING
    )
    doc = ride.model_dump()
    await db.rides.insert_one(doc)
    return ride

@api_router.get("/rides/{ride_id}", response_model=Ride)
async def get_ride(ride_id: str):
    ride = await db.rides.find_one({"id": ride_id}, {"_id": 0})
    if not ride:
        raise HTTPException(status_code=404, detail="Ride not found")
    return Ride(**ride)

@api_router.get("/rides/user/{user_id}", response_model=List[Ride])
async def get_user_rides(user_id: str):
    rides = await db.rides.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1).to_list(50)
    return rides

@api_router.patch("/rides/{ride_id}/assign-driver")
async def assign_driver(ride_id: str):
    drivers = await db.drivers.find({"is_active": True}, {"_id": 0}).to_list(100)
    if not drivers:
        raise HTTPException(status_code=404, detail="No available drivers")
    driver = random.choice(drivers)
    result = await db.rides.update_one({"id": ride_id}, {"$set": {"driver_id": driver['id'], "status": RideStatus.DRIVER_FOUND}})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Ride not found")
    return {"driver_id": driver['id'], "driver": driver}

@api_router.patch("/rides/{ride_id}/status")
async def update_ride_status(ride_id: str, status: RideStatus):
    update_data = {"status": status}
    if status == RideStatus.COMPLETED:
        update_data["completed_at"] = now_utc()
        ride = await db.rides.find_one({"id": ride_id})
        if ride and ride.get('driver_id'):
            await db.drivers.update_one({"id": ride['driver_id']}, {"$inc": {"total_rides": 1}})
    result = await db.rides.update_one({"id": ride_id}, {"$set": update_data})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Ride not found")
    return {"status": "updated"}

@api_router.patch("/rides/{ride_id}/cancel")
async def cancel_ride(ride_id: str):
    result = await db.rides.update_one({"id": ride_id}, {"$set": {"status": RideStatus.CANCELLED}})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Ride not found")
    return {"status": "cancelled"}

# -------------------------
# Reviews
# -------------------------
@api_router.post("/reviews", response_model=Review)
async def create_review(review_data: ReviewCreate):
    if not 1 <= review_data.rating <= 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    review = Review(**review_data.model_dump())
    doc = review.model_dump()
    await db.reviews.insert_one(doc)
    reviews = await db.reviews.find({"driver_id": review_data.driver_id}, {"_id": 0}).to_list(1000)
    avg_rating = sum(r['rating'] for r in reviews) / len(reviews) if reviews else 5.0
    await db.drivers.update_one({"id": review_data.driver_id}, {"$set": {"rating": round(avg_rating, 2)}})
    return review

@api_router.get("/reviews", response_model=List[Review])
async def get_all_reviews():
    reviews = await db.reviews.find({}, {"_id": 0}).sort("created_at", -1).to_list(100)
    return reviews

@api_router.patch("/reviews/{review_id}/respond")
async def respond_to_review(review_id: str, response: str = Query(...)):
    result = await db.reviews.update_one({"id": review_id}, {"$set": {"admin_response": response}})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Review not found")
    return {"status": "responded"}

# -------------------------
# Admin stats
# -------------------------
@api_router.get("/admin/stats", response_model=AdminStats)
async def get_admin_stats():
    now = now_utc()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)
    total_rides = await db.rides.count_documents({"status": RideStatus.COMPLETED})
    rides_today = await db.rides.count_documents({"status": RideStatus.COMPLETED, "created_at": {"$gte": today_start}})
    rides_week = await db.rides.count_documents({"status": RideStatus.COMPLETED, "created_at": {"$gte": week_start}})
    completed_rides = await db.rides.find({"status": RideStatus.COMPLETED}, {"_id": 0}).to_list(10000)
    total_revenue = sum(r.get('price', 0) for r in completed_rides)
    reviews = await db.reviews.find({}, {"_id": 0}).to_list(10000)
    average_rating = (sum(r['rating'] for r in reviews) / len(reviews)) if reviews else 5.0
    complaints = sum(1 for r in reviews if r['rating'] <= 2)
    drivers = await db.drivers.find({}, {"_id": 0}).sort("rating", -1).to_list(5)
    top_drivers = [{"name": d['name'], "rating": d['rating'], "rides": d.get('total_rides', 0)} for d in drivers]
    return AdminStats(
        total_rides=total_rides,
        rides_today=rides_today,
        rides_week=rides_week,
        total_revenue=round(total_revenue, 2),
        average_rating=round(average_rating, 2),
        total_complaints=complaints,
        top_drivers=top_drivers
    )

# -------------------------
# Price calc & root
# -------------------------
@api_router.post("/calculate-price")
async def calculate_ride_price(
    pickup_lat: float,
    pickup_lng: float,
    dest_lat: float,
    dest_lng: float,
    tariff: TariffType
):
    distance = calculate_distance(pickup_lat, pickup_lng, dest_lat, dest_lng)
    surge, _, _ = await get_surge_multiplier()
    price = calculate_price(tariff, distance, surge)
    original_price = calculate_price(tariff, distance, 1.0)
    duration = calculate_ride_duration(distance)
    return {
        "distance_km": round(distance, 2),
        "price": price,
        "original_price": original_price,
        "surge_multiplier": surge,
        "estimated_duration_min": duration,
        "tariff": tariff
    }

@api_router.get("/")
async def root():
    return {"message": "Kremenchuk Taxi API", "status": "running"}

# include router
app.include_router(api_router)

# logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()