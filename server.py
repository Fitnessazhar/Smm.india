from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pathlib import Path
import os
import logging
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta, timezone
import uuid
import jwt
import bcrypt
import random
import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from emergentintegrations.llm.chat import LlmChat, UserMessage
from emergentintegrations.llm.openai.image_generation import OpenAIImageGeneration
import base64
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Environment variables
MONGO_URL = os.environ.get('MONGO_URL')
DB_NAME = os.environ.get('DB_NAME', 'azharfit_saas')
JWT_SECRET = os.environ.get('JWT_SECRET')
ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-3.5-turbo')
WHATSAPP_PHONE_ID = os.environ.get('WHATSAPP_PHONE_NUMBER_ID')
WHATSAPP_TOKEN = os.environ.get('WHATSAPP_ACCESS_TOKEN')

# Logging
logging.basicConfig(
    level=logging.DEBUG if os.environ.get('LOG_LEVEL') == 'debug' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MongoDB connection
client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# FITBOT Greeting Variations (10+)
GREETINGS = [
    "🔥 {name}! Time to crush it at Azhar Fitness Center! Your gains are waiting!",
    "💪 Hey {name}! The iron doesn't lift itself. Let's GO!",
    "⚡ {name}, your body is calling - answer at AFC Gym NOW!",
    "🏋️ What's up {name}! Champions train, losers complain. Which one are you?",
    "🚀 {name}! Your future self will thank you. Get moving!",
    "🎯 {name}, missed you at the gym! Time to get back on track!",
    "💯 {name}! Excuses don't burn calories. ACTION DOES!",
    "🔱 {name}, Azhar Fitness is calling your name! Let's build that beast!",
    "⚔️ {name}! The only bad workout is the one that didn't happen!",
    "👑 {name}, kings and queens train at AFC. Where have you been?",
    "🌟 {name}! Your goals don't care if you're tired. Let's MOVE!",
    "⏰ {name}, time waits for nobody. Neither does your fitness journey!"
]

RETENTION_MESSAGES = [
    "The only bad workout is the one that didn't happen.",
    "Your goals don't care if you're tired.",
    "Success is the sum of small efforts repeated day in and day out.",
    "Don't stop when you're tired. Stop when you're DONE.",
    "The pain you feel today will be the strength you feel tomorrow."
]

# Instagram themes
INSTAGRAM_THEMES = [
    {
        "name": "Heavy Lifting",
        "prompt": "Create a hyper-realistic HD fitness poster showing intense heavy lifting workout at AZHAR FITNESS CENTER gym. Dramatic lighting with blue, orange, and chrome color scheme. Muscular athlete performing deadlift or squat. Professional photography style, 9:16 vertical format. Include motivational energy and gym equipment. High contrast, cinematic lighting."
    },
    {
        "name": "Healthy Meal Prep",
        "prompt": "Create a stunning HD poster of healthy meal prep for fitness enthusiasts at AZHAR FITNESS CENTER. Arranged protein-rich meals in containers, vibrant vegetables, and supplements. Blue, orange, and chrome color palette. Clean, professional food photography. 9:16 vertical format. Bright, appetizing lighting with fitness theme."
    },
    {
        "name": "Success Quotes",
        "prompt": "Create an inspiring HD motivational fitness poster for AZHAR FITNESS CENTER. Bold typography with powerful gym success quote. Background shows silhouette of fit athlete. Blue, orange, and chrome gradient design. Modern, energetic style. 9:16 vertical format. Professional graphic design with strong visual impact."
    },
    {
        "name": "Call to Action",
        "prompt": "Create a compelling HD gym promotional poster for AZHAR FITNESS CENTER with 'SEE YOU AT AFC GYM' message. Show welcoming gym interior with modern equipment. Blue, orange, and chrome branding. Energetic atmosphere, professional photography. 9:16 vertical format. Include visual elements that invite action and excitement."
    }
]

# Models
class Member(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    phone: str
    email: Optional[str] = None
    last_checkin: Optional[datetime] = None
    join_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    membership_status: str = "active"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MemberCreate(BaseModel):
    name: str
    phone: str
    email: Optional[str] = None

class MemberUpdate(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    membership_status: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class WhatsAppMessage(BaseModel):
    phone: str
    message: str
    buttons: Optional[List[str]] = None

class InstagramPost(BaseModel):
    id: str
    prompt: str
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    theme: str
    created_at: datetime
    posted: bool

class DashboardStats(BaseModel):
    total_members: int
    active_members: int
    inactive_members: int
    checkins_today: int

# Create FastAPI app
app = FastAPI(title="FITBOT API - Azhar Fitness Center")
api_router = APIRouter(prefix="/api")
security = HTTPBearer()

# Auth dependency
async def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        email = payload.get("email")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Helper functions
async def get_member_by_phone(phone: str):
    """Get member by phone number"""
    return await db.members.find_one({"phone": phone}, {"_id": 0})

async def get_unused_greeting(member_id: str, member_name: str) -> str:
    """Get a greeting that hasn't been used recently"""
    # Get recently used greetings (last 5)
    used = await db.greeting_history.find(
        {"member_id": member_id},
        {"greeting_used": 1, "_id": 0}
    ).sort("used_at", -1).limit(5).to_list(length=5)
    
    used_greetings = [item['greeting_used'] for item in used]
    
    # Find unused greeting
    available = [g for g in GREETINGS if g not in used_greetings]
    if not available:
        # Reset if all used
        await db.greeting_history.delete_many({"member_id": member_id})
        available = GREETINGS
    
    # Select random greeting and format with name
    greeting = random.choice(available).format(name=member_name)
    
    # Save to history
    await db.greeting_history.insert_one({
        "id": str(uuid.uuid4()),
        "member_id": member_id,
        "greeting_used": greeting,
        "used_at": datetime.now(timezone.utc).isoformat()
    })
    
    return greeting

async def send_whatsapp_message(phone: str, message: str, buttons: Optional[List[str]] = None):
    """Send WhatsApp message via Business API"""
    try:
        # Format phone number (remove + if present)
        phone_formatted = phone.replace('+', '').replace(' ', '')
        
        url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_ID}/messages"
        headers = {
            "Authorization": f"Bearer {WHATSAPP_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Build message payload
        if buttons:
            # Interactive message with buttons
            payload = {
                "messaging_product": "whatsapp",
                "to": phone_formatted,
                "type": "interactive",
                "interactive": {
                    "type": "button",
                    "body": {"text": message},
                    "action": {
                        "buttons": [
                            {"type": "reply", "reply": {"id": f"btn_{i}", "title": btn[:20]}}
                            for i, btn in enumerate(buttons[:3])  # WhatsApp limit: 3 buttons
                        ]
                    }
                }
            }
        else:
            # Simple text message
            payload = {
                "messaging_product": "whatsapp",
                "to": phone_formatted,
                "type": "text",
                "text": {"body": message}
            }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            logger.info(f"WhatsApp message sent to {phone}: {response.status_code}")
            return response.json()
    except Exception as e:
        logger.error(f"Failed to send WhatsApp message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"WhatsApp send failed: {str(e)}")

async def generate_ai_response(member_name: str, message: str, member_id: str) -> str:
    """Generate AI response using OpenAI"""
    try:
        chat = LlmChat(
            api_key=OPENAI_API_KEY or EMERGENT_LLM_KEY,
            session_id=f"member_{member_id}",
            system_message=f"You are FITBOT, a high-energy, motivational fitness coach at Azhar Fitness Center. Your personality is dynamic, no-nonsense, and inspiring. Keep responses concise (2-3 sentences max). Use psychological triggers and motivational language. Member's name is {member_name}."
        ).with_model("openai", OPENAI_MODEL)
        
        user_message = UserMessage(text=message)
        response = await chat.send_message(user_message)
        
        return response
    except Exception as e:
        logger.error(f"AI response generation failed: {str(e)}")
        return f"Hey {member_name}! {random.choice(RETENTION_MESSAGES)} Let's get back to training! 💪"

async def generate_instagram_content():
    """Generate daily Instagram content"""
    try:
        # Get last theme used
        last = await db.instagram_posts.find_one({}, sort=[("created_at", -1)])
        
        # Find next theme
        theme_names = [t['name'] for t in INSTAGRAM_THEMES]
        if last and last.get('theme') in theme_names:
            current_idx = theme_names.index(last['theme'])
            next_idx = (current_idx + 1) % len(INSTAGRAM_THEMES)
        else:
            next_idx = 0
        
        theme = INSTAGRAM_THEMES[next_idx]
        
        # Generate image
        image_gen = OpenAIImageGeneration(api_key=OPENAI_API_KEY or EMERGENT_LLM_KEY)
        images = await image_gen.generate_images(
            prompt=theme['prompt'],
            model="gpt-image-1",
            number_of_images=1
        )
        
        image_base64 = None
        if images and len(images) > 0:
            image_base64 = base64.b64encode(images[0]).decode('utf-8')
        
        # Save to database
        post_id = str(uuid.uuid4())
        await db.instagram_posts.insert_one({
            "id": post_id,
            "prompt": theme['prompt'],
            "image_base64": image_base64,
            "theme": theme['name'],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "posted": False
        })
        
        logger.info(f"Generated Instagram content for theme: {theme['name']}")
        return post_id
    except Exception as e:
        logger.error(f"Instagram content generation failed: {str(e)}")
        return None

async def check_inactive_members():
    """Check for inactive members and send retention messages"""
    try:
        # Find members inactive for >3 days
        threshold = datetime.now(timezone.utc) - timedelta(days=3)
        
        inactive_members = await db.members.find({
            "membership_status": "active",
            "$or": [
                {"last_checkin": {"$exists": False}},
                {"last_checkin": None},
                {"last_checkin": {"$lt": threshold.isoformat()}}
            ]
        }).to_list(length=1000)
        
        logger.info(f"Found {len(inactive_members)} inactive members")
        
        for member in inactive_members:
            # Get personalized greeting
            greeting = await get_unused_greeting(member['id'], member['name'])
            
            # Add retention message
            retention_msg = random.choice(RETENTION_MESSAGES)
            full_message = f"{greeting}\n\n{retention_msg}\n\nSee you at AFC Gym! 💪"
            
            # Send WhatsApp with buttons
            await send_whatsapp_message(
                member['phone'],
                full_message,
                buttons=["Let's Go!", "Not Today", "Contact Admin"]
            )
            
            logger.info(f"Sent retention message to {member['name']}")
    except Exception as e:
        logger.error(f"Inactive member check failed: {str(e)}")

# API Routes
@api_router.post("/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    """Admin login"""
    admin = await db.admins.find_one({"email": req.email}, {"_id": 0})
    
    if not admin:
        # Create admin if doesn't exist (first time)
        if req.email == ADMIN_EMAIL and req.password == ADMIN_PASSWORD:
            password_hash = bcrypt.hashpw(req.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            admin_id = str(uuid.uuid4())
            await db.admins.insert_one({
                "id": admin_id,
                "email": req.email,
                "password_hash": password_hash,
                "created_at": datetime.now(timezone.utc).isoformat()
            })
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    else:
        # Verify password
        if not bcrypt.checkpw(req.password.encode('utf-8'), admin['password_hash'].encode('utf-8')):
            raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Generate JWT
    token = jwt.encode(
        {"email": req.email, "exp": datetime.now(timezone.utc) + timedelta(days=7)},
        JWT_SECRET,
        algorithm="HS256"
    )
    
    return LoginResponse(access_token=token)

@api_router.get("/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats(admin: str = Depends(get_current_admin)):
    """Get dashboard statistics"""
    # Total members
    total = await db.members.count_documents({})
    
    # Active members
    active = await db.members.count_documents({"membership_status": "active"})
    
    # Inactive members (>3 days)
    threshold = datetime.now(timezone.utc) - timedelta(days=3)
    inactive = await db.members.count_documents({
        "membership_status": "active",
        "$or": [
            {"last_checkin": {"$exists": False}},
            {"last_checkin": None},
            {"last_checkin": {"$lt": threshold.isoformat()}}
        ]
    })
    
    # Check-ins today
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    checkins_today = await db.members.count_documents({
        "last_checkin": {"$gte": today_start.isoformat()}
    })
    
    return DashboardStats(
        total_members=total,
        active_members=active,
        inactive_members=inactive,
        checkins_today=checkins_today
    )

@api_router.get("/members", response_model=List[Member])
async def get_members(admin: str = Depends(get_current_admin)):
    """Get all members"""
    members = await db.members.find({}, {"_id": 0}).sort("created_at", -1).to_list(length=1000)
    return members

@api_router.post("/members", response_model=Member)
async def create_member(member: MemberCreate, admin: str = Depends(get_current_admin)):
    """Create new member"""
    member_id = str(uuid.uuid4())
    member_data = {
        "id": member_id,
        "name": member.name,
        "phone": member.phone,
        "email": member.email,
        "last_checkin": None,
        "join_date": datetime.now(timezone.utc).isoformat(),
        "membership_status": "active",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.members.insert_one(member_data)
    return member_data

@api_router.put("/members/{member_id}", response_model=Member)
async def update_member(member_id: str, member: MemberUpdate, admin: str = Depends(get_current_admin)):
    """Update member"""
    updates = {}
    
    if member.name:
        updates["name"] = member.name
    if member.phone:
        updates["phone"] = member.phone
    if member.email is not None:
        updates["email"] = member.email
    if member.membership_status:
        updates["membership_status"] = member.membership_status
    
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    await db.members.update_one({"id": member_id}, {"$set": updates})
    updated_member = await db.members.find_one({"id": member_id}, {"_id": 0})
    return updated_member

@api_router.delete("/members/{member_id}")
async def delete_member(member_id: str, admin: str = Depends(get_current_admin)):
    """Delete member"""
    await db.members.delete_one({"id": member_id})
    return {"message": "Member deleted successfully"}

@api_router.post("/members/{member_id}/checkin")
async def member_checkin(member_id: str, admin: str = Depends(get_current_admin)):
    """Record member check-in"""
    await db.members.update_one(
        {"id": member_id},
        {"$set": {"last_checkin": datetime.now(timezone.utc).isoformat()}}
    )
    return {"message": "Check-in recorded"}

@api_router.post("/whatsapp/send")
async def send_whatsapp(msg: WhatsAppMessage, admin: str = Depends(get_current_admin)):
    """Send WhatsApp message"""
    result = await send_whatsapp_message(msg.phone, msg.message, msg.buttons)
    return result

@api_router.post("/whatsapp/test-inactive")
async def test_inactive_check(admin: str = Depends(get_current_admin)):
    """Test inactive member check (manual trigger)"""
    await check_inactive_members()
    return {"message": "Inactive member check completed"}

@api_router.get("/instagram/posts", response_model=List[InstagramPost])
async def get_instagram_posts(admin: str = Depends(get_current_admin)):
    """Get Instagram posts"""
    posts = await db.instagram_posts.find({}, {"_id": 0}).sort("created_at", -1).limit(10).to_list(length=10)
    return posts

@api_router.post("/instagram/generate")
async def generate_instagram(admin: str = Depends(get_current_admin)):
    """Generate Instagram content (manual trigger)"""
    post_id = await generate_instagram_content()
    if post_id:
        return {"message": "Instagram content generated", "post_id": post_id}
    else:
        raise HTTPException(status_code=500, detail="Generation failed")

@api_router.get("/")
async def root():
    return {"message": "FITBOT API - Azhar Fitness Center", "status": "active"}

# Include router
app.include_router(api_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Scheduler setup
scheduler = AsyncIOScheduler()

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    logger.info("Starting FITBOT API...")
    
    # Schedule daily tasks
    # Check inactive members every day at 9 AM
    scheduler.add_job(check_inactive_members, 'cron', hour=9, minute=0)
    
    # Generate Instagram content every day at 10 AM
    scheduler.add_job(generate_instagram_content, 'cron', hour=10, minute=0)
    
    scheduler.start()
    logger.info("FITBOT API started successfully with scheduled tasks")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    client.close()
    scheduler.shutdown()
    logger.info("FITBOT API shutdown")
