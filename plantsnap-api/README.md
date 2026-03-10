# Tech Stack direct comparision #
Tomcat:   runs .war files (Java web apps)
Uvicorn:  runs ASGI apps (FastAPI, Starlette)

Jetty:    lightweight Java server
Uvicorn:  lightweight Python server ✅

Gunicorn: production Python server
          = like Tomcat in production
          
Uvicorn:  development Python server
          = like embedded Jetty for dev ✅


Java microservice:          PlantSnap API:
──────────────────────────────────────────
Spring Boot framework   =   FastAPI
Tomcat/Jetty server     =   Uvicorn
Hibernate ORM           =   SQLAlchemy
Bean Validation         =   Pydantic
Maven/Gradle            =   pip + requirements.txt
application.properties  =   .env file          



#Java world anology#
Java world:                Python world:
─────────────────────────────────────────
Tomcat / Jetty             Uvicorn / Gunicorn
     ↑                          ↑
Application server         Application server
Runs your Java app         Runs your Python app
Listens for HTTP requests  Listens for HTTP requests
Passes to your code        Passes to your code
Returns responses          Returns responses


#Technology Stack#
    iOS app sends:
    POST /feedback
    {
        "predicted_herb": "basil",
        "correct_herb": "chamomile",
        "confidence": 0.45
    }

    Uvicorn:    receives the request
    Pydantic:   validates confidence is a float ✅
    FastAPI:    routes to submit_feedback() function
    SQLAlchemy: stores in SQLite database
    Response:   {"id": 1, "status": "saved"} back to iOS ✅


#FastAPI#

    The web framework — the main thing!

    What it does:
    Lets you create API endpoints in Python
    @app.post("/feedback") ← this is FastAPI!
    
    Without it:
    You'd need to write raw HTTP handling
    Hundreds of lines of complex code ❌

    With it:
    def submit_feedback():... = working endpoint ✅
    
    Analogy:
    Building a house from scratch vs
    using IKEA furniture ← FastAPI is IKEA 😄


#Uvicorn#
    The server — actually RUNS your FastAPI app

    What it does:
    Listens for incoming HTTP requests
    Passes them to your FastAPI code
    Sends responses back

    Without it:
    FastAPI code just sits there doing nothing ❌
    
    With it:
    uvicorn main:app --reload
    = "run the 'app' object in main.py
        and reload when I change code" ✅

    Analogy:
    FastAPI = the restaurant kitchen (your code)
    Uvicorn = the waiter who takes orders
                and brings food out 😄

#SQLAlchemy#
    The database toolkit — talks to SQLite

    What it does:
    Lets you interact with database using Python
    Instead of writing raw SQL queries

    Without it:
    db.execute("INSERT INTO feedback VALUES (?, ?, ?)", ...)
    Complex, error-prone ❌

    With it:
    db.add(feedback_object)
    db.commit()
    Clean Python! ✅

    Analogy:
    SQLAlchemy = translator between
    Python and the database 😄

#Pydantic# 
    Data validation — checks inputs are correct

    What it does:
    Validates that requests have right fields
    Converts types automatically
    
    Without it:
    User sends confidence = "hello" (should be a float!)
    Your code crashes 😱

    With it:
    class FeedbackCreate(BaseModel):
        confidence: float  ← pydantic enforces this!
    
    User sends "hello" → automatic error response ✅
    User sends 0.85   → works perfectly ✅

    Analogy:
    Pydantic = bouncer at the door
    checking everyone has correct ID
    before letting them in 😄

#Deployment# 

AWS/Google Cloud:
  Massive general purpose cloud
  You manage servers, VMs, networking
  Complex configuration
  Expensive

Railway:
  Built their OWN hardware + software
  Abstracts all complexity away
  Push code → it just runs ✅
  Much cheaper


You push code to GitHub
      ↓
Railway detects Python app
      ↓
Builds Docker container automatically
      ↓
Runs on Railway's OWN servers
      ↓
Gives you a public URL instantly ✅ 


#Table explanation#
field            source              purpose
─────────────────────────────────────────────────────
image_id         iOS generates UUID  link feedback to specific scan
predicted_herb   CoreML output       what model predicted
correct_herb     user selection      ground truth label
confidence       CoreML probability  how sure model was
device_id        UIDevice API        anonymous usage analytics
app_version      Bundle info         track which app version had errors


# Architecture decisions #

Current design:
  Server stores: "image_id, predicted=basil, correct=chamomile"
  
  Retraining needs:
  actual_chamomile_photo.jpg + label "chamomile" ✅
  
  Without the image → useless for retraining! ❌

## Solution 1 — Upload image WITH feedback (simplest) ##
When user corrects herb:
  iOS sends BOTH:
    1. The image (as base64 or multipart)
    2. The feedback JSON

Server stores:
    Image → file system or S3
    Feedback → SQLite with image path

Simple! But images take storage space.
For portfolio project → totally fine ✅

## Solution 2 — Store image on device, upload separately ##

Step 1: User scans herb
  iOS saves image locally with UUID
  "550e8400.jpg" saved to device

Step 2: User corrects prediction
  Feedback sent immediately (tiny JSON) ✅

Step 3: Background upload when on WiFi
  Image uploaded separately
  Matched by same UUID ✅

Better for battery + data usage ✅
More complex to implement ❌

## Solution 3 — Only upload low-confidence images ## 

CoreML confidence > 0.8:
  Model was confident → probably right
  Don't bother uploading image ✅
  
CoreML confidence < 0.5:
  Model was unsure → upload image!
  This is where retraining helps most ✅

Saves storage + bandwidth ✅

## For PlantSnap — recommended approach: ## 
Best for your portfolio right now:
  Solution 1 + Solution 3 combined!

  If confidence < 0.7 AND user corrects:
    Upload image + feedback together
    Store image on server
    Use for retraining ✅

  If confidence > 0.7:
    Just store feedback JSON
    Model was confident, less urgent ✅


## The problem with Solution 1+3 at scale: ## 
Portfolio stage (now):
  10-100 users
  Images stored on Railway server ✅
  SQLite database ✅
  Works perfectly!

Production stage (1000+ users):
  Railway server disk fills up 😱
  SQLite gets slow with concurrent writes 😱
  Images lost if server restarts 😱
  Can't scale horizontally 😱    

## The actual production architecture ##

 iOS App
   ↓
FastAPI Server (Railway/AWS)
   ↓              ↓
PostgreSQL      S3 Bucket
(metadata)      (actual images) 

## Why separate storage for images ## 

Database (PostgreSQL):
  Stores TEXT data efficiently
  Fast queries and relationships
  NOT designed for binary files ❌

Object Storage (S3/GCS):
  Designed specifically for files/images
  Infinitely scalable ✅
  Cheap ($0.023/GB on AWS) ✅
  CDN built in ✅
  Never loses files ✅
  
Rule of thumb:
  Metadata → database
  Files    → object storage


## The production flow ## 
Step 1: User corrects herb (confidence < 0.7)
   iOS has image in memory

Step 2: iOS requests upload URL from API
   GET /feedback/upload-url?image_id=550e8400
   
Step 3: FastAPI generates presigned S3 URL
   "https://s3.amazonaws.com/plantsnap/
    feedback/550e8400.jpg?token=xyz&expires=300"
   Returns URL to iOS

Step 4: iOS uploads DIRECTLY to S3
   PUT image → S3 (bypasses your server!)
   Fast, cheap, no server bottleneck ✅

Step 5: iOS sends feedback JSON to API
   POST /feedback
   {image_id, predicted, correct, confidence}
   NO image in this request! ✅

Step 6: FastAPI stores in PostgreSQL:
   {image_id, s3_path, predicted, correct...}  


## Why presinged URLs are brilliant ## 
Without presigned URLs:
  iOS → FastAPI → S3
  Image goes through YOUR server
  Bandwidth costs $$$ 💸
  Server bottleneck ❌
  Slow ❌

With presigned URLs:
  iOS → S3 directly! ✅
  FastAPI just generates the permission
  Server never touches the image ✅
  Fast ✅
  Cheap ✅
  Scales infinitely ✅

Same pattern used by:
  Instagram photo uploads ✅
  Dropbox file uploads ✅
  WhatsApp media ✅   


## The complete production stack: ## 
Current (portfolio):          Production:
──────────────────────────────────────────────
Railway server               AWS/GCP/Railway
SQLite                       PostgreSQL (RDS)
Local file system            S3 / GCS
Direct image upload          Presigned URLs
Single server                Load balanced
No CDN                       CloudFront CDN  


# .env #
Local (no .env):
  USE_S3 = False
  Images saved to feedback_images/ folder
  Works perfectly for development ✅

Railway (with env vars set):
  USE_S3 = True
  Images saved to S3 automatically
  Same code, different behaviour! ✅

No code changes needed between environments!
Just environment variables change ✅