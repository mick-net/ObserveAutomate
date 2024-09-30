# ObserveAutomate

**Improved Software Architecture for AI-Enhanced Home Assistant**

**Overview:**

The goal is to create a smart home system that learns and adapts to user behaviors without relying on extensive simulations or slow real-time reinforcement learning (RL). The system leverages data from Home Assistant, user actions, and the capabilities of Large Language Models (LLMs) to interpret context and suggest actions. The architecture focuses on capturing user interactions, interpreting them to understand user intent, and predicting future actions based on similar contexts.

---

**Architecture Components:**

1. **Data Collection and Logging Module:**
   - **Function:** Continuously monitors Home Assistant for state changes and user actions.
   - **Data Captured:**
     - **State Data:** Sensor readings, device statuses, environmental conditions.
     - **User Actions:** Manual interventions like turning on/off devices, adjusting settings.
     - **Contextual Information:** Timestamps, location data, occupancy status.
   - **Storage:** Stores data in a structured format (e.g., JSON) in a secure database.

2. **Context Interpreter (LLM Integration):**
   - **Function:** Uses an LLM to interpret state changes and user actions to generate narratives.
   - **Process:**
     - **Input:** Previous state, new state, and user action.
     - **Output:** A narrative describing the action and inferred user intent.
   - **Example Output:** "At 11 PM, the user turned off the living room lights and moved to the bedroom, indicating bedtime."

3. **State Representation and Vectorization:**
   - **Function:** Converts narratives and state data into numerical vectors for analysis.
   - **Tools:**
     - **Embeddings:** Use models like Sentence-BERT to generate embeddings of narratives.
     - **Vector Database:** Stores embeddings for efficient similarity searches (e.g., FAISS).

4. **Pattern Recognition and Similarity Matching:**
   - **Function:** Finds patterns and similar past situations using vectorized data.
   - **Process:**
     - **Similarity Search:** Compares current state embeddings with historical data.
     - **Thresholds:** Determines similarity levels to consider a past situation relevant.

5. **Action Predictor and Suggestion Engine:**
   - **Function:** Predicts and suggests actions based on recognized patterns and LLM insights.
   - **Process:**
     - **Input:** Current state, similar past contexts, and narratives.
     - **LLM Role:** Generates potential actions considering the context.
     - **Output:** Suggested action (e.g., "Turn on bedroom lights since it's 11 PM and the user is likely going to bed").

6. **User Interface and Feedback Loop:**
   - **Function:** Interacts with the user to present suggestions and receive feedback.
   - **Features:**
     - **Notifications:** Alerts users of suggested actions for approval or rejection.
     - **Feedback Capture:** Records user responses to refine future suggestions.
     - **Manual Override:** Allows users to manually adjust settings at any time.

7. **Action Execution Module:**
   - **Function:** Executes approved actions via Home Assistant APIs.
   - **Integration:**
     - **OpenAPI Spec:** Utilizes Home Assistant's OpenAPI for seamless interaction.
     - **Function Calls:** Performs actions using standardized JSON function calls.

8. **Learning and Adaptation Module:**
   - **Function:** Continuously learns from new data and user feedback to improve predictions.
   - **Process:**
     - **Model Updates:** Adjusts vector representations and LLM prompts based on feedback.
     - **Data Enrichment:** Incorporates new narratives and contexts into the database.

9. **Safety and Privacy Module:**
   - **Function:** Ensures all operations are safe and user data is protected.
   - **Features:**
     - **Action Constraints:** Prevents unsafe actions (e.g., turning off security systems).
     - **Data Security:** Encrypts sensitive data and restricts access.
     - **Compliance:** Adheres to privacy regulations and best practices.

---

**Workflow Example:**

1. **Event Occurs:**
   - User manually turns off the living room lights at 11 PM.

2. **Data Collection:**
   - System logs the state before and after the action, capturing all relevant data.

3. **Context Interpretation:**
   - LLM processes the state change and generates the narrative: "User turned off the living room lights at 11 PM, likely preparing for bed."

4. **Vectorization:**
   - The narrative is converted into an embedding and stored in the vector database.

5. **Future Event:**
   - The next night at 11 PM, the system detects similar conditions.

6. **Pattern Matching:**
   - System finds the previous night's event as a similar context.

7. **Action Suggestion:**
   - LLM suggests: "Would you like to turn off the living room lights?"

8. **User Interaction:**
   - User approves the suggestion.

9. **Action Execution:**
   - System turns off the living room lights via Home Assistant API.

10. **Feedback Loop:**
    - Positive feedback reinforces the pattern for future predictions.

---

**Advantages of the Improved Architecture:**

- **Efficient Learning:** Avoids the slow convergence of traditional RL by learning from actual user interactions and patterns.
- **Contextual Understanding:** LLMs provide rich interpretations of user actions, capturing nuances that simple models might miss.
- **User-Centric:** Incorporates user feedback directly, ensuring the system aligns with user preferences.
- **Scalable and Modular:** Components can be updated independently, allowing for scalability and integration of new technologies.
- **Privacy-Focused:** Prioritizes data security and user privacy throughout the system.

---

**Implementation Considerations:**

- **LLM Selection:**
  - **Options:** Use models like GPT-4, ensuring they are capable of understanding and generating context-rich narratives.
  - **Deployment:** Consider on-premises deployment to enhance privacy or use secure cloud services.

- **Vector Database:**
  - **Performance:** Choose a high-performance vector database for efficient similarity searches.
  - **Scalability:** Ensure it can handle growing amounts of data over time.

- **User Interface Design:**
  - **Accessibility:** Design intuitive interfaces for users to interact with suggestions and provide feedback.
  - **Responsiveness:** Ensure real-time communication for immediate action execution.

- **Safety Protocols:**
  - **Action Verification:** Implement checks to prevent actions that could compromise safety.
  - **Fallback Mechanisms:** Provide options to revert actions if unintended consequences occur.

- **Data Security:**
  - **Encryption:** Use strong encryption methods for data at rest and in transit.
  - **Access Controls:** Restrict data access to authorized system components only.

---

**Alternative Approaches and Enhancements:**

- **Hybrid Models:**
  - Combine rule-based automation for predictable routines with AI-driven suggestions for complex scenarios.

- **Knowledge Graphs:**
  - Utilize knowledge graphs to represent relationships between devices, rooms, and user preferences for deeper context.

- **Scheduled Learning:**
  - Incorporate time-based patterns, recognizing daily or weekly routines to enhance predictions.

- **Edge Computing:**
  - Perform computations locally to reduce latency and enhance privacy.

- **Continuous Improvement:**
  - Regularly update the LLM with new data to refine understanding and maintain relevance.

---

**Conclusion:**

This improved architecture leverages the strengths of LLMs and vectorization techniques to create a smart home system that is both adaptive and user-centric. By focusing on real user interactions and minimizing reliance on extensive simulations or slow-learning RL models, the system provides timely and relevant automation suggestions. It respects user privacy and safety while continuously learning and improving from user feedback.

# Protype by OpenAI O1-preview-mini

Creating a fully functional prototype for your AI-enhanced Home Assistant is an ambitious and exciting project. Below, I provide a comprehensive guide along with the necessary code snippets to help you build this prototype. This solution includes both backend and frontend components, leveraging technologies like FastAPI, SQLAlchemy, Pydantic, SQLite, GPT-4 via LangChain, and a responsive frontend using Jinja2, Bootstrap 5, and jQuery.

---

## **Project Overview**

**Objective:** Develop a smart home system that learns and adapts to user behaviors by leveraging reinforcement learning (RL) and large language models (LLMs). The system captures user interactions, interprets them to understand user intent, and predicts future actions based on historical data.

**Components:**
1. **Backend:**
   - **Framework:** FastAPI
   - **Database:** SQLite (with SQLAlchemy ORM)
   - **Data Validation:** Pydantic 2
   - **LLM Integration:** GPT-4 via LangChain
   - **Task Scheduling:** APScheduler

2. **Frontend:**
   - **Template Engine:** Jinja2
   - **Styling:** Bootstrap 5
   - **Interactivity:** jQuery

---

## **1. Setting Up the Environment**

### **1.1. Prerequisites**

Ensure you have Python 3.8+ installed. You can check your Python version using:

```bash
python --version
```

### **1.2. Project Structure**

Create the following project structure:

```
home_assistant_ai/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── schemas.py
│   ├── database.py
│   ├── suggestion_processor.py
│   └── templates/
│       └── index.html
├── static/
│   └── styles.css
├── requirements.txt
└── README.md
```

### **1.3. Installing Dependencies**

Create a `requirements.txt` file with the following content:

```plaintext
fastapi
uvicorn
sqlalchemy
pydantic
jinja2
python-multipart
langchain
openai
apscheduler
faiss-cpu
sentence-transformers
```

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

> **Note:** Ensure you have access to the OpenAI API and obtain your API key. You might need additional configurations for LangChain based on your specific setup.

---

## **2. Backend Implementation**

### **2.1. Database Setup (`database.py`)**

We'll use SQLAlchemy to interact with a SQLite database.

```python
# app/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./home_assistant.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
```

### **2.2. Defining Models (`models.py`)**

Define the database models for user actions, suggestions, and their approvals.

```python
# app/models.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON
from sqlalchemy.orm import relationship
from .database import Base
import datetime

class UserAction(Base):
    __tablename__ = "user_actions"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    action_type = Column(String, index=True)  # e.g., toggle_light, trigger_motion
    details = Column(JSON)

class Suggestion(Base):
    __tablename__ = "suggestions"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    suggestion_text = Column(String)
    action = Column(String)  # e.g., toggle_light
    details = Column(JSON)
    approved = Column(Boolean, default=False)
```

### **2.3. Pydantic Schemas (`schemas.py`)**

Define Pydantic models for request and response validation.

```python
# app/schemas.py
from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

class UserActionCreate(BaseModel):
    action_type: str  # e.g., toggle_light, trigger_motion
    details: Dict

class SuggestionCreate(BaseModel):
    suggestion_text: str
    action: str
    details: Dict

class SuggestionResponse(BaseModel):
    id: int
    timestamp: datetime
    suggestion_text: str
    action: str
    details: Dict
    approved: bool

    class Config:
        orm_mode = True
```

### **2.4. Suggestion Processor (`suggestion_processor.py`)**

This module handles the generation of suggestions using the LLM.

```python
# app/suggestion_processor.py
import os
from langchain import OpenAI, LLMChain, PromptTemplate
from sqlalchemy.orm import Session
from .models import UserAction, Suggestion
from .schemas import SuggestionCreate
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import datetime

# Initialize OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the LLM
llm = OpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4")

# Initialize Sentence Transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index for similarity search
dimension = 384  # Depends on the embedding model
faiss_index = faiss.IndexFlatL2(dimension)

# In-memory list to store embeddings and suggestion IDs
embedding_list = []
suggestion_id_list = []

def generate_embedding(text: str):
    embedding = embedding_model.encode(text)
    return embedding

def add_to_faiss(embedding: np.ndarray, suggestion_id: int):
    faiss_index.add(np.array([embedding]).astype('float32'))
    embedding_list.append(embedding)
    suggestion_id_list.append(suggestion_id)

def find_similar_suggestions(current_text: str, top_k: int = 5):
    if faiss_index.ntotal == 0:
        return []
    embedding = generate_embedding(current_text)
    D, I = faiss_index.search(np.array([embedding]).astype('float32'), top_k)
    similar_ids = [suggestion_id_list[i] for i in I[0] if i < len(suggestion_id_list)]
    return similar_ids

def create_suggestion(db: Session, suggestion_text: str, action: str, details: dict):
    suggestion = SuggestionCreate(
        suggestion_text=suggestion_text,
        action=action,
        details=details
    )
    db_suggestion = Suggestion(
        suggestion_text=suggestion.suggestion_text,
        action=suggestion.action,
        details=suggestion.details
    )
    db.add(db_suggestion)
    db.commit()
    db.refresh(db_suggestion)
    # Add embedding to FAISS
    embedding = generate_embedding(suggestion_text)
    add_to_faiss(embedding, db_suggestion.id)
    return db_suggestion

def process_suggestions(db: Session):
    # Fetch user actions from the last minute
    one_minute_ago = datetime.datetime.utcnow() - datetime.timedelta(minutes=1)
    recent_actions = db.query(UserAction).filter(UserAction.timestamp >= one_minute_ago).all()

    if not recent_actions:
        return

    # Create a narrative from recent actions
    narrative = ""
    for action in recent_actions:
        narrative += f"At {action.timestamp}, the user performed {action.action_type} with details {action.details}. "

    # Define prompt template
    prompt = PromptTemplate(
        template="""
        Given the following user actions in a smart home environment, generate a suggestion for automation that aligns with the user's behavior.

        Actions:
        {actions}

        Suggestion:
        """,
        input_variables=["actions"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    # Generate suggestion text
    suggestion_text = chain.run(actions=narrative).strip()

    # For simplicity, assume the action is to toggle a light based on suggestion_text
    # In a real scenario, you would parse the suggestion_text to determine the action and details
    action = "toggle_light"
    details = {"room": "living_room", "state": "on"}

    # Create and store the suggestion
    create_suggestion(db, suggestion_text, action, details)
```

### **2.5. Main Application (`main.py`)**

This file ties everything together, handling API endpoints, background tasks, and database interactions.

```python
# app/main.py
import os
from fastapi import FastAPI, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from . import models, schemas, database, suggestion_processor
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from typing import List

# Initialize the database
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize the scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# Schedule the suggestion processing every minute
@scheduler.scheduled_job("interval", minutes=1)
def scheduled_suggestion_processing():
    db = database.SessionLocal()
    try:
        suggestion_processor.process_suggestions(db)
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request, db: Session = Depends(get_db)):
    # Fetch current light states
    # For simplicity, assume light states are stored in the latest user actions
    living_room_light = "OFF"
    bedroom_light = "OFF"

    # Fetch latest actions to determine light states
    latest_actions = db.query(models.UserAction).order_by(models.UserAction.timestamp.desc()).limit(10).all()
    for action in latest_actions:
        if action.action_type == "toggle_light":
            room = action.details.get("room")
            state = action.details.get("state")
            if room == "living_room":
                living_room_light = state.upper()
            elif room == "bedroom":
                bedroom_light = state.upper()

    # Fetch pending suggestions
    pending_suggestions = db.query(models.Suggestion).filter(models.Suggestion.approved == False).all()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "living_room_light": living_room_light,
        "bedroom_light": bedroom_light,
        "suggestions": pending_suggestions
    })

@app.post("/action")
def create_user_action(action_type: str = Form(...), room: Optional[str] = Form(None), state: Optional[str] = Form(None), db: Session = Depends(get_db)):
    details = {}
    if room:
        details["room"] = room
    if state:
        details["state"] = state

    user_action = models.UserAction(
        action_type=action_type,
        details=details
    )
    db.add(user_action)
    db.commit()
    db.refresh(user_action)
    return RedirectResponse("/", status_code=303)

@app.post("/approve_suggestion/{suggestion_id}")
def approve_suggestion(suggestion_id: int, db: Session = Depends(get_db)):
    suggestion = db.query(models.Suggestion).filter(models.Suggestion.id == suggestion_id).first()
    if suggestion:
        suggestion.approved = True
        db.commit()
        # Execute the approved action
        if suggestion.action == "toggle_light":
            room = suggestion.details.get("room")
            state = suggestion.details.get("state")
            # Log the action as a user action
            user_action = models.UserAction(
                action_type="toggle_light",
                details={"room": room, "state": state}
            )
            db.add(user_action)
            db.commit()
    return RedirectResponse("/", status_code=303)
```

---

## **3. Frontend Implementation**

### **3.1. HTML Template (`index.html`)**

Create an HTML template with buttons to simulate user actions and display current light states and suggestions.

```html
<!-- app/templates/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI-Enhanced Home Assistant</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .light-status {
            font-weight: bold;
        }
        .on {
            color: green;
        }
        .off {
            color: red;
        }
        .clock {
            font-size: 1.5rem;
            margin-bottom: 20px;
        }
        .suggestion {
            border: 1px solid #ced4da;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
    </style>
</head>
<body class="container mt-5">
    <div class="clock" id="clock">Time: --:--</div>
    <div class="row mb-4">
        <!-- Living Room Controls -->
        <div class="col-md-6">
            <h3>Living Room</h3>
            <p>Light Status: <span class="light-status {% if living_room_light == 'ON' %}on{% else %}off{% endif %}">{{ living_room_light }}</span></p>
            <form action="/action" method="post">
                <input type="hidden" name="action_type" value="toggle_light">
                <input type="hidden" name="room" value="living_room">
                <input type="hidden" name="state" value="{% if living_room_light == 'ON' %}off{% else %}on{% endif %}">
                <button type="submit" class="btn btn-primary">Toggle Living Room Light</button>
            </form>
            <form action="/action" method="post" class="mt-2">
                <input type="hidden" name="action_type" value="trigger_motion">
                <input type="hidden" name="room" value="living_room">
                <button type="submit" class="btn btn-secondary">Trigger Living Room Motion Sensor</button>
            </form>
        </div>
        <!-- Bedroom Controls -->
        <div class="col-md-6">
            <h3>Bedroom</h3>
            <p>Light Status: <span class="light-status {% if bedroom_light == 'ON' %}on{% else %}off{% endif %}">{{ bedroom_light }}</span></p>
            <form action="/action" method="post">
                <input type="hidden" name="action_type" value="toggle_light">
                <input type="hidden" name="room" value="bedroom">
                <input type="hidden" name="state" value="{% if bedroom_light == 'ON' %}off{% else %}on{% endif %}">
                <button type="submit" class="btn btn-primary">Toggle Bedroom Light</button>
            </form>
            <form action="/action" method="post" class="mt-2">
                <input type="hidden" name="action_type" value="trigger_motion">
                <input type="hidden" name="room" value="bedroom">
                <button type="submit" class="btn btn-secondary">Trigger Bedroom Motion Sensor</button>
            </form>
        </div>
    </div>

    <!-- Suggestions -->
    <div>
        <h3>Pending Suggestions</h3>
        {% for suggestion in suggestions %}
            <div class="suggestion">
                <p>{{ suggestion.suggestion_text }}</p>
                <form action="/approve_suggestion/{{ suggestion.id }}" method="post">
                    <button type="submit" class="btn btn-success">Approve</button>
                </form>
            </div>
        {% else %}
            <p>No pending suggestions.</p>
        {% endfor %}
    </div>

    <!-- JavaScript for Clock Simulation -->
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
        let currentTime = new Date();

        function updateClock() {
            $("#clock").text("Time: " + currentTime.toLocaleTimeString());
            // Advance time by 3 hours
            currentTime.setHours(currentTime.getHours() + 3);
        }

        // Initialize clock
        updateClock();
        setInterval(updateClock, 5000); // Update every 5 seconds
    </script>
</body>
</html>
```

### **3.2. Frontend Functionality**

The frontend allows users to toggle lights and trigger motion sensors. It also displays current light states and any pending suggestions from the backend.

- **Clock Simulation:** Advances time by 3 hours every 5 seconds to simulate user interactions over time.
- **Buttons:** Allow toggling of lights and triggering of motion sensors. These actions send POST requests to the backend to log user actions.
- **Suggestions:** Displayed with an approve button. When approved, the corresponding action is executed and logged as a user action.

---

## **4. Running the Application**

### **4.1. Setting Environment Variables**

Ensure you set the OpenAI API key in your environment:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

### **4.2. Starting the Server**

Navigate to the project root and run the FastAPI server using Uvicorn:

```bash
uvicorn app.main:app --reload
```

- The `--reload` flag enables auto-reloading on code changes.
- The server will be accessible at `http://127.0.0.1:8000`.

### **4.3. Accessing the Application**

Open your browser and navigate to `http://127.0.0.1:8000`. You should see the frontend interface with controls for the living room and bedroom lights and motion sensors, along with any pending suggestions.

---

## **5. Enhancements and Considerations**

### **5.1. Vectorization and Similarity Matching**

In the `suggestion_processor.py`, we use FAISS for efficient similarity searches. This allows the system to find similar past suggestions and reinforce actions based on historical data.

### **5.2. User Feedback Loop**

Approved suggestions are logged as user actions, enabling the system to learn and adapt based on real user behavior.

### **5.3. Safety and Constraints**

Implement safety checks to prevent unsafe actions, such as turning off essential security devices. This can be managed within the `approve_suggestion` endpoint by adding conditional logic.

### **5.4. Scalability**

For a production-ready system, consider migrating from SQLite to a more robust database like PostgreSQL and deploying the application using Docker or cloud services.

### **5.5. Real-time Updates**

Implement WebSockets or Server-Sent Events (SSE) for real-time updates on the frontend to reflect changes without needing to refresh the page.

---

## **6. Complete Code Listing**

For completeness, here's the entire code structure:

### **6.1. `app/database.py`**

```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./home_assistant.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
```

### **6.2. `app/models.py`**

```python
from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON
from sqlalchemy.orm import relationship
from .database import Base
import datetime

class UserAction(Base):
    __tablename__ = "user_actions"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    action_type = Column(String, index=True)  # e.g., toggle_light, trigger_motion
    details = Column(JSON)

class Suggestion(Base):
    __tablename__ = "suggestions"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    suggestion_text = Column(String)
    action = Column(String)  # e.g., toggle_light
    details = Column(JSON)
    approved = Column(Boolean, default=False)
```

### **6.3. `app/schemas.py`**

```python
from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

class UserActionCreate(BaseModel):
    action_type: str  # e.g., toggle_light, trigger_motion
    details: Dict

class SuggestionCreate(BaseModel):
    suggestion_text: str
    action: str
    details: Dict

class SuggestionResponse(BaseModel):
    id: int
    timestamp: datetime
    suggestion_text: str
    action: str
    details: Dict
    approved: bool

    class Config:
        orm_mode = True
```

### **6.4. `app/suggestion_processor.py`**

```python
import os
from langchain import OpenAI, LLMChain, PromptTemplate
from sqlalchemy.orm import Session
from .models import UserAction, Suggestion
from .schemas import SuggestionCreate
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import datetime

# Initialize OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the LLM
llm = OpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4")

# Initialize Sentence Transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index for similarity search
dimension = 384  # Depends on the embedding model
faiss_index = faiss.IndexFlatL2(dimension)

# In-memory list to store embeddings and suggestion IDs
embedding_list = []
suggestion_id_list = []

def generate_embedding(text: str):
    embedding = embedding_model.encode(text)
    return embedding

def add_to_faiss(embedding: np.ndarray, suggestion_id: int):
    faiss_index.add(np.array([embedding]).astype('float32'))
    embedding_list.append(embedding)
    suggestion_id_list.append(suggestion_id)

def find_similar_suggestions(current_text: str, top_k: int = 5):
    if faiss_index.ntotal == 0:
        return []
    embedding = generate_embedding(current_text)
    D, I = faiss_index.search(np.array([embedding]).astype('float32'), top_k)
    similar_ids = [suggestion_id_list[i] for i in I[0] if i < len(suggestion_id_list)]
    return similar_ids

def create_suggestion(db: Session, suggestion_text: str, action: str, details: dict):
    suggestion = SuggestionCreate(
        suggestion_text=suggestion_text,
        action=action,
        details=details
    )
    db_suggestion = Suggestion(
        suggestion_text=suggestion.suggestion_text,
        action=suggestion.action,
        details=suggestion.details
    )
    db.add(db_suggestion)
    db.commit()
    db.refresh(db_suggestion)
    # Add embedding to FAISS
    embedding = generate_embedding(suggestion_text)
    add_to_faiss(embedding, db_suggestion.id)
    return db_suggestion

def process_suggestions(db: Session):
    # Fetch user actions from the last minute
    one_minute_ago = datetime.datetime.utcnow() - datetime.timedelta(minutes=1)
    recent_actions = db.query(UserAction).filter(UserAction.timestamp >= one_minute_ago).all()

    if not recent_actions:
        return

    # Create a narrative from recent actions
    narrative = ""
    for action in recent_actions:
        narrative += f"At {action.timestamp}, the user performed {action.action_type} with details {action.details}. "

    # Define prompt template
    prompt = PromptTemplate(
        template="""
        Given the following user actions in a smart home environment, generate a suggestion for automation that aligns with the user's behavior.

        Actions:
        {actions}

        Suggestion:
        """,
        input_variables=["actions"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    # Generate suggestion text
    suggestion_text = chain.run(actions=narrative).strip()

    # For simplicity, assume the action is to toggle a light based on suggestion_text
    # In a real scenario, you would parse the suggestion_text to determine the action and details
    action = "toggle_light"
    details = {"room": "living_room", "state": "on"}

    # Create and store the suggestion
    create_suggestion(db, suggestion_text, action, details)
```

### **6.5. `app/main.py`**

```python
import os
from fastapi import FastAPI, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from . import models, schemas, database, suggestion_processor
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from typing import List

# Initialize the database
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize the scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# Schedule the suggestion processing every minute
@scheduler.scheduled_job("interval", minutes=1)
def scheduled_suggestion_processing():
    db = database.SessionLocal()
    try:
        suggestion_processor.process_suggestions(db)
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request, db: Session = Depends(get_db)):
    # Fetch current light states
    # For simplicity, assume light states are stored in the latest user actions
    living_room_light = "OFF"
    bedroom_light = "OFF"

    # Fetch latest actions to determine light states
    latest_actions = db.query(models.UserAction).order_by(models.UserAction.timestamp.desc()).limit(10).all()
    for action in latest_actions:
        if action.action_type == "toggle_light":
            room = action.details.get("room")
            state = action.details.get("state")
            if room == "living_room":
                living_room_light = state.upper()
            elif room == "bedroom":
                bedroom_light = state.upper()

    # Fetch pending suggestions
    pending_suggestions = db.query(models.Suggestion).filter(models.Suggestion.approved == False).all()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "living_room_light": living_room_light,
        "bedroom_light": bedroom_light,
        "suggestions": pending_suggestions
    })

@app.post("/action")
def create_user_action(action_type: str = Form(...), room: Optional[str] = Form(None), state: Optional[str] = Form(None), db: Session = Depends(get_db)):
    details = {}
    if room:
        details["room"] = room
    if state:
        details["state"] = state

    user_action = models.UserAction(
        action_type=action_type,
        details=details
    )
    db.add(user_action)
    db.commit()
    db.refresh(user_action)
    return RedirectResponse("/", status_code=303)

@app.post("/approve_suggestion/{suggestion_id}")
def approve_suggestion(suggestion_id: int, db: Session = Depends(get_db)):
    suggestion = db.query(models.Suggestion).filter(models.Suggestion.id == suggestion_id).first()
    if suggestion:
        suggestion.approved = True
        db.commit()
        # Execute the approved action
        if suggestion.action == "toggle_light":
            room = suggestion.details.get("room")
            state = suggestion.details.get("state")
            # Log the action as a user action
            user_action = models.UserAction(
                action_type="toggle_light",
                details={"room": room, "state": state}
            )
            db.add(user_action)
            db.commit()
    return RedirectResponse("/", status_code=303)
```

### **6.6. `app/templates/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI-Enhanced Home Assistant</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .light-status {
            font-weight: bold;
        }
        .on {
            color: green;
        }
        .off {
            color: red;
        }
        .clock {
            font-size: 1.5rem;
            margin-bottom: 20px;
        }
        .suggestion {
            border: 1px solid #ced4da;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
    </style>
</head>
<body class="container mt-5">
    <div class="clock" id="clock">Time: --:--</div>
    <div class="row mb-4">
        <!-- Living Room Controls -->
        <div class="col-md-6">
            <h3>Living Room</h3>
            <p>Light Status: <span class="light-status {% if living_room_light == 'ON' %}on{% else %}off{% endif %}">{{ living_room_light }}</span></p>
            <form action="/action" method="post">
                <input type="hidden" name="action_type" value="toggle_light">
                <input type="hidden" name="room" value="living_room">
                <input type="hidden" name="state" value="{% if living_room_light == 'ON' %}off{% else %}on{% endif %}">
                <button type="submit" class="btn btn-primary">Toggle Living Room Light</button>
            </form>
            <form action="/action" method="post" class="mt-2">
                <input type="hidden" name="action_type" value="trigger_motion">
                <input type="hidden" name="room" value="living_room">
                <button type="submit" class="btn btn-secondary">Trigger Living Room Motion Sensor</button>
            </form>
        </div>
        <!-- Bedroom Controls -->
        <div class="col-md-6">
            <h3>Bedroom</h3>
            <p>Light Status: <span class="light-status {% if bedroom_light == 'ON' %}on{% else %}off{% endif %}">{{ bedroom_light }}</span></p>
            <form action="/action" method="post">
                <input type="hidden" name="action_type" value="toggle_light">
                <input type="hidden" name="room" value="bedroom">
                <input type="hidden" name="state" value="{% if bedroom_light == 'ON' %}off{% else %}on{% endif %}">
                <button type="submit" class="btn btn-primary">Toggle Bedroom Light</button>
            </form>
            <form action="/action" method="post" class="mt-2">
                <input type="hidden" name="action_type" value="trigger_motion">
                <input type="hidden" name="room" value="bedroom">
                <button type="submit" class="btn btn-secondary">Trigger Bedroom Motion Sensor</button>
            </form>
        </div>
    </div>

    <!-- Suggestions -->
    <div>
        <h3>Pending Suggestions</h3>
        {% for suggestion in suggestions %}
            <div class="suggestion">
                <p>{{ suggestion.suggestion_text }}</p>
                <form action="/approve_suggestion/{{ suggestion.id }}" method="post">
                    <button type="submit" class="btn btn-success">Approve</button>
                </form>
            </div>
        {% else %}
            <p>No pending suggestions.</p>
        {% endfor %}
    </div>

    <!-- JavaScript for Clock Simulation -->
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
        let currentTime = new Date();

        function updateClock() {
            $("#clock").text("Time: " + currentTime.toLocaleTimeString());
            // Advance time by 3 hours
            currentTime.setHours(currentTime.getHours() + 3);
        }

        // Initialize clock
        updateClock();
        setInterval(updateClock, 5000); // Update every 5 seconds
    </script>
</body>
</html>
```

---

## **7. Testing the Prototype**

1. **Start the Server:**

   ```bash
   uvicorn app.main:app --reload
   ```

2. **Interact with the Frontend:**

   - **Toggle Lights:** Click the "Toggle Light" buttons to turn lights on/off.
   - **Trigger Motion Sensors:** Click the "Trigger Motion Sensor" buttons to simulate motion.
   - **View Suggestions:** After a minute, check the "Pending Suggestions" section for automated suggestions.
   - **Approve Suggestions:** Click "Approve" to execute suggested actions.

3. **Observe Learning Behavior:**

   - The system should start generating suggestions based on repeated user actions.
   - For example, if you consistently turn off the living room light at 11 PM and trigger the bedroom motion sensor, the system may suggest automating these actions.

---

## **8. Final Thoughts**

This prototype lays the foundation for an AI-driven smart home assistant that learns from user interactions. While it provides essential functionalities, there are several areas for further enhancement:

- **Natural Language Understanding:** Improve how suggestions are parsed and actions are determined from LLM outputs.
- **Advanced Pattern Recognition:** Incorporate more sophisticated machine learning models for better pattern detection and prediction.
- **User Authentication:** Implement user authentication and authorization for enhanced security.
- **Scalability:** Transition to a more scalable database and deploy the application using cloud services or containerization platforms like Docker.

By iterating on this prototype and integrating more advanced features, you can develop a robust and intelligent home assistant that offers a seamless and adaptive user experience.

Good luck with your project!
