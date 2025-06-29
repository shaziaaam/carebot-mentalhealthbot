import streamlit as st 
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# --- Konfigurasi Halaman Streamlit ---
# st.set_page_config() HARUS menjadi perintah Streamlit pertama setelah import.
st.set_page_config(page_title="Mental Health Chatbot", page_icon="🤖")

# --- 1. Load dataset Mental_Health_FAQ dari HuggingFace ---
# Gunakan st.cache_data agar data hanya dimuat sekali saat aplikasi berjalan
@st.cache_data
def load_data():
    try:
        df_loaded = pd.read_csv("hf://datasets/tolu07/Mental_Health_FAQ/Mental_Health_FAQ.csv")
        if 'Questions' not in df_loaded.columns or 'Answers' not in df_loaded.columns:
            st.error("Error: Dataset must contain 'Questions' and 'Answers' columns.")
            raise ValueError("Dataset must contain 'Questions' and 'Answers' columns.")
        return df_loaded
    except Exception as e:
        st.error(f"Error loading dataset: {e}. Falling back to dummy data.")
        data = {
            'Questions': [
                "What is anxiety?", "How to deal with stress?", "Symptoms of depression",
                "I feel tired all the time", "How to relax?", "What is mental health?",
                "Tips for better sleep", "How to manage anger?", "Signs of burnout",
                "I feel overwhelmed", "Ways to cope with sadness", "What is therapy?"
            ],
            'Answers': [
                "Anxiety is a feeling of worry, nervousness, or unease, typically about an event or something with an uncertain outcome.",
                "Managing stress involves techniques like mindfulness, exercise, and setting realistic goals. Try taking breaks.",
                "Symptoms of depression can include persistent sadness, loss of interest, changes in appetite or sleep, and feelings of worthlessness.",
                "Feeling tired can be a sign of many things, from lack of sleep to stress. Ensure you're getting enough rest and managing your energy.",
                "Mental health refers to our emotional, psychological, and social well-being. It affects how we think, feel, and act.",
                "For better sleep, establish a regular sleep schedule, create a relaxing bedtime routine, and avoid caffeine before bed.",
                "To manage anger, try deep breathing, taking a timeout, or expressing your feelings assertively but calmly.",
                "Signs of burnout include chronic fatigue, cynicism, and reduced professional efficacy. Take time for self-care.",
                "Feeling overwhelmed is common. Break down tasks into smaller steps, prioritize, and don't hesitate to ask for help.",
                "Coping with sadness can involve allowing yourself to feel emotions, talking to someone, engaging in hobbies, or seeking professional help.",
                "Therapy is a way to treat mental health issues by talking to a trained professional, helping you understand and cope with your thoughts and feelings."
            ]
        }
        return pd.DataFrame(data)

df = load_data()

# --- 2. TF-IDF Vectorization ---
@st.cache_resource
def initialize_vectorizer(dataframe):
    vectorizer_obj = TfidfVectorizer()
    faq_vectors_obj = vectorizer_obj.fit_transform(dataframe['Questions'])
    return vectorizer_obj, faq_vectors_obj

vectorizer, faq_vectors = initialize_vectorizer(df)

# --- 3. Chatbot FAQ Response ---
def chatbot_response(user_input, vectorizer_obj, faq_vectors_obj, dataframe):
    user_vector = vectorizer_obj.transform([user_input])
    similarities = cosine_similarity(user_vector, faq_vectors_obj)
    idx = similarities.argmax()

    if similarities[0, idx] < 0.2:
        return "I'm not sure I understand your question. Could you rephrase it or ask about a general mental health topic? Please use English."

    return dataframe.iloc[idx]['Answers']

# --- 4. Stress Level Classification ---
def classify_stress_level(user_input):
    input_lower = user_input.lower()

    keywords = {
        "low": [
            "slightly stressed", "a bit worried", "uneasy", "restless", "on edge", "nervous",
            "a little anxious", "mildly pressured", "irritable", "impatient", "fretful",
            "apprehensive", "concerned", "on tenterhooks", "a little bothered", "disquieted",
            "edgy", "uptight", "jittery", "slightly agitated",
            "feel tired", "a bit tired", "sleepy", "feeling drained a little", "just tired",
            "slight headache", "minor ache", "feeling off", "a bit overwhelmed",
            "light pressure", "some pressure", "distracted a bit", "not quite focused",
            "mildly frustrated", "a little annoyed", "slightly irritable", "feeling a bit down",
            "can't focus much", "a bit unfocused", "okay", "fine", "good", "alright", "normal",
            "feeling well", "doing well", "no stress", "relaxed", "calm", "peaceful"
        ],
        "optimum": [
            "moderately stressed", "somewhat anxious", "frustrated", "drained", "easily distracted",
            "difficulty concentrating slightly", "feeling the pressure", "a bit overwhelmed", "moody",
            "agitated", "restless", "impatient", "on edge", "tense", "worried",
            "preoccupied", "a little down", "unsettled", "querulous", "irritable",
            "busy day", "heavy workload", "tight deadline", "under pressure", "challenging",
            "feeling responsible", "has responsibilities", "need to finish", "pushing myself",
            "feeling the grind", "productive stress", "focused on tasks", "energized",
            "motivated", "alert", "aware", "sharp", "responsive", "productive", "motivated"
        ],
        "moderate": [
            "stressed", "anxious", "worried", "overwhelmed", "tired", "drained", "frustrated",
            "irritable", "nervous", "restless", "difficulty concentrating", "sleep disturbances",
            "feeling tense", "on edge", "agitated", "preoccupied", "down", "discouraged",
            "impatient", "easily upset", "on a short fuse", "trouble relaxing", "mentally fatigued",
            "emotionally drained",
            "exhausted", "constantly tired", "burnout", "can't cope", "heavy burden",
            "significant pressure", "feeling stuck", "lost interest", "lack of motivation",
            "feeling withdrawn", "avoiding people", "trouble sleeping", "insomnia",
            "headaches often", "stomach issues", "muscle tension", "frequent mood swings",
            "stressed out", "quite anxious", "struggling", "depressed", "sad"
        ],
        "high": [
            "very stressed", "highly anxious", "panicked", "overwhelmed", "unable to cope",
            "severe difficulty concentrating", "significant sleep problems", "intensely frustrated",
            "extremely irritable", "constantly agitated", "feeling trapped", "loss of motivation",
            "helpless", "desperate", "on the verge of breakdown", "consumed by worry",
            "unable to think clearly", "severe mood swings", "social withdrawal", "increased heart rate",
            "muscle tension", "headaches",
            "panic attack", "constant fear", "racing thoughts", "can't stop worrying",
            "feeling of dread", "overthinking", "shaking", "shortness of breath",
            "chest tightness", "dizzy", "nauseous", "trembling", "sweating", "restless nights",
            "nightmares", "avoiding everything", "crying spells", "really bad", "terrible", "miserable",
            "breakdown", "can't handle", "suffering"
        ],
        "very high": [
            "severely stressed", "debilitating anxiety", "panic attacks", "completely overwhelmed",
            "hopeless", "helpless", "suicidal thoughts", "worthless", "unable to function",
            "crushed", "paralyzed by stress", "intense emotional pain", "constant state of fear",
            "disconnection from reality", "inability to perform daily tasks", "extreme fatigue",
            "loss of appetite or overeating", "isolation", "feeling like dying", "persistent sadness",
            "self-harm thoughts", "can't get out of bed", "complete shutdown", "no hope",
            "life isn't worth it", "nothing matters", "empty inside", "feeling numb",
            "terrified", "hyperventilating", "loss of control", "screaming inside", "want to die",
            "end it all", "give up", "can't go on", "self-harm", "harm myself", "no purpose"
        ]
    }
    for level, words in keywords.items():
        if any(word in input_lower for word in words):
            return level
    return random.choice(["moderate", "optimum"])

# --- Streamlit UI Components ---
st.title("Mental Health Chatbot 🤖")
st.markdown("Hello! I'm here to listen and try to help you.")
st.markdown("You can tell me how you feel or ask about mental health topics.")
st.markdown("---")

# Inisialisasi session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "last_bot_prompt" not in st.session_state:
    st.session_state.last_bot_prompt = ""

# --- Alur Chat Utama ---
if not st.session_state.user_name:
    user_name_input = st.text_input("Bot: First, what should I call you?", key="name_input")
    if user_name_input:
        st.session_state.user_name = user_name_input.strip()
        if not st.session_state.user_name:
            st.session_state.user_name = "User"

        initial_greeting = f"Nice to meet you, {st.session_state.user_name.capitalize()}! Let's begin. How can I help you today?"
        st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
        st.session_state.last_bot_prompt = f"How can I help you today, {st.session_state.user_name.capitalize()}?"

        st.rerun()

else: # Jika nama sudah diatur
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input(f"Your message, {st.session_state.user_name.capitalize()}:")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        input_lower = user_input.lower().strip()
        bot_response_parts = []

        if st.session_state.last_bot_prompt and ("yes" in input_lower or "yeah" in input_lower or "yup" in input_lower or "no" in input_lower or "nope" in input_lower):
            if "yes" in input_lower or "yeah" in input_lower or "yup" in input_lower:
                if "additional resources" in st.session_state.last_bot_prompt:
                    bot_response_parts.append("Certainly. Your campus might have student counseling services. You can also look for professional psychologists or try meditation apps like Calm or Headspace. Do you want more specific details about any type of resource?")
                elif "other tips" in st.session_state.last_bot_prompt:
                    bot_response_parts.append("Great! Some general tips for maintaining mental health include: getting enough sleep (7-9 hours), regular exercise, eating nutritious food, limiting screen time, and making time for hobbies. Which one interests you most?")
                elif "explore your feelings further" in st.session_state.last_bot_prompt:
                    bot_response_parts.append("Please tell me more about what's making you feel that way. Sometimes, just talking about it can bring some relief.")
                elif "more specific details" in st.session_state.last_bot_prompt:
                    bot_response_parts.append("Campus counseling offers direct and often free support. Professional psychologists provide more in-depth therapy. Apps can give you tools for meditation or mood tracking. What do you need most right now?")
                elif "interests you most" in st.session_state.last_bot_prompt:
                    bot_response_parts.append("I'm ready to provide more information about that!")
                else:
                    bot_response_parts.append("Okay, let's continue.")
                st.session_state.last_bot_prompt = ""
            elif "no" in input_lower or "nope" in input_lower:
                bot_response_parts.append("Okay, no problem. Is there anything else I can help you with or another topic you'd like to discuss?")
                st.session_state.last_bot_prompt = ""

            if "what's your name" in input_lower and "what's your name" not in bot_response_parts[0].lower():
                bot_response_parts.append("I'm a chatbot designed to support your mental health. You can call me your Mental Assistant.")
            elif "how are you" in input_lower and "how are you" not in bot_response_parts[0].lower():
                bot_response_parts.append("I don't have feelings like humans, but I'm here and ready to help you! Thanks for asking.")

            final_bot_response = "\n\n".join(bot_response_parts)

        else:
            faq_answer = chatbot_response(user_input, vectorizer, faq_vectors, df)
            stress_level = classify_stress_level(user_input)

            bot_response_parts.append(f"Personal Response: {faq_answer}")
            bot_response_parts.append(f"Detected Stress Level: {stress_level.capitalize()}")

            if stress_level in ["high", "very high"]:
                st.session_state.last_bot_prompt = f"I understand you're feeling {stress_level} right now, {st.session_state.user_name.capitalize()}. Remember, seeking professional help is a strong act, not a weakness. Would you like to know about additional resources for this situation?"
            elif stress_level == "moderate":
                st.session_state.last_bot_prompt = f"Thanks for sharing, {st.session_state.user_name.capitalize()}. Feeling {stress_level} is common. Remember, it's important to take care of yourself. Is there anything specific you'd like to explore further about your feelings?"
            elif stress_level in ["low", "optimum"]:
                st.session_state.last_bot_prompt = f"Glad to hear you're feeling {stress_level}, {st.session_state.user_name.capitalize()}! Keep up the great work on your mental well-being. Would you like to know any other tips to stay on this positive track?"
            else:
                st.session_state.last_bot_prompt = f"Is there anything else you'd like to talk about or ask, {st.session_state.user_name.capitalize()}?"

            bot_response_parts.append(f"Bot: {st.session_state.last_bot_prompt}")
            final_bot_response = "\n\n".join(bot_response_parts)

        with st.chat_message("assistant"):
            st.markdown(final_bot_response)
        st.session_state.messages.append({"role": "assistant", "content": final_bot_response})
