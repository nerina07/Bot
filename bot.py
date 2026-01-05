
training_data = [
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("hey", "greeting"),
    ("how can you help", "greeting"),

    ("where is my order", "order_status"),
    ("track my order", "order_status"),
    ("order not delivered", "order_status"),

    ("i want a refund", "refund"),
    ("return my product", "refund"),
    ("refund status", "refund"),

    ("how can i contact support", "contact"),
    ("customer care number", "contact"),
    ("talk to agent", "contact"),

    ("bye", "goodbye"),
    ("thank you", "goodbye")
    ("ok","goodbye")
]


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Training data
X = [text for text, label in training_data]
y = [label for text, label in training_data]

# Convert text to vectors
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

# Response dictionary
responses = {
    "greeting": "Hello! How can I help you today?",
    "order_status": "Please share your order ID to track your order.",
    "refund": "Your refund request has been initiated. It will be processed within 5–7 days.",
    "contact": "You can contact our support team at support@example.com.",
    "goodbye": "Thank you for contacting us. Have a great day!"
}

# Chatbot function
def chatbot(user_input):
    user_vec = vectorizer.transform([user_input])
    intent = model.predict(user_vec)[0]
    return responses[intent]

# Chat loop
print("Customer Support Chatbot (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break
    print("Bot:", chatbot(user_input))
