# main.py
from src.core.pipeline import Pipeline

if __name__ == "__main__":
    pipeline = Pipeline()

    while True:
        prompt = input("You: ")
        try:
            response = pipeline.run(user_id="user_123", prompt=prompt)
            print("RockLM:", response)
        except Exception as e:
            print("[Blocked]", str(e))
