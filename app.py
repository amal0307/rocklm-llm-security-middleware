import gradio as gr
from src.core.pipeline import Pipeline
from src.core.logger import get_logger

pipeline = Pipeline()
logger = get_logger("pipeline")

def inference(user_id, prompt):
    try:
        return pipeline.run(user_id, prompt)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return f"Error: {e}"

iface = gr.Interface(
    fn=inference,
    inputs=["text", "text"],
    outputs="text",
    title="RockLM Security Pipeline",
    description="Enter your user_id and prompt",
    flagging_mode="never"
)

if __name__ == "__main__":
    iface.launch()
