import gradio as gr
from llmlingua import PromptCompressor

compressor = PromptCompressor(model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank")

def compress_prompt(context, rate):
    result = compressor.compress_prompt(context, rate=float(rate))
    return result['compressed_prompt'], f"Tokens originaux : {result['origin_tokens']} -> Tokens compressés : {result['compressed_tokens']}"

demo = gr.Interface(
    fn=compress_prompt,
    inputs=[
        gr.Textbox(lines=10, label="Prompt ou Texte Long à épuré"),
        gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.1, label="Taux de conservation (0.5 = garde 50%)")
    ],
    outputs=[
        gr.Textbox(label="Prompt Compressé"),
        gr.Textbox(label="Statistiques de Tokens")
    ],
    title="Épureur de Prompt LLMLingua-2"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
