from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Cargar el modelo y tokenizer desde Hugging Face
model_id = "bigcode/starcoder"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

app = FastAPI()

class CodeInput(BaseModel):
    code: str

@app.post("/corregir")
async def corregir_codigo(data: CodeInput):
    prompt = (
        "Corrige el siguiente código, explica los errores de forma sencilla "
        "y sugiere mejoras siguiendo buenas prácticas:\n\n"
        f"{data.code}\n\n"
    )
    output = generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)[0]["generated_text"]
    return {"respuesta": output}
