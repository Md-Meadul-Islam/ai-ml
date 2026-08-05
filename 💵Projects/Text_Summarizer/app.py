from pydantic import BaseModel
from fastapi import FastAPI, Request

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

#device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

templates = Jinja2Templates(directory=".")

#Input Schema for input
class DialogueInput(BaseModel):
    dialogue:str

def clean_data(text):
    text = re.sub(r"\n\r", " ", text)#lines
    text = re.sub(r"\s+", " ", text)#spaces
    text = re.sub(r"<.*?>", " ", text)#html tags
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue):
  dialogue = clean_data(dialogue)

  #tokenize
  inputs = tokenizer(
      dialogue,
      max_length=512,
      padding="max_length",
      truncation=True,
      return_tensors="pt"
  )
  #generate the summary => token ids
  model.to(device)
  targets = model.generate(
      input_ids = inputs['input_ids'],
      attention_mask = inputs['attention_mask'],
      max_length = 150,
      num_beams = 4,
      early_stopping = True
  )
  summary = tokenizer.decode(targets[0], skip_special_tokens=True)

  return summary


@app.post("/summarize/")
def summarize(dialogue_input:DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class= HTMLResponse)
def home(request:Request):
    return templates.TemplateResponse(name="index.html",request= {"request": request})