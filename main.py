import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, List

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from pydantic import BaseModel
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.textanalytics import TextAnalyticsClient

import re

load_dotenv()

app = FastAPI(title="Azure Speech Semester Project")

"""
pipeline phase 1
"""

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
ALLOWED_EXTENSIONS = {".wav", ".mp3"}
ALLOWED_CONTENT_TYPES = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3"}


def _build_audio_config(file_path: str, ext: str) -> speechsdk.audio.AudioConfig:
    """Builds an Azure AudioConfig for either a WAV or compressed MP3 file."""
    #TODO the mp3 part doesnt work 
    if ext == ".mp3":
        # MP3 requires a compressed PushAudioInputStream
        compressed_format = speechsdk.audio.AudioStreamFormat(
            compressed_stream_format=speechsdk.AudioStreamContainerFormat.MP3
        )
        push_stream = speechsdk.audio.PushAudioInputStream(
            stream_format=compressed_format
        )

        with open(file_path, "rb") as audio_file:
            push_stream.write(audio_file.read())
        push_stream.close()

        return speechsdk.audio.AudioConfig(stream=push_stream)

    # Default: WAV file input
    return speechsdk.audio.AudioConfig(filename=file_path)


def _run_continuous_recognition(file_path: str, ext: str) -> Dict[str, Any]:
    """Runs Azure continuous speech recognition synchronously inside a worker thread."""
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")

    if not speech_key or not speech_region:
        raise RuntimeError("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set.")

    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key, region=speech_region
    )

    # Request Detailed JSON output to capture word-level confidence scores
    speech_config.output_format = speechsdk.OutputFormat.Detailed

    # Configure candidate languages (up to 4 allowed in continuous recognition)
    auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=["en-US" ]#, "es-ES", "fr-FR", "de-DE"] #TODO dont need for testing
    )

    audio_config = _build_audio_config(file_path, ext)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_detect_config,
        audio_config=audio_config,
    )

    transcript_segments: List[str] = []
    words_with_confidence: List[Dict[str, Any]] = []
    detected_language: str = "unknown"
    total_speech_ticks: int = 0
    segment_confidences: List[float] = []
    error_message: str | None = None

    # Use an asyncio/threading event to block until the audio stream finishes
    import threading

    done_event = threading.Event()

    def on_recognized(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        nonlocal detected_language, total_speech_ticks, segment_confidences
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            transcript_segments.append(evt.result.text)

            # Extract detected language
            auto_detect_result = speechsdk.AutoDetectSourceLanguageResult(evt.result)
            if auto_detect_result.language:
                detected_language = auto_detect_result.language

            # Track duration in 100-nanosecond ticks
            total_speech_ticks += evt.result.duration

            # Parse detailed JSON payload for per-word confidence scores
            raw_json = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult
            )
            if raw_json:
                try:
                    detailed = json.loads(raw_json)
                    nbest = detailed.get("NBest", [])
                    if nbest:
                        top_result = nbest[0]
                        if "Confidence" in top_result:
                            segment_confidences.append(float(top_result["Confidence"]))
                    if nbest and isinstance(nbest[0].get("Words"), list):
                        for w in nbest[0]["Words"]:
                            words_with_confidence.append(
                                {
                                    "word": w.get("Word"),
                                    "confidence": w.get("Confidence"),
                                    "offset": w.get("Offset"),
                                    "duration": w.get("Duration"),
                                }
                            )
                except json.JSONDecodeError:
                    pass

    def on_canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
        nonlocal error_message
        if evt.reason == speechsdk.CancellationReason.Error:
            error_message = (
                f"Azure Speech Error ({evt.error_code}): {evt.error_details}"
            )
        done_event.set()

    def on_session_stopped(evt: speechsdk.SessionEventArgs) -> None:
        done_event.set()

    # Connect event handlers
    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_stopped.connect(on_session_stopped)

    # Start continuous recognition asynchronously
    recognizer.start_continuous_recognition_async().get()

    # Wait until session_stopped or canceled fires
    done_event.wait()

    # Stop recognition cleanly
    recognizer.stop_continuous_recognition_async().get()

    if error_message:
        raise RuntimeError(error_message)
    
    # Convert ticks to seconds (uses total spoken audio; swap to max_end_ticks for span up to last word)
    # 10_000_000 ticks per second
    duration_seconds = round(total_speech_ticks / 10_000_000, 2)

    # Calculate overall confidence (average of segment confidences)
    overall_confidence = (
        round(sum(segment_confidences) / len(segment_confidences), 4)
        if segment_confidences
        else 0.0
    )

    #TODO missing several sections - durationSeconds, confidence
    return {
        "transcript": " ".join(transcript_segments).strip(),
        "duration_seconds": duration_seconds,
        "confidence": overall_confidence,
        "language": detected_language,
        "words": words_with_confidence,
    }

"""
TODO format for json response:
"transcript": "Meeting starts at 3pm. Please review the proposal before
then.",
"language": "en-US",
"duration_seconds": 4.2,
"confidence": 0.94,
"words": [
{ "word": "Meeting", "offset": 0.0, "duration": 0.32, "confidence": 0.98 },
...]

"""

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)) -> Dict[str, Any]: #-> JSONResponse:
    # 1. Validate file extension & MIME type
    filename = audio.filename or ""
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS and audio.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only WAV and MP3 files are supported.",
        )

    # Default extension fallback if uploaded via raw MIME type without extension
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".mp3" if audio.content_type in {"audio/mpeg", "audio/mp3"} else ".wav"

    temp_file_path: str | None = None

    try:
        # 2. Save upload temporarily to disk while enforcing the 25 MB limit
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            temp_file_path = tmp.name
            bytes_written = 0

            while chunk := await audio.read(1024 * 1024):  # Read in 1 MB chunks
                bytes_written += len(chunk)
                if bytes_written > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the maximum allowed size of 25 MB.",
                    )
                tmp.write(chunk)

        # 3. Offload blocking Azure SDK event loop to a background thread
        result = await asyncio.to_thread(_run_continuous_recognition, temp_file_path, ext)
        #return JSONResponse(status_code=200, content=result) #TODO TESTING
        return result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        # 4. Always delete the temporary file when finished
        await audio.close()
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as cleanup_err:
                print(f"Warning: Failed to remove temp file {temp_file_path}: {cleanup_err}")



"""
pipeline phase 2
"""

# Initialize the Azure Text Analytics Client
endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
key = os.getenv("AZURE_LANGUAGE_KEY")

if not endpoint or not key:
    raise RuntimeError("Missing AZURE_LANGUAGE_ENDPOINT or AZURE_LANGUAGE_KEY environment variables.")

client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))


# Request Model
#TODO do this for the other one also 
class TranscriptRequest(BaseModel):
    transcript: str


@app.post("/analyze")
async def analyze(payload: TranscriptRequest) -> Dict[str, Any]:
    #TODO change so this takes the "transcript" field of the prev json blob
    if not payload.transcript.strip():
        raise HTTPException(status_code=400, detail="The 'transcript' field cannot be empty.")

    documents = [payload.transcript]

    # 1. Key Phrase Extraction
    kp_response = client.extract_key_phrases(documents=documents)[0]
    if kp_response.is_error:
        raise HTTPException(status_code=500, detail=f"Key Phrase Error: {kp_response.error.message}")
    key_phrases = kp_response.key_phrases

    # 2. Named Entity Recognition (NER)
    ner_response = client.recognize_entities(documents=documents)[0]
    if ner_response.is_error:
        raise HTTPException(status_code=500, detail=f"NER Error: {ner_response.error.message}")
    entities = [
        {
            "text": entity.text,
            "category": entity.category,
            "subcategory": entity.subcategory,
            "confidence_score": entity.confidence_score
        }
        for entity in ner_response.entities
    ]

    # 3. Sentiment Analysis
    sentiment_response = client.analyze_sentiment(documents=documents)[0]
    if sentiment_response.is_error:
        raise HTTPException(status_code=500, detail=f"Sentiment Error: {sentiment_response.error.message}")
    sentiment = {
        "overall": sentiment_response.sentiment,
        "confidence_scores": {
            "positive": sentiment_response.confidence_scores.positive,
            "neutral": sentiment_response.confidence_scores.neutral,
            "negative": sentiment_response.confidence_scores.negative,
        }
    }

    # 4. Linked Entities (Graceful fallback for post-Sept 2025 resources)
    linked_entities_result: Dict[str, Any] = {
        "status": "supported",
        "entities": [],
        "note": None
    }

    try:
        link_response = client.recognize_linked_entities(documents=documents)[0]
        if link_response.is_error:
            linked_entities_result["status"] = "skipped"
            linked_entities_result["note"] = (
                f"Entity Linking skipped: {link_response.error.message} "
                "(Note: Microsoft stopped onboarding new resources to Entity Linking in Sept 2025)."
            )
        else:
            linked_entities_result["entities"] = [
                {
                    "name": entity.name,
                    "url": entity.url,
                    "data_source": entity.data_source,
                    "matches": [
                        {"text": m.text, "confidence_score": m.confidence_score}
                        for m in entity.matches
                    ]
                }
                for entity in link_response.entities
            ]
    except HttpResponseError as e:
        linked_entities_result["status"] = "skipped"
        linked_entities_result["note"] = (
            f"Entity Linking rejected by Azure resource (HTTP {e.status_code}). "
            "New resources created after September 2025 do not have access to this deprecated feature."
        )

    # Return unified JSON structure for frontend panels
    return {
        "key_topics": key_phrases,
        "named_entities": entities,
        "sentiment": sentiment,
        "linked_entities": linked_entities_result
    }
    #TODO change object type to JsonResponse for consistency 
    # format: return JSONResponse(status_code=200, content=result)

"""
pipeline - process
"""

@app.post("/process")
async def process(audio: UploadFile = File(...)) -> JSONResponse:
    audio_results = await transcribe_audio(audio)
    payload = TranscriptRequest(**audio_results)
    analysis = await analyze(payload)   
    action_items = await extract_action_items(audio_results.get("transcript"), analysis) 

    #generate tts summary
    """
    sample:
    "Your memo mentions 3 key topics: project deadline, budget approval, and Q3
    review.
    The overall tone is neutral. I detected 2 people, 1 organization, and 1 date."
    """
    topics = analysis.get("key_topics", [])
    tts_summary = f"Your memo mentions {len(topics)} key topics: "
    for i in range(0, len(topics) - 1):
        tts_summary += topics[i] + ", "
    
    tts_summary += "and " + topics[len(topics) - 1] + ". "
    overall_tone = analysis.get("sentiment", {}).get("overall", "neutral")
    tts_summary += f"The overall tone is {overall_tone}. "
    #TODO not sure how to do the people and org thing

    #return all results as a json summary
    result = {
        "pipeline_step_1": audio_results,
        "pipeline_step_2": analysis,
        "action_items": action_items,
        "tts_summary": tts_summary
    }
    return JSONResponse(status_code=200, content=result)


"""
extract action items
"""

# --- Response Schemas ---
class ActionItem(BaseModel):
    task: str
    assignee: Optional[str] = None
    due_date: Optional[str] = None
    related_topics: List[str] = []
    source_sentence: str

# --- Action Item Extraction Logic ---
ACTION_PATTERNS = re.compile(
    r"\b(need to|needs to|have to|has to|must|should|will|going to|"
    r"follow up|send|email|call|schedule|review|update|create|prepare|"
    r"submit|check|finalize|reach out|set up|complete|finish|share|assign)\b",
    re.IGNORECASE,
)

@app.post("/extract-action-items")
async def extract_action_items( transcript: str, nlp_data: Dict[str, Any] ) -> List[Dict[str, Any]]:
    """
    Combines transcript sentences with Azure AI Language entities & key phrases
    to isolate actionable tasks, assignees, and due dates.
    """
    if not transcript.strip():
        return []

    # Split transcript into sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if s.strip()]

    entities = nlp_data.get("entities", [])
    key_phrases = nlp_data.get("key_topics", [])

    # Group recognized entities by category
    people = [e["text"] for e in entities if e.get("category") == "Person"]
    dates = [e["text"] for e in entities if e.get("category") == "DateTime"]

    action_items: List[ActionItem] = []

    for sentence in sentences:
        if ACTION_PATTERNS.search(sentence):
            sentence_lower = sentence.lower()

            # Match entities and key phrases appearing within this action sentence
            matched_assignee = next(
                (p for p in people if p.lower() in sentence_lower), None
            )
            matched_due_date = next(
                (d for d in dates if d.lower() in sentence_lower), None
            )
            matched_topics = [
                kp for kp in key_phrases if kp.lower() in sentence_lower
            ]

            action_items.append(
                ActionItem(
                    task=sentence,
                    assignee=matched_assignee,
                    due_date=matched_due_date,
                    related_topics=matched_topics,
                    source_sentence=sentence,
                ).dict()
            )

    return action_items