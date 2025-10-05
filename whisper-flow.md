
# Repo: whisper-flow


## File: README.md

````md
<div align="center">
<h1 align="center"> Whisper Flow </h1> 
<h3>Real-Time Transcription Using OpenAI Whisper</br></h3>
<img src="https://img.shields.io/badge/Progress-100%25-red"> <img src="https://img.shields.io/badge/Feedback-Welcome-green">
</br>
</br>
<kbd>
<img src="https://github.com/dimastatz/whisper-flow/blob/da8b67c6180566b987854b2fb94670fee92e6682/docs/imgs/whisper-flow.png?raw=true" width="256px"> 
</kbd>
</div>

## About The Project

### OpenAI Whisper 
OpenAI [Whisper](https://github.com/openai/whisper) is a versatile speech recognition model designed for general use. Trained on a vast and varied audio dataset, Whisper can handle tasks such as multilingual speech recognition, speech translation, and language identification. It is commonly used for batch transcription, where you provide the entire audio or video file to Whisper, which then converts the speech into text. This process is not done in real-time; instead, Whisper processes the files and returns the text afterward, similar to handing over a recording and receiving the transcript later.

### Whisper Flow 
Using Whisper Flow, you can generate real-time transcriptions for your media content. Unlike batch transcriptions, where media files are uploaded and processed, streaming media is delivered to Whisper Flow in real time, and the service returns a transcript immediately.

### What is Streaming
Streaming content is sent as a series of sequential data packets, or 'chunks,' which Whisper Flow transcribes on the spot. The benefits of using streaming over batch processing include the ability to incorporate real-time speech-to-text functionality into your applications and achieving faster transcription times. However, this speed may come at the expense of accuracy in some cases.

### Stream Windowing
In scenarios involving time-streaming, it's typical to perform operations on data within specific time frames known as temporal windows. One common approach is using the [tumbling window](https://learn.microsoft.com/en-us/azure/stream-analytics/stream-analytics-window-functions#tumbling-window) technique, which involves gathering events into segments until a certain condition is met.

<div align="center">
<img src="https://github.com/dimastatz/whisper-flow/blob/main/docs/imgs/streaming.png?raw=true"> 
<div>Tumbling Window</div>
</div><br/>

### Streaming Results
Whisper Flow splits the audio stream into segments based on natural speech patterns, like speaker changes or pauses. The transcription is sent back as a series of events, with each response containing more transcribed speech until the entire segment is complete.

| Transcript                                    | EndTime  | IsPartial |
| :-------------------------------------------- | :------: | --------: |
| Reality                                       |   0.55   | True      |
| Reality is created                            |   1.05   | True      |
| Reality is created by the                     |   1.50   | True      |
| Reality is created by the mind                |   2.15   | True      |
| Reality is created by the mind                |   2.65   | False     |
| we can                                        |   3.05   | True      |
| we can change                                 |   3.45   | True      |
| we can change reality                         |   4.05   | True      |
| we can change reality by changing             |   4.45   | True      |
| we can change reality by changing our mind    |   5.05   | True      |
| we can change reality by changing our mind    |   5.55   | False     |

### Benchmarking
The evaluation metrics for comparing the performance of Whisper Flow are Word Error Rate (WER) and latency. Latency is measured as the time between two subsequent partial results, with the goal of achieving sub-second latency. We're not starting from scratch, as several quality benchmarks have already been performed for different ASR engines. I will rely on the research article ["Benchmarking Open Source and Paid Services for Speech to Text"](https://www.frontiersin.org/articles/10.3389/fdata.2023.1210559/full) for guidance. For benchmarking the current implementation of Whisper Flow, I use [LibriSpeech](https://www.openslr.org/12).

```bash
| Partial | Latency | Result |

True  175.47  when we took
True  185.14  When we took her.
True  237.83  when we took our seat.
True  176.42  when we took our seats.
True  198.59  when we took our seats at the
True  186.72  when we took our seats at the
True  210.04  when we took our seats at the breakfast.
True  220.36  when we took our seats at the breakfast table.
True  203.46  when we took our seats at the breakfast table.
True  242.63  When we took our seats at the breakfast table, it will
True  237.41  When we took our seats at the breakfast table, it was with
True  246.36  When we took our seats at the breakfast table, it was with the
True  278.96  When we took our seats at the breakfast table, it was with the feeling.
True  285.03  When we took our seats at the breakfast table, it was with the feeling of being.
True  295.39  When we took our seats at the breakfast table, it was with the feeling of being no
True  270.88  When we took our seats at the breakfast table, it was with the feeling of being no longer
True  320.43  When we took our seats at the breakfast table, it was with the feeling of being no longer looked
True  303.66  When we took our seats at the breakfast table, it was with the feeling of being no longer looked upon.
True  470.73  When we took our seats at the breakfast table, it was with the feeling of being no longer
True  353.25  When we took our seats at the breakfast table, it was with the feeling of being no longer looked upon as connected.
True  345.74  When we took our seats at the breakfast table, it was with the feeling of being no longer looked upon as connected in any way.
True  368.66  When we took our seats at the breakfast table, it was with the feeling of being no longer looked upon as connected in any way with the
True  400.25  When we took our seats at the breakfast table, it was with the feeling of being no longer looked upon as connected in any way with this case.
True  382.71  When we took our seats at the breakfast table, it was with the feeling of being no longer looked upon as connected in any way with this case.
False 405.02  When we took our seats at the breakfast table, it was with the feeling of being no longer looked upon as connected in any way with this case.
```

When running this benchmark on a MacBook Air with an [M1 chip and 16GB of RAM](https://support.apple.com/en-il/111883#:~:text=Testing%20conducted%20by%20Apple%20in,to%208%20clicks%20from%20bottom.), we achieve impressive performance metrics. The latency is consistently well below 500ms, ensuring real-time responsiveness. Additionally, the word error rate is around 7%, demonstrating the accuracy of the transcription.

```bash
Latency Stats:
count     26.000000
mean     275.223077
std       84.525695
min      154.700000
25%      205.105000
50%      258.620000
75%      339.412500
max      470.700000
```

### How To Use it

#### As a Web Server
To run WhisperFlow as a web server, start by cloning the repository to your local machine.
```bash
git clone https://github.com/dimastatz/whisper-flow.git
```
Then navigate to WhisperFlow folder, create a local venv with all dependencies and run the web server on port 8181.
```bash
cd whisper-flow
./run.sh -local
source .venv/bin/activate
./run.sh -benchmark
```

#### As a Python Package
Set up a WebSocket endpoint for real-time transcription by retrieving the transcription model and creating asynchronous functions for transcribing audio chunks and sending JSON responses. Manage the WebSocket connection by continuously processing incoming audio data. Handle terminate exception to stop the session and close the connection if needed.

Start with installing whisper python package

```bash
pip install whisperflow
```

Now import whsiperflow and transcriber modules

```Python
import whisperflow.streaming as st
import whisperflow.transcriber as ts

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    model = ts.get_model()

    async def transcribe_async(chunks: list):
        return await ts.transcribe_pcm_chunks_async(model, chunks)

    async def send_back_async(data: dict):
        await websocket.send_json(data)

    try:
        await websocket.accept()
        session = st.TrancribeSession(transcribe_async, send_back_async)

        while True:
            data = await websocket.receive_bytes()
            session.add_chunk(data)
    except Exception as exception:
        await session.stop()
        await websocket.close()
```
#### Roadmap
- [X] Release v1.0-RC - Includes transcription streaming implementation.
- [X] Release v1.1 - Bug fixes and implementation of the most requested changes.
- [ ] Release v1.2 - Prepare the package for integration with the py-speech package.

````

## File: setup.py

````py
from pathlib import Path
from setuptools import setup
from whisperflow import __version__
from pkg_resources import parse_requirements


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='whisperflow',
    version=__version__,
    url='https://github.com/dimastatz/whisper-flow',
    author='Dima Statz',
    author_email='dima.statz@gmail.com',
    py_modules=['whisperflow'],
    python_requires=">=3.8",
    install_requires=[
        str(r)
        for r in parse_requirements(
            Path(__file__).with_name("requirements.txt").open()
        )
    ],
    description='WhisperFlow: Real-Time Transcription Powered by OpenAI Whisper',
    long_description = long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    package_data={'': ['static/*']},
)

````

## File: __init__.py

````py

````

## File: __init__.py

````py

````

## File: test_audio.py

````py
""" test chat room """

import queue
import asyncio
import pytest
import numpy as np
import whisperflow.audio.microphone as mic


@pytest.mark.asyncio
async def test_capture_mic():
    """test capturing microphone"""
    stop_event = asyncio.Event()
    audio_chunks = queue.Queue()

    async def stop_capturing():
        await asyncio.sleep(0.1)
        stop_event.set()

    await asyncio.gather(mic.capture_audio(audio_chunks, stop_event), stop_capturing())
    assert stop_event.is_set()
    assert not audio_chunks.empty()


def test_is_silent():
    """test silence detection"""
    silence_threshold = 500
    # Create a silent audio buffer (all zeros)
    silent_data = np.zeros(1024, dtype=np.int16).tobytes()
    assert mic.is_silent(silent_data), "Silent data should be detected as silent"

    # Create a loud audio buffer (above threshold)
    loud_data = (np.ones(1024, dtype=np.int16) * (silence_threshold + 1000)).tobytes()
    assert not mic.is_silent(loud_data), "Loud data should not be detected as silent"

    # Create a borderline case (right at the threshold)
    threshold_data = (np.ones(1024, dtype=np.int16) * silence_threshold).tobytes()
    assert not mic.is_silent(
        threshold_data
    ), "Threshold-level data should not be detected as silent"


@pytest.mark.asyncio
async def test_play_audio():
    """
    Test the play_audio function by adding dummy audio data to a queue,
    running the function, and ensuring the queue is empty after processing.
    """
    queue_chunks = queue.Queue()
    stop_event = asyncio.Event()

    # Add some dummy audio data to the queue
    dummy_data = b"\x00\x01" * 1024  # 2MB per sample, 1024 samples

    for _ in range(0, 10):
        queue_chunks.put(dummy_data)

    # Run play_audio in a separate task
    play_task = asyncio.create_task(mic.play_audio(queue_chunks, stop_event))

    # Allow some time for play_audio to process the queue
    await asyncio.sleep(0.1)

    # Stop the play_audio function
    stop_event.set()
    await play_task

    # Check that the queue is empty after processing
    assert not queue_chunks.empty()

````

## File: __init__.py

````py

````

## File: test_benchmark.py

````py
"""benchamrk"""

import json
import time
import pandas as pd

import requests
import jiwer as jw
import websocket as ws
import tests.utils as ut


def test_health(url="http://localhost:8181/health"):
    """basic test"""
    result = requests.get(url=url, timeout=1)
    assert result.status_code == 200


def get_res(websocket):
    """try read with timout"""
    try:
        result = json.loads(websocket.recv())
        print_result(result)
        return result
    except ws.WebSocketTimeoutException:
        return {}


def print_result(result: dict):
    """print result and execution time"""
    print(result["is_partial"], round(result["time"], 2), result["data"]["text"])


def test_send_chunks(url="ws://localhost:8181/ws", chunk_size=4096):
    """send chunks"""
    websocket = ws.create_connection(url)
    websocket.settimeout(0.1)

    resource = ut.load_resource("3081-166546-0000")
    chunks = [
        resource["audio"][i : i + chunk_size]
        for i in range(0, len(resource["audio"]), chunk_size)
    ]

    df_result = pd.DataFrame(columns=["is_partial", "latency", "result"])
    for chunk in chunks:
        websocket.send_bytes(chunk)
        res = get_res(websocket)
        if res:
            df_result.loc[len(df_result)] = [
                res["is_partial"],
                round(res["time"], 2),
                res["data"]["text"],
            ]

    attempts = 0
    while attempts < 3:
        res = get_res(websocket)
        if res:
            attempts = 0
            df_result.loc[len(df_result)] = [
                res["is_partial"],
                round(res["time"], 2),
                res["data"]["text"],
            ]
        else:
            attempts += 1
            time.sleep(1)

    pd.set_option("max_colwidth", 800)
    # print(df_result.to_string(justify='left', index=False))
    print("Latency Stats:\n", df_result["latency"].describe())

    actual = df_result.loc[len(df_result) - 1]["result"].lower().strip()
    expected = resource["expected"]["final_ground_truth"].lower().strip()

    error = round(jw.wer(actual, expected), 2)
    assert error < 0.1
    websocket.close()


if __name__ == "__main__":
    print("Starting Whisper-Flow Benchmark")
    test_send_chunks()
    print("Whisper-Flow Benchmark Completed")

````

## File: mic_transcribe.py

````py
""" 
a test app that streams 
audio  from the mic to whisper flow
requires pip install PyAudio
"""

import json
import asyncio
import pyaudio
import websockets


async def start_transcription(url="ws://0.0.0.0:8181/ws"):
    """stream mic audio to server"""
    async with websockets.connect(url) as websocket:
        result = []
        await asyncio.gather(
            capture_audio(websocket, result), receive_transcription(websocket, result)
        )
        print(f"* done recording, collecting data")
        print("Colllected text is \n", " ".join(result))


async def capture_audio(websocket: websockets.WebSocketClientProtocol, result: list):
    """capture the mic stream"""
    chunk, rate, record_sec = 1024, 16000, 30
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        input=True,
        frames_per_buffer=chunk,
    )
    print("* recording")

    for _ in range(0, int(rate / chunk * record_sec)):
        data = stream.read(chunk)
        await websocket.send(data)
        await asyncio.sleep(0.01)

    stream.close()
    p.terminate()


async def receive_transcription(websocket, result: list):
    """print transcription"""
    while True:
        try:
            await asyncio.sleep(0.01)
            tmp = json.loads(await websocket.recv())
            if not tmp["is_partial"]:
                result.append(tmp["data"]["text"])
            print(tmp["is_partial"], round(tmp["time"], 2), tmp["data"]["text"])
        except Exception:
            print("No transcription available")


asyncio.run(start_transcription())

````

## File: test_chat_room.py

````py
""" test chat room """

import queue
import asyncio
import pytest

from whisperflow.chat_room import ChatRoom


async def listener_mock(queue_in: queue.Queue, stop_event: asyncio.Event):
    """collect items from queue"""
    while not stop_event.is_set():
        await asyncio.sleep(0.1)
        queue_in.put("hello")


async def processor_mock(queue_in, queue_out, stop_event):
    """collect items from queue"""
    while not stop_event.is_set():
        await asyncio.sleep(0.1)
        if not queue_in.empty():
            item = queue_in.get()
            queue_out.put(item)


async def speaker_mock(queue_in: queue.Queue, stop_event: asyncio.Event):
    """mock playing sound"""
    while not stop_event.is_set():
        await asyncio.sleep(0.1)
        if not queue_in.empty():
            item = queue_in.get()
            assert item is not None


@pytest.mark.asyncio
async def test_chat_room():
    """mock playing sound"""
    room = ChatRoom(listener_mock, speaker_mock, processor_mock)

    async def stop_chat():
        await asyncio.sleep(1)
        room.stop_chat()

    await asyncio.gather(room.start_chat(), stop_chat())
    assert room.stop_chat_event.is_set()

````

## File: test_streaming.py

````py
""" test scenario module """

import asyncio
from queue import Queue

import pytest
import tests.utils as ut
import whisperflow.streaming as st
import whisperflow.fast_server as fs
import whisperflow.transcriber as ts


@pytest.mark.asyncio
async def test_simple():
    """test asyncio"""

    queue, should_stop = Queue(), [False]
    queue.put(1)

    async def dummy_transcriber(items: list) -> dict:
        await asyncio.sleep(0.1)
        if queue.qsize() == 0:
            should_stop[0] = True
        return {"text": str(len(items))}

    async def dummy_segment_closed(text: str) -> None:
        await asyncio.sleep(0.01)
        print(text)

    await st.transcribe(should_stop, queue, dummy_transcriber, dummy_segment_closed)
    assert queue.qsize() == 0


@pytest.mark.asyncio
async def test_transcribe_streaming(chunk_size=4096):
    """test streaming"""

    model = ts.get_model()
    queue, should_stop = Queue(), [False]
    res = ut.load_resource("3081-166546-0000")
    chunks = [
        res["audio"][i : i + chunk_size]
        for i in range(0, len(res["audio"]), chunk_size)
    ]

    async def dummy_transcriber(items: list) -> str:
        await asyncio.sleep(0.01)
        result = ts.transcribe_pcm_chunks(model, items)
        return result

    result = []

    async def dummy_segment_closed(text: str) -> None:
        await asyncio.sleep(0.01)
        result.append(text)

    task = asyncio.create_task(
        st.transcribe(should_stop, queue, dummy_transcriber, dummy_segment_closed)
    )

    for chunk in chunks:
        queue.put(chunk)
        await asyncio.sleep(0.01)

    await asyncio.sleep(1)
    should_stop[0] = True
    await task

    assert len(result) > 0


def test_streaming():
    """test hugging face image generation"""
    queue = Queue()
    queue.put(1)
    queue.put(2)
    res = st.get_all(queue)
    assert res == [1, 2]

    res = st.get_all(None)
    assert not res


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_ws(chunk_size=4096):
    """test health api"""
    client = ut.TestClient(fs.app)
    with client.websocket_connect("/ws") as websocket:
        res = ut.load_resource("3081-166546-0000")
        chunks = [
            res["audio"][i : i + chunk_size]
            for i in range(0, len(res["audio"]), chunk_size)
        ]

        for chunk in chunks:
            websocket.send_bytes(chunk)

        await asyncio.sleep(3)
        websocket.close()

    assert client

````

## File: test_transcriber.py

````py
""" test transcriber """

import pytest
from jiwer import wer
import tests.utils as ut

import whisperflow.fast_server as fr
import whisperflow.transcriber as tr


def test_load_model():
    """test load model from disl"""
    model = tr.get_model()
    assert model is not None

    resource = ut.load_resource("3081-166546-0000")

    result = tr.transcribe_pcm_chunks(model, [resource["audio"]])
    expected = resource["expected"]["final_ground_truth"]

    error = wer(result["text"].lower(), expected.lower())
    assert error < 0.1


def test_transcribe_chunk():
    """test transcribe pcm chunk"""
    resource = ut.load_resource("3081-166546-0000")
    client = ut.TestClient(fr.app)
    response = client.get("/health")
    assert response.status_code == 200

    path = ut.get_resource_path("3081-166546-0000", "wav")
    with open(path, "br") as file:
        response = client.post(
            url="/transcribe_pcm_chunk",
            data={"model_name": "tiny.en.pt"},
            files=[("files", file)],
        )

    assert response.status_code == 200

    expected = resource["expected"]["final_ground_truth"]
    error = wer(response.json()["text"].lower(), expected.lower())
    assert error < 0.1


@pytest.mark.asyncio
async def test_transcribe_chunk_async():
    """test transcribe async"""
    model = tr.get_model()
    assert model is not None
    resource = ut.load_resource("3081-166546-0000")
    result = await tr.transcribe_pcm_chunks_async(model, [resource["audio"]])
    expected = resource["expected"]["final_ground_truth"]
    error = wer(result["text"].lower(), expected.lower())
    assert error < 0.1

````

## File: utils.py

````py
""" test utils class """

import os
import json
from starlette.testclient import TestClient
import whisperflow.fast_server as fs


def get_resource_path(name: str, extension: str) -> str:
    "get resources path"
    current_path = os.path.dirname(__file__)
    path = os.path.join(current_path, f"./resources/{name}")
    return f"{path}.{extension}"


def load_resource(name: str) -> dict:
    "load resource"
    result = {}

    with open(get_resource_path(name, "wav"), "br") as file:
        result["audio"] = file.read()

    with open(get_resource_path(name, "json"), "r", encoding="utf-8") as file:
        result["expected"] = json.load(file)

    return result


def test_fast_api():
    """test health api"""
    with TestClient(fs.app) as client:
        response = client.get("/health")
        assert response.status_code == 200 and bool(response.text)

````

## File: __init__.py

````py
""" add package version """

__version__ = "1.0.0"

````

## File: microphone.py

````py
"""
capture audio from microphone
"""

import queue
import asyncio
import pyaudio
import numpy as np


async def capture_audio(
    queue_chunks: queue.Queue, stop_event: asyncio.Event
):  # pragma: no cover
    """capture the mic stream"""
    chunk, rate = 1024, 16000
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        input=True,
        frames_per_buffer=chunk,
    )

    while not stop_event.is_set():
        data = stream.read(chunk)
        queue_chunks.put_nowait(data)
        await asyncio.sleep(0.001)

    stream.close()
    audio.terminate()


async def play_audio(
    queue_chunks: queue.Queue, stop_event: asyncio.Event
):  # pragma: no cover
    """play audio from queue"""
    chunk, rate = 1024, 16000
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        output=True,
        frames_per_buffer=chunk,
    )

    while not stop_event.is_set():
        if not queue_chunks.empty():
            data = queue_chunks.get()
            stream.write(data)
        await asyncio.sleep(0.001)

    stream.close()
    audio.terminate()


def is_silent(data, silence_threshold=500):  # pragma: no cover
    """is chunk is silence"""
    return np.max(np.frombuffer(data, dtype=np.int16)) < silence_threshold

````

## File: chat_room.py

````py
""" 
Implements conversation loop: capture 
audio -> speech to text -> custom action -> text to speech -> play audio 
"""

import queue
import asyncio
import pytest
import whisperflow.audio.microphone as mic


class ChatRoom:
    """
    A class enabling real-time communication with microphone input and speaker output.
    It supports speech-to-text (STT) and text-to-speech (TTS)
    processing, with an optional handler for custom text analysis.
    """

    def __init__(self, listener, speaker, processor):
        self.audio_in = queue.Queue()
        self.audio_out = queue.Queue()
        self.listener = listener
        self.speaker = speaker
        self.processor = processor
        self.stop_chat_event = asyncio.Event()

    async def start_chat(self):
        """start chat by listening to mic"""
        self.stop_chat_event.clear()

        # start listener and processor
        await asyncio.gather(
            self.listener(self.audio_in, self.stop_chat_event),
            self.processor(self.audio_in, self.audio_out, self.stop_chat_event),
            self.speaker(self.audio_out, self.stop_chat_event),
        )

    def stop_chat(self):
        """stop chat and release resources"""
        self.stop_chat_event.set()
        assert self.stop_chat_event.is_set()


@pytest.mark.skip(reason="requires audio hardware")
def main():  # pragma: no cover
    """main function that runs the chat room"""

    # Create a dummy processor
    async def dummy_proc(
        audio_in: queue.Queue, audio_out: queue.Queue, stop: asyncio.Event
    ):
        """dummy processor"""
        while not stop.is_set():
            if not audio_in.empty():
                data = audio_in.get()
                audio_out.put(data)
            await asyncio.sleep(0.001)

    chat_room = ChatRoom(mic.capture_audio, mic.play_audio, dummy_proc)

    try:
        # Run the async main function
        chat_room.start_chat()
    except KeyboardInterrupt:
        chat_room.stop_chat()
        print("Chat stopped")


if __name__ == "__main__":
    main()

````

## File: fast_server.py

````py
""" fast api declaration """

import logging
from typing import List
from fastapi import FastAPI, WebSocket, Form, File, UploadFile

from whisperflow import __version__
import whisperflow.streaming as st
import whisperflow.transcriber as ts


app = FastAPI()
sessions = {}


@app.get("/health", response_model=str)
def health():
    """health function on API"""
    return f"Whisper Flow V{__version__}"


@app.post("/transcribe_pcm_chunk", response_model=dict)
def transcribe_pcm_chunk(
    model_name: str = Form(...), files: List[UploadFile] = File(...)
):
    """transcribe chunk"""
    model = ts.get_model(model_name)
    content = files[0].file.read()
    return ts.transcribe_pcm_chunks(model, [content])


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """webscoket implementation"""
    model = ts.get_model()

    async def transcribe_async(chunks: list):
        return await ts.transcribe_pcm_chunks_async(model, chunks)

    async def send_back_async(data: dict):
        await websocket.send_json(data)

    try:
        await websocket.accept()
        session = st.TranscribeSession(transcribe_async, send_back_async)
        sessions[session.id] = session

        while True:
            data = await websocket.receive_bytes()
            session.add_chunk(data)
    except Exception as exception:  # pylint: disable=broad-except
        logging.error(exception)
        await session.stop()
        await websocket.close()

````

## File: streaming.py

````py
""" test scenario module """

import time
import uuid
import asyncio
from queue import Queue
from typing import Callable


def get_all(queue: Queue) -> list:
    """get_all from queue"""
    res = []
    while queue and not queue.empty():
        res.append(queue.get())
    return res


async def transcribe(
    should_stop: list,
    queue: Queue,
    transcriber: Callable[[list], str],
    segment_closed: Callable[[dict], None],
):
    """the transcription loop"""
    window, prev_result, cycles = [], {}, 0

    while not should_stop[0]:
        start = time.time()
        await asyncio.sleep(0.01)
        window.extend(get_all(queue))

        if not window:
            continue

        result = {
            "is_partial": True,
            "data": await transcriber(window),
            "time": (time.time() - start) * 1000,
        }

        if should_close_segment(result, prev_result, cycles):
            window, prev_result, cycles = [], {}, 0
            result["is_partial"] = False
        elif result["data"]["text"] == prev_result.get("data", {}).get("text", ""):
            cycles += 1
        else:
            cycles = 0
            prev_result = result

        if result["data"]["text"]:
            await segment_closed(result)


def should_close_segment(result: dict, prev_result: dict, cycles, max_cycles=1):
    """return if segment should be closed"""
    return cycles >= max_cycles and result["data"]["text"] == prev_result.get(
        "data", {}
    ).get("text", "")


class TranscribeSession:  # pylint: disable=too-few-public-methods
    """transcription state"""

    def __init__(self, transcribe_async, send_back_async) -> None:
        """ctor"""
        self.id = uuid.uuid4()  # pylint: disable=invalid-name
        self.queue = Queue()
        self.should_stop = [False]
        self.task = asyncio.create_task(
            transcribe(self.should_stop, self.queue, transcribe_async, send_back_async)
        )

    def add_chunk(self, chunk: bytes):
        """add new chunk"""
        self.queue.put_nowait(chunk)

    async def stop(self):
        """stop session"""
        self.should_stop[0] = True
        await self.task

````

## File: transcriber.py

````py
""" transcriber """

import os
import asyncio

import torch
import numpy as np

import whisper
from whisper import Whisper


models = {}


def get_model(file_name="tiny.en.pt") -> Whisper:
    """load models from disk"""
    if file_name not in models:
        path = os.path.join(os.path.dirname(__file__), f"./models/{file_name}")
        models[file_name] = whisper.load_model(path).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return models[file_name]


def transcribe_pcm_chunks(
    model: Whisper, chunks: list, lang="en", temperature=0.1, log_prob=-0.5
) -> dict:
    """transcribes pcm chunks list"""
    arr = (
        np.frombuffer(b"".join(chunks), np.int16).flatten().astype(np.float32) / 32768.0
    )
    return model.transcribe(
        arr,
        fp16=False,
        language=lang,
        logprob_threshold=log_prob,
        temperature=temperature,
    )


async def transcribe_pcm_chunks_async(
    model: Whisper, chunks: list, lang="en", temperature=0.1, log_prob=-0.5
) -> dict:
    """transcribes pcm chunks async"""
    return await asyncio.get_running_loop().run_in_executor(
        None, transcribe_pcm_chunks, model, chunks, lang, temperature, log_prob
    )

````
