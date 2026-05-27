"use client";

import { useRef, useState } from "react";
import { Mic, Square } from "lucide-react";

// Modal live transcription WebSocket endpoint
const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL ||
  "wss://boxerab--aavaaz-live-livetranscriber-web.modal.run/ws";

const SAMPLE_RATE = 16000;

export default function LiveDemoPage() {
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [status, setStatus] = useState("Ready");
  const [language, setLanguage] = useState("");
  const [model, setModel] = useState("large-v3");
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  async function startRecording() {
    setTranscript("");
    setStatus("Requesting microphone...");

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: true },
      });
    } catch {
      setStatus("Microphone access denied");
      return;
    }
    streamRef.current = stream;

    setStatus("Connecting...");
    const ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("Connected — speak now");
      setRecording(true);

      // Send WhisperLive client options
      const options = {
        uid: crypto.randomUUID(),
        language: language || null,
        task: "transcribe",
        model: model,
        use_vad: true,
        word_timestamps: false,
      };
      ws.send(JSON.stringify(options));

      // Start audio capture at 16kHz, send raw Float32 PCM
      const audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (ws.readyState === WebSocket.OPEN) {
          const float32 = e.inputBuffer.getChannelData(0);
          ws.send(float32.buffer);
        }
      };
      source.connect(processor);
      processor.connect(audioContext.destination);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data as string);
        if (data.segments) {
          const text = data.segments.map((s: { text: string }) => s.text).join(" ");
          setTranscript(text);
        } else if (data.message === "DISCONNECT") {
          stopRecording();
          setStatus("Server disconnected");
        } else if (data.status === "WAIT") {
          setStatus(
            `Server busy — wait ~${Math.ceil(data.message)} min`
          );
        }
      } catch {
        // non-JSON message
      }
    };

    ws.onerror = () => {
      setStatus("Connection error");
      stopRecording();
    };

    ws.onclose = () => {
      setStatus("Disconnected");
      setRecording(false);
    };
  }

  function stopRecording() {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      // Send END_OF_AUDIO signal
      wsRef.current.send(new TextEncoder().encode("END_OF_AUDIO"));
      setTimeout(() => wsRef.current?.close(), 500);
    }
    setRecording(false);
    setStatus("Stopped");
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold">Live Transcription</h1>
        <p className="text-muted-foreground mt-1">
          Real-time speech-to-text from your microphone
        </p>
      </div>

      <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 px-4 py-3 text-sm text-amber-200">
        <strong>Note:</strong> The transcription server uses serverless GPUs that scale to zero when idle.
        The first connection may take <strong>30–60 seconds</strong> for a cold start while the model loads.
        Subsequent connections will be instant.
      </div>

      <div className="rounded-lg border bg-card p-8 text-center space-y-6">
        {/* Controls */}
        <div className="flex items-center justify-center gap-4 flex-wrap">
          <div>
            <label htmlFor="language" className="text-sm text-muted-foreground mr-2">
              Language:
            </label>
            <select
              id="language"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              disabled={recording}
              className="rounded-md border border-input bg-background px-2 py-1 text-sm"
            >
              <option value="">Auto-detect</option>
              <option value="en">English</option>
              <option value="es">Spanish</option>
              <option value="fr">French</option>
              <option value="de">German</option>
              <option value="it">Italian</option>
              <option value="pt">Portuguese</option>
              <option value="ja">Japanese</option>
              <option value="zh">Chinese</option>
            </select>
          </div>
          <div>
            <label htmlFor="model" className="text-sm text-muted-foreground mr-2">
              Model:
            </label>
            <select
              id="model"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={recording}
              className="rounded-md border border-input bg-background px-2 py-1 text-sm"
            >
              <option value="large-v3">large-v3</option>
              <option value="small">small</option>
              <option value="tiny">tiny</option>
            </select>
          </div>
        </div>

        {/* Status indicator */}
        <div
          className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
            recording
              ? "bg-green-500/10 text-green-500"
              : "bg-muted text-muted-foreground"
          }`}
        >
          <span
            className={`h-2 w-2 rounded-full ${
              recording ? "bg-green-500 animate-pulse" : "bg-muted-foreground"
            }`}
          />
          {status}
        </div>

        {/* Record button */}
        <div>
          {!recording ? (
            <button
              onClick={startRecording}
              className="inline-flex items-center gap-2 rounded-full bg-primary px-8 py-4 text-lg font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              <Mic className="h-5 w-5" />
              Start Recording
            </button>
          ) : (
            <button
              onClick={stopRecording}
              className="inline-flex items-center gap-2 rounded-full bg-destructive px-8 py-4 text-lg font-medium text-destructive-foreground hover:bg-destructive/90 transition-colors"
            >
              <Square className="h-5 w-5" />
              Stop
            </button>
          )}
        </div>

        {/* Transcript display */}
        <div className="min-h-[200px] max-h-[400px] overflow-y-auto rounded-lg border bg-background p-6 text-left">
          {transcript ? (
            <p className="text-foreground leading-relaxed">{transcript}</p>
          ) : (
            <p className="text-muted-foreground italic">
              Transcript will appear here as you speak...
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
