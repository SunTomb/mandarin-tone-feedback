import { useRef, useState } from "react";

function encodeWav(audioBuffer: AudioBuffer): Blob {
  const channel = audioBuffer.getChannelData(0);
  const sampleRate = audioBuffer.sampleRate;
  const buffer = new ArrayBuffer(44 + channel.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) view.setUint8(offset + index, value.charCodeAt(index));
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + channel.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, channel.length * 2, true);

  let offset = 44;
  for (const sample of channel) {
    const clamped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

async function recordingBlobToWavFile(blob: Blob): Promise<File> {
  const audioContext = new AudioContext();
  try {
    const arrayBuffer = await blob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const wavBlob = encodeWav(audioBuffer);
    return new File([wavBlob], "recording.wav", { type: "audio/wav" });
  } finally {
    await audioContext.close();
  }
}

export function Recorder({ onDemo, onFile }: { onDemo: () => void; onFile: (file: File) => void }) {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedFile, setRecordedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const canRecord = typeof navigator !== "undefined" && Boolean(navigator.mediaDevices) && typeof MediaRecorder !== "undefined";

  async function startRecording() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      chunksRef.current = [];
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        const blob = new Blob(chunksRef.current, { type: mediaRecorder.mimeType || "audio/webm" });
        try {
          setRecordedFile(await recordingBlobToWavFile(blob));
        } catch {
          setError("录音格式转换失败，请改用 WAV 文件上传。");
        }
        setIsRecording(false);
      };
      mediaRecorder.start();
      setIsRecording(true);
    } catch {
      setError("无法启动录音，请检查浏览器麦克风权限，或改用音频文件上传。");
      setIsRecording(false);
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
  }

  return (
    <section className="recorder">
      <h2>录音 / 上传</h2>
      <p>当前网页以“目标声调 + F0 曲线诊断”为主，预测声调仅供参考；请录制 1–2 秒清晰单字或短词。</p>
      {canRecord ? (
        <div className="recorder-actions">
          <button type="button" onClick={isRecording ? stopRecording : startRecording}>
            {isRecording ? "停止录音" : "开始录音"}
          </button>
          {recordedFile && (
            <>
              <audio controls src={URL.createObjectURL(recordedFile)} />
              <button type="button" onClick={() => onFile(recordedFile)}>提交录音</button>
            </>
          )}
        </div>
      ) : (
        <p>当前浏览器不支持直接录音，请使用音频文件上传。</p>
      )}
      {error && <p className="recording-error">{error}</p>}
      <label className="file-upload">
        上传音频文件
        <input
          type="file"
          accept="audio/*"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onFile(file);
          }}
        />
      </label>
      <button onClick={onDemo}>生成演示反馈</button>
    </section>
  );
}
