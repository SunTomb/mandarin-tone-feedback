export function Recorder({ onDemo, onFile }: { onDemo: () => void; onFile: (file: File) => void }) {
  return (
    <section className="recorder">
      <h2>录音 / 上传</h2>
      <p>上传一个短音频文件，或使用演示接口查看反馈闭环。</p>
      <input
        type="file"
        accept="audio/*"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onFile(file);
        }}
      />
      <button onClick={onDemo}>生成演示反馈</button>
    </section>
  );
}
