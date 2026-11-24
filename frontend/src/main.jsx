import { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const API = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [service, setService] = useState({ checked: false, online: false, reachable: false, model: '' });

  useEffect(() => {
    let active = true;
    Promise.all([
      fetch(`${API}/health`).then((response) => response.ok ? response.json() : null).catch(() => null),
      fetch(`${API}/model-info`).then((response) => response.ok ? response.json() : null).catch(() => null),
    ]).then(([health, info]) => {
      if (!active) return;
      const ready = typeof health?.model_loaded === 'boolean' ? health.model_loaded : health?.status === 'ok' || health?.status === 'healthy' || health?.ok === true || health?.ready === true;
      setService({ checked: true, online: ready, reachable: Boolean(health || info), model: info?.model_name || health?.model_backend || info?.model_backend || '' });
    }).catch(() => {
      if (active) setService({ checked: true, online: false, reachable: false, model: '' });
    });
    return () => { active = false; };
  }, []);

  async function analyze(event) {
    event.preventDefault();
    if (!text.trim() || loading) return;
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const response = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) throw new Error(response.status === 422 ? 'Văn bản không hợp lệ. Vui lòng kiểm tra nội dung và thử lại.' : `Máy chủ không thể phân tích lúc này (mã ${response.status}).`);
      const data = await response.json();
      if (typeof data.label_id !== 'number' || !data.probabilities || typeof data.confidence !== 'number') {
        throw new Error('Phản hồi từ máy chủ không đúng định dạng.');
      }
      setResult(data);
    } catch (err) {
      setError(err instanceof TypeError ? 'Không kết nối được máy chủ cục bộ. Hãy kiểm tra backend đang chạy tại địa chỉ đã cấu hình.' : err.message || 'Đã xảy ra lỗi khi phân tích.');
    } finally {
      setLoading(false);
    }
  }

  const clean = Number(result?.probabilities?.CLEAN ?? 0);
  const toxic = Number(result?.probabilities?.TOXIC ?? 0);
  const percentage = (value) => `${Math.round(value * 100)}%`;

  return (
    <main className="page-shell">
      <header className="topbar">
        <a className="wordmark" href="#main" aria-label="N11, về đầu trang"><span className="mark">N</span><span>N11 <small>NGÔN NGỮ AN TOÀN</small></span></a>
        <div className={`service-pill ${service.checked ? (service.online ? 'is-online' : 'is-offline') : ''}`} role="status">
          <span className="status-dot" />
          {service.checked ? (service.online ? 'Mô hình sẵn sàng' : service.reachable ? 'Mô hình chưa sẵn sàng' : 'Backend chưa kết nối') : 'Đang kiểm tra backend'}
          {service.model && <span className="model-name"> · {service.model}</span>}
        </div>
      </header>

      <section className="intro" id="main">
        <div className="intro-copy">
          <p className="kicker"><span /> CÔNG CỤ PHÂN TÍCH CỤC BỘ</p>
          <h1>Lời nói có thể<br />để lại <em>dấu vết.</em></h1>
          <p className="summary">Kiểm tra bình luận tiếng Việt để nhận diện nội dung độc hại. Văn bản được xử lý bởi mô hình chạy trên máy của bạn.</p>
        </div>
        <div className="motif" aria-hidden="true"><span className="motif-core">N11</span><span className="orbit orbit-one" /><span className="orbit orbit-two" /><i className="orbit-dot" /></div>
      </section>

      <section className="workspace" aria-label="Phân tích bình luận">
        <form className="composer" onSubmit={analyze}>
          <div className="section-heading"><label htmlFor="comment">Nội dung bình luận</label><span>Tiếng Việt · tối đa 1.000 ký tự</span></div>
          <textarea id="comment" value={text} onChange={(event) => setText(event.target.value.slice(0, 1000))} maxLength={1000} rows={7} placeholder="Dán hoặc nhập bình luận bạn muốn kiểm tra…" required aria-describedby="counter" />
          <div className="composer-footer"><span id="counter" className="counter">{text.length} / 1.000</span><button type="submit" disabled={!text.trim() || loading}>{loading ? <><span className="spinner" /> Đang phân tích</> : <>Phân tích bình luận <span aria-hidden="true">↗</span></>}</button></div>
        </form>

        <section className={`result-panel ${result ? 'has-result' : ''}`} aria-live="polite" aria-atomic="true">
          {error ? <div className="message error-message"><span className="message-icon" aria-hidden="true">!</span><div><strong>Chưa thể phân tích</strong><p>{error}</p></div></div> : result ? <>
            <div className="result-top"><div><span className="result-caption">KẾT QUẢ PHÂN TÍCH</span><h2 className={result.label_id === 1 ? 'toxic' : 'clean'}>{result.label_id === 1 ? 'Độc hại' : 'Lành tính'}</h2></div><div className="confidence"><strong>{percentage(result.confidence)}</strong><span>độ tin cậy</span></div></div>
            <div className="probabilities">
              <div className="probability-row"><span><i className="clean-swatch" />Lành tính</span><strong>{percentage(clean)}</strong></div>
              <div className="track"><span className="clean-bar" style={{ width: percentage(clean) }} /></div>
              <div className="probability-row"><span><i className="toxic-swatch" />Độc hại</span><strong>{percentage(toxic)}</strong></div>
              <div className="track"><span className="toxic-bar" style={{ width: percentage(toxic) }} /></div>
            </div>
            <p className="result-meta">{result.is_calibrated ? 'Xác suất đã hiệu chuẩn' : 'Xác suất chưa hiệu chuẩn'}{typeof result.latency_ms === 'number' ? ` · ${Math.round(result.latency_ms)} ms` : ''}</p>
          </> : <div className="empty-result"><span className="empty-mark" aria-hidden="true">···</span><div><strong>Kết quả sẽ xuất hiện ở đây</strong><p>Nhập một bình luận để xem nhãn và xác suất dự đoán.</p></div></div>}
        </section>
      </section>
      <footer><span>Phân tích chạy cục bộ trên thiết bị của bạn.</span><span>N11 <span className="footer-separator">/</span> Nhận diện văn bản tiếng Việt</span></footer>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
