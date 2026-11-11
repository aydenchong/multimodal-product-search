import { ChangeEvent, DragEvent, useCallback, useEffect, useRef, useState } from 'react'
import {
  ArrowUpRight,
  Check,
  ChevronRight,
  CircleAlert,
  Clock3,
  FileImage,
  ImagePlus,
  LoaderCircle,
  RotateCcw,
  Search,
  Sparkles,
  UploadCloud,
  X,
} from 'lucide-react'
import { analyzeImage, AnalysisResult, RetrievalItem } from './api'

const MAX_FILE_SIZE = 5 * 1024 * 1024
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp']

type Status = 'idle' | 'loading' | 'success' | 'error'

function formatFileSize(bytes: number) {
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`
}

function formatTime(milliseconds: number | null) {
  if (milliseconds === null) return '—'
  return milliseconds < 1000 ? `${Math.round(milliseconds)} ms` : `${(milliseconds / 1000).toFixed(2)} s`
}

function formatPercent(value: number | null) {
  return value === null ? '—' : `${Math.round(value * 100)}%`
}

function validateFile(file: File) {
  if (!ACCEPTED_TYPES.includes(file.type)) return 'Use a JPEG, PNG, or WebP image.'
  if (file.size > MAX_FILE_SIZE) return 'That image is larger than 5 MB. Choose a smaller file.'
  return null
}

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const controllerRef = useRef<AbortController | null>(null)
  const requestIdRef = useRef(0)

  useEffect(() => () => {
    controllerRef.current?.abort()
    if (previewUrl) URL.revokeObjectURL(previewUrl)
  }, [previewUrl])

  const handleFile = useCallback((nextFile: File | undefined) => {
    if (!nextFile) return
    const validationError = validateFile(nextFile)
    controllerRef.current?.abort()
    setStatus(validationError ? 'error' : 'idle')
    setError(validationError)
    setResult(null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    if (validationError) {
      setFile(null)
      setPreviewUrl(null)
      return
    }
    setFile(nextFile)
    setPreviewUrl(URL.createObjectURL(nextFile))
  }, [previewUrl])

  const onInputChange = (event: ChangeEvent<HTMLInputElement>) => handleFile(event.target.files?.[0])

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragging(false)
    handleFile(event.dataTransfer.files?.[0])
  }

  const reset = useCallback(() => {
    controllerRef.current?.abort()
    requestIdRef.current += 1
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setFile(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
    setStatus('idle')
    if (inputRef.current) inputRef.current.value = ''
  }, [previewUrl])

  const submit = async () => {
    if (!file || status === 'loading') return
    const requestId = requestIdRef.current + 1
    requestIdRef.current = requestId
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    setStatus('loading')
    setError(null)
    try {
      const analysis = await analyzeImage(file, controller.signal)
      if (requestId !== requestIdRef.current) return
      setResult(analysis)
      setStatus('success')
    } catch (caught) {
      if (controller.signal.aborted || requestId !== requestIdRef.current) return
      setStatus('error')
      setError(caught instanceof Error ? caught.message : 'The analysis could not be completed.')
    }
  }

  const hasAnalysis = status === 'success' && result

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="/" aria-label="ShopSense AI home">
          <span className="brand-mark">S</span>
          <span className="brand-name">ShopSense <span>AI</span></span>
        </a>
        <nav className="primary-nav" aria-label="Primary navigation">
          <a className="nav-link active" href="#analyze">Analyze</a>
          <a className="nav-link" href="#how-it-works">How it works</a>
          <span className="status-pill"><span className="status-dot" /> Inference ready</span>
        </nav>
        <button className="icon-button mobile-menu" type="button" aria-label="Open navigation"><Search size={18} /></button>
      </header>

      <main id="analyze" className="page-content">
        <section className="intro-block" aria-labelledby="page-title">
          <div className="intro-copy">
            <p className="eyebrow"><Sparkles size={15} /> Product intelligence workspace</p>
            <h1 id="page-title">See what your product image is trying to tell you.</h1>
            <p className="intro-lede">Upload one product image to classify its category and find the closest visual matches in your catalog.</p>
          </div>
          <div className="intro-aside" aria-label="Analysis pipeline">
            <span>Image signal</span><ChevronRight size={14} /><span>Category</span><ChevronRight size={14} /><span>Matches</span>
          </div>
        </section>

        <section className="workspace-grid" aria-label="Product image analysis">
          <div className="upload-panel panel">
            <div className="panel-heading">
              <div>
                <p className="panel-kicker">01 / Input</p>
                <h2>Bring an image into focus</h2>
              </div>
              <span className="step-badge">JPEG · PNG · WebP</span>
            </div>

            <div
              className={`dropzone ${isDragging ? 'dragging' : ''} ${previewUrl ? 'has-preview' : ''}`}
              onDragEnter={(event) => { event.preventDefault(); setIsDragging(true) }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={(event) => { if (event.currentTarget === event.target) setIsDragging(false) }}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') inputRef.current?.click() }}
              role="button"
              tabIndex={0}
              aria-label="Upload product image"
            >
              <input ref={inputRef} type="file" accept={ACCEPTED_TYPES.join(',')} onChange={onInputChange} hidden />
              {previewUrl ? (
                <div className="preview-wrap">
                  <img src={previewUrl} alt="Selected product preview" className="preview-image" />
                  <div className="preview-overlay"><span>Change image</span><ImagePlus size={16} /></div>
                </div>
              ) : (
                <div className="dropzone-empty">
                  <span className="upload-icon"><UploadCloud size={23} strokeWidth={1.7} /></span>
                  <strong>Drop your product image here</strong>
                  <span>or click to browse your files</span>
                  <small>Maximum file size 5 MB</small>
                </div>
              )}
            </div>

            {file && (
              <div className="file-meta">
                <div className="file-meta-icon"><FileImage size={17} /></div>
                <div className="file-meta-copy"><strong>{file.name}</strong><span>{formatFileSize(file.size)}</span></div>
                <Check size={17} className="file-ok" aria-label="File ready" />
                <button className="quiet-button" type="button" onClick={reset} aria-label="Remove selected image"><X size={16} /></button>
              </div>
            )}

            {error && status !== 'success' && (
              <div className="inline-error" role="alert"><CircleAlert size={16} /><span>{error}</span></div>
            )}

            <div className="upload-actions">
              <button className="secondary-button" type="button" onClick={reset} disabled={!file && !error}><RotateCcw size={16} /> Reset</button>
              <button className="primary-button" type="button" onClick={submit} disabled={!file || status === 'loading'}>
                {status === 'loading' ? <><LoaderCircle className="spin" size={17} /> Analyzing image</> : <>Analyze image <ArrowUpRight size={17} /></>}
              </button>
            </div>
          </div>

          <div className="results-panel panel">
            <div className="panel-heading results-heading">
              <div><p className="panel-kicker">02 / Intelligence</p><h2>What the models see</h2></div>
              {hasAnalysis && <span className="live-badge"><span className="status-dot" /> Live result</span>}
            </div>

            {status === 'loading' && <LoadingState />}
            {status === 'idle' && <EmptyState />}
            {status === 'error' && !result && <ErrorState message={error || 'The analysis could not be completed.'} onRetry={submit} />}
            {hasAnalysis && <AnalysisView result={result} />}
          </div>
        </section>

        <footer className="page-footer" id="how-it-works">
          <span><span className="footer-mark" /> ShopSense AI</span>
          <span>ResNet18 classification · CLIP + FAISS retrieval</span>
          <span>Top 5 matches</span>
        </footer>
      </main>
    </div>
  )
}

function EmptyState() {
  return <div className="state empty-state"><div className="signal-graphic" aria-hidden="true"><span /><span /><span /><span /><span /><i /></div><h3>Your signal map is waiting</h3><p>Upload a product image to reveal the predicted category and the five closest catalog matches.</p><span className="state-note">No analysis data yet</span></div>
}

function LoadingState() {
  return <div className="state loading-state"><div className="loading-orbit"><LoaderCircle size={30} className="spin" /></div><h3>Reading the visual signal</h3><p>Running classification, embedding search, and catalog matching.</p><div className="loading-bars"><span /><span /><span /></div></div>
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return <div className="state error-state"><div className="error-icon"><CircleAlert size={24} /></div><h3>We could not complete that analysis</h3><p>{message}</p><button className="secondary-button" type="button" onClick={onRetry}><RotateCcw size={16} /> Try again</button></div>
}

function AnalysisView({ result }: { result: AnalysisResult }) {
  return <div className="analysis-content">
    <div className="classification-summary">
      <div><span className="summary-label">Predicted category</span><h3>{result.predictedCategory}</h3></div>
      <div className="confidence-block"><span className="summary-label">Softmax confidence</span><strong>{formatPercent(result.confidence)}</strong>{result.confidence !== null && <div className="confidence-track"><span style={{ width: `${result.confidence * 100}%` }} /></div>}</div>
      <div className="time-block"><Clock3 size={16} /><span><b>{formatTime(result.processingTimeMs)}</b><small>processing time</small></span></div>
    </div>
    <div className="retrieval-heading"><div><span className="summary-label">Visual neighbors</span><h3>Closest catalog matches</h3></div><span className="retrieval-count">{result.retrievals.length} of 5</span></div>
    {result.retrievals.length ? <div className="retrieval-list">{result.retrievals.map((item, index) => <RetrievalCard key={`${item.id}-${index}`} item={item} rank={index + 1} />)}</div> : <div className="no-retrievals"><Search size={18} /><span>The classifier returned a category, but no retrieval records were returned.</span></div>}
  </div>
}

function RetrievalCard({ item, rank }: { item: RetrievalItem; rank: number }) {
  return <article className="retrieval-card">
    <div className="rank">0{rank}</div>
    <div className="product-image">{item.imageUrl ? <img src={item.imageUrl} alt={item.name} /> : <div className="image-unavailable"><FileImage size={19} /><span>No image</span></div>}</div>
    <div className="product-copy"><h4>{item.name}</h4><div className="product-details"><span>{item.category}</span><span>{item.articleType}</span></div></div>
    <div className="similarity"><strong>{item.similarity === null ? '—' : item.similarity.toFixed(3)}</strong><span>cosine</span></div>
  </article>
}

export default App
