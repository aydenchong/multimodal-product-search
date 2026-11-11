export type RetrievalItem = {
  id: string
  name: string
  category: string
  articleType: string
  similarity: number | null
  imageUrl: string | null
}

export type AnalysisResult = {
  predictedCategory: string
  confidence: number | null
  processingTimeMs: number | null
  retrievals: RetrievalItem[]
}

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ??
  (import.meta.env.DEV ? 'http://localhost:8000' : '')
).replace(/\/$/, '')
type UnknownRecord = Record<string, unknown>

function asRecord(value: unknown): UnknownRecord {
  return value && typeof value === 'object' ? (value as UnknownRecord) : {}
}

function firstString(...values: unknown[]): string {
  return values.find((value) => typeof value === 'string' && value.trim().length > 0)?.toString().trim() || ''
}

function firstNumber(...values: unknown[]): number | null {
  const value = values.find((candidate) => typeof candidate === 'number' && Number.isFinite(candidate))
  return typeof value === 'number' ? value : null
}

function normalizeConfidence(value: number | null): number | null {
  if (value === null) return null
  return value > 1 ? Math.min(value / 100, 1) : Math.max(value, 0)
}

function normalizeImageUrl(value: unknown): string | null {
  if (typeof value !== 'string' || !value.trim()) return null
  const trimmed = value.trim()
  if (trimmed.startsWith('data:image/')) return trimmed
  if (trimmed.startsWith('http://') || trimmed.startsWith('https://')) return trimmed
  return trimmed.startsWith('/') ? `${API_BASE_URL}${trimmed}` : `${API_BASE_URL}/${trimmed}`
}

export function normalizeAnalysisResponse(payload: unknown): AnalysisResult {
  const root = asRecord(payload)
  const classification = asRecord(root.classification || root.prediction || root.classifier_result)
  const predictedCategory = firstString(
    classification.predicted_category,
    classification.category,
    classification.label,
    root.predicted_category,
    root.predictedCategory,
    root.category,
  ) || 'Unclassified'

  const confidence = normalizeConfidence(firstNumber(
    classification.confidence,
    classification.softmax_confidence,
    classification.softmaxConfidence,
    root.confidence,
    root.softmax_confidence,
  ))

  const processingTimeMs = firstNumber(
    root.processing_time_ms,
    root.processingTimeMs,
    root.processing_time,
    root.elapsed_ms,
  )

  const rawRetrievals = root.retrievals || root.retrieval_results || root.similar_products || root.results || []
  const retrievals = Array.isArray(rawRetrievals)
    ? rawRetrievals.slice(0, 5).map((item, index) => {
        const record = asRecord(item)
        return {
          id: firstString(record.product_id, record.productId, record.id, record.article_id) || `result-${index + 1}`,
          name: firstString(record.product_name, record.productName, record.name, record.title) || 'Unnamed product',
          category: firstString(record.category, record.product_category) || 'Category unavailable',
          articleType: firstString(record.article_type, record.articleType, record.type) || 'Article type unavailable',
          similarity: firstNumber(record.cosine_similarity, record.cosineSimilarity, record.similarity, record.score),
          imageUrl: normalizeImageUrl(record.image_url || record.imageUrl || record.image_endpoint || record.imageEndpoint),
        }
      })
    : []

  return { predictedCategory, confidence, processingTimeMs, retrievals }
}

export async function analyzeImage(file: File, signal: AbortSignal): Promise<AnalysisResult> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('top_k', '5')

  const response = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
    method: 'POST',
    body: formData,
    signal,
  })

  if (!response.ok) {
    let detail = `Analysis failed with status ${response.status}.`
    try {
      const body = asRecord(await response.json())
      detail = firstString(body.detail, body.message, body.error) || detail
    } catch {
      // The status message is enough when the response is not JSON.
    }
    throw new Error(detail)
  }

  return normalizeAnalysisResponse(await response.json())
}

export { API_BASE_URL }
