const MODELS_DEV_URL = "https://models.dev/models.json"
let enrichCache = null

export const ENRICH_FIELDS = [
  "name",
  "description",
  "family",
  "reasoning",
  "attachment",
  "tool_call",
  "structured_output",
  "temperature",
  "knowledge",
  "open_weights",
  "release_date",
  "last_updated",
  "modalities",
  "limit",
]

async function fetchEnrichData(fetchFn) {
  const useFetch = fetchFn || globalThis.fetch
  const response = await useFetch(MODELS_DEV_URL)
  if (!response.ok) {
    throw new Error(`models.dev fetch failed: HTTP ${response.status}`)
  }
  return await response.json()
}

function buildModelIdIndex(data) {
  const byExact = new Map()
  const bySuffix = new Map()

  for (const [key, value] of Object.entries(data)) {
    byExact.set(key, value)
    const id = value.id || key
    byExact.set(id, value)

    const parts = id.split("/")
    if (parts.length > 1) {
      const suffix = parts.slice(1).join("/")
      bySuffix.set(suffix, value)
    }
  }

  return { byExact, bySuffix }
}

function matchModelId(modelId, index) {
  if (!modelId || typeof modelId !== "string") return null

  const exact = index.byExact.get(modelId)
  if (exact) return exact

  const suffix = index.bySuffix.get(modelId)
  if (suffix) return suffix

  for (const [key, value] of index.byExact) {
    if (key.endsWith(`/${modelId}`)) return value
  }

  return null
}

function pickEnrichFields(source, fields) {
  const result = {}
  for (const field of fields) {
    if (field in source) {
      result[field] = JSON.parse(JSON.stringify(source[field]))
    }
  }
  return result
}

/**
 * Fetch models.dev data and build a lookup cache.
 * Returns a Map<modelId, partialEntry> or null on failure.
 * Call this once before batch-processing models.
 */
export async function fetchAndBuildEnrichCache(options = {}) {
  const { fetch: fetchFn, onError, fields = ENRICH_FIELDS } = options

  try {
    const data = await fetchEnrichData(fetchFn)
    const index = buildModelIdIndex(data)
    const cache = new Map()

    const seen = new Set()

    for (const modelId of index.byExact.keys()) {
      const matched = matchModelId(modelId, index)
      if (matched && !seen.has(modelId)) {
        cache.set(modelId, pickEnrichFields(matched, fields))
        seen.add(modelId)
      }
    }

    for (const modelId of index.bySuffix.keys()) {
      if (seen.has(modelId)) continue
      const matched = matchModelId(modelId, index)
      if (matched) {
        cache.set(modelId, pickEnrichFields(matched, fields))
        seen.add(modelId)
      }
    }

    enrichCache = cache
    return cache
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    if (typeof onError === "function") onError(message)
    return null
  }
}

/**
 * Synchronously look up an enriched entry from the pre-built cache.
 * Returns null if cache is not built or no match found.
 */
export function lookupEnrichment(modelId) {
  if (!enrichCache || !modelId) return null
  return enrichCache.get(modelId) || null
}

export async function enrichModelEntry(modelId, existingEntry, options = {}) {
  const {
    fields = ENRICH_FIELDS,
    fetch: fetchFn,
    onError,
  } = options

  if (!enrichCache) {
    try {
      const data = await fetchEnrichData(fetchFn)
      enrichCache = buildModelIdIndex(data)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      if (typeof onError === "function") onError(message)
      return null
    }
  }

  const matched = matchModelId(modelId, enrichCache)
  if (!matched) return null

  return pickEnrichFields(matched, fields)
}

export function resetCache() {
  enrichCache = null
}
