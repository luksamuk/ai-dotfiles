/**
 * OpenCode plugin: auto-sync models for existing OpenAI-compatible providers.
 *
 * 同时支持 OpenCode v1 和 v2：
 * - v1 使用 `server()` 函数，通过 config hook 修改配置
 * - v2 使用 `id` + `setup()` 函数，通过 provider.transform 修改模型
 *
 * 它在启动时为每个配置的 provider 获取 `${baseURL}/models`，
 * 并将发现的模型列表注入到 provider 的 models 中。
 * 现有的本地模型定义会被保留和合并，手动维护的元数据优先级更高。
 *
 * `provider` 接受单个 ID 字符串或 ID 字符串/配置对象数组
 * `{ id, baseURL?, apiKey?, apiKeyEnv? }`。
 *
 * 当启用 `enrich`（默认：false）时，插件会获取
 * https://models.dev/models.json 并从匹配的条目填充缺失字段。
 */

import { fetchAndBuildEnrichCache, lookupEnrichment, resetCache as resetEnrichCache } from "./enrichment.js"

const DEFAULT_INPUT_MODALITIES = ["text"]
const DEFAULT_OUTPUT_MODALITIES = ["text"]
const DEFAULT_TIMEOUT = 1000
const DEFAULT_STARTUP_TIMEOUT = 3000

const modelCache = new Map()
const inflightRequests = new Map()

// v2 专用：记录每个 provider 已同步的模型列表
const syncedModels = new Map()

function getCacheKey(baseURL, apiKey) {
  return `${baseURL}|${apiKey || ""}`
}

function normalizeBaseUrl(baseURL) {
  if (typeof baseURL !== "string" || !baseURL.trim()) return null
  return baseURL.replace(/\/+$/, "")
}

function normalizeModelId(value) {
  if (typeof value !== "string") return null
  const id = value.trim()
  return id ? id : null
}

function looksLikeReasoningModel(modelId) {
  return /(reason|thinking|o1|o3|o4|r1|r\d|deepseek|glm-5|minimax-m2\.5)/i.test(modelId)
}

function deriveLimit(model) {
  const context =
    Number(model?.context_window) ||
    Number(model?.contextWindow) ||
    Number(model?.input_token_limit) ||
    Number(model?.inputTokenLimit) ||
    Number(model?.max_input_tokens) ||
    Number(model?.maxInputTokens) ||
    null

  const output =
    Number(model?.max_output_tokens) ||
    Number(model?.maxOutputTokens) ||
    Number(model?.output_token_limit) ||
    Number(model?.outputTokenLimit) ||
    null

  if (!context && !output) return undefined
  return {
    ...(context ? { context } : {}),
    ...(output ? { output } : {}),
  }
}

function deriveModalities(model) {
  const rawInput = Array.isArray(model?.input_modalities)
    ? model.input_modalities
    : Array.isArray(model?.modalities?.input)
      ? model.modalities.input
      : null

  const rawOutput = Array.isArray(model?.output_modalities)
    ? model.output_modalities
    : Array.isArray(model?.modalities?.output)
      ? model.modalities.output
      : null

  const input = rawInput?.filter((item) => typeof item === "string" && item.trim()) || DEFAULT_INPUT_MODALITIES
  const output = rawOutput?.filter((item) => typeof item === "string" && item.trim()) || DEFAULT_OUTPUT_MODALITIES

  return { input, output }
}

function toDisplayName(modelId, remoteModel) {
  if (typeof remoteModel?.name === "string" && remoteModel.name.trim()) return remoteModel.name.trim()
  return modelId
}

function resolveEnvTemplate(value) {
  if (typeof value === "string") {
    const match = value.match(/^\{env:([^}]+)\}$/)
    if (match) return process.env[match[1]] || undefined
  }
  return value
}

// ---- v1 enrichment ----

function buildModelEntry(modelId, remoteModel, existingEntry, enrichCache) {
  const generated = {
    name: toDisplayName(modelId, remoteModel),
    modalities: deriveModalities(remoteModel),
    ...(looksLikeReasoningModel(modelId) ? { reasoning: true } : {}),
    ...(deriveLimit(remoteModel) ? { limit: deriveLimit(remoteModel) } : {}),
  }

  if (!existingEntry || typeof existingEntry !== "object") {
    if (enrichCache) {
      return applyEnrichmentV1(modelId, generated, enrichCache)
    }
    return generated
  }

  const merged = {
    ...generated,
    ...existingEntry,
    modalities: existingEntry.modalities || generated.modalities,
    limit: existingEntry.limit || generated.limit,
  }

  if (enrichCache) {
    return applyEnrichmentV1(modelId, merged, enrichCache)
  }

  return merged
}

function applyEnrichmentV1(modelId, entry, enrichCache) {
  if (!enrichCache) return entry

  const matched = lookupEnrichment(modelId)
  if (!matched) return entry

  const result = { ...entry }
  for (const [key, value] of Object.entries(matched)) {
    if (result[key] === undefined || result[key] === null) {
      result[key] = JSON.parse(JSON.stringify(value))
    } else if (key === "name" && result.name === modelId) {
      result.name = JSON.parse(JSON.stringify(value))
    } else if (key === "modalities" && isDefaultModalities(result.modalities)) {
      result.modalities = JSON.parse(JSON.stringify(value))
    } else if (key === "limit" && isDefaultLimit(result.limit)) {
      result.limit = JSON.parse(JSON.stringify(value))
    }
  }

  return result
}

// ---- v2 enrichment ----

function applyEnrichmentV2(modelId, patch, enrichCache) {
  if (!enrichCache) return patch
  const matched = lookupEnrichment(modelId)
  if (!matched) return patch

  const result = { ...patch }
  if ((result.name === undefined || result.name === modelId) && typeof matched.name === "string") {
    result.name = matched.name
  }
  if (typeof matched.family === "string" && !result.family) result.family = matched.family

  result.capabilities = { ...(result.capabilities || { tools: true, input: [], output: [] }) }
  if (Array.isArray(matched.modalities?.input) && matched.modalities.input.length) {
    result.capabilities.input = [...matched.modalities.input]
  }
  if (Array.isArray(matched.modalities?.output) && matched.modalities.output.length) {
    result.capabilities.output = [...matched.modalities.output]
  }
  if (typeof matched.tool_call === "boolean") result.capabilities.tools = matched.tool_call
  if (matched.limit && typeof matched.limit === "object") {
    result.limit = {
      ...(result.limit || {}),
      ...(result.limit?.context === undefined ? { context: matched.limit.context } : {}),
      ...(result.limit?.output === undefined ? { output: matched.limit.output } : {}),
    }
  }
  return result
}

function buildV2ModelPatch(modelId, remoteModel, existingInfo, enrichCache) {
  const modalities = deriveModalities(remoteModel)
  const patch = {
    name: toDisplayName(modelId, remoteModel),
    capabilities: {
      tools: true,
      input: modalities.input,
      output: modalities.output,
    },
  }

  // 如果远程 model ID 与 OpenCode model ID 不同，设置 modelID
  if (remoteModel?.id && remoteModel.id !== modelId) {
    patch.modelID = remoteModel.id
  }

  const limit = deriveLimit(remoteModel)
  if (limit) patch.limit = limit

  const result = applyEnrichmentV2(modelId, patch, enrichCache)

  if (existingInfo && typeof existingInfo === "object") {
    if (existingInfo.name) result.name = existingInfo.name
    if (existingInfo.modelID) result.modelID = existingInfo.modelID
    if (existingInfo.capabilities) result.capabilities = existingInfo.capabilities
    if (existingInfo.limit) result.limit = existingInfo.limit
    if (existingInfo.family) result.family = existingInfo.family
  }

  return result
}

// ---- helpers ----

function isDefaultModalities(mod) {
  if (!mod || typeof mod !== "object") return true
  const { input, output } = mod
  return (
    (!input || (input.length === 1 && input[0] === "text")) &&
    (!output || (output.length === 1 && output[0] === "text"))
  )
}

function isDefaultLimit(limit) {
  return !limit || typeof limit !== "object" || Object.keys(limit).length === 0
}

function getApiKey(options, perProviderOpts, globalOpts) {
  const envName = typeof perProviderOpts?.apiKeyEnv === "string"
    ? perProviderOpts.apiKeyEnv
    : typeof globalOpts?.apiKeyEnv === "string"
      ? globalOpts.apiKeyEnv
      : null
  if (envName && process.env[envName]) return process.env[envName]
  if (typeof perProviderOpts?.apiKey === "string" && perProviderOpts.apiKey.trim()) return perProviderOpts.apiKey.trim()
  if (typeof globalOpts?.apiKey === "string" && globalOpts.apiKey.trim()) return globalOpts.apiKey.trim()
  if (typeof options?.apiKey === "string" && options.apiKey.trim()) return options.apiKey.trim()
  return null
}

async function fetchRemoteModels(baseURL, apiKey, timeoutMs) {
  const cacheKey = getCacheKey(baseURL, apiKey)

  const inflight = inflightRequests.get(cacheKey)
  if (inflight) return inflight

  const promise = (async () => {
    const headers = { "content-type": "application/json" }
    if (apiKey) headers.authorization = `Bearer ${apiKey}`
    const reqUrl = `${baseURL}/models`
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs)

    try {
      const response = await fetch(reqUrl, { headers, signal: controller.signal })
      clearTimeout(timer)

      if (!response.ok) {
        console.error(`[auto-provider-models] request failed: HTTP ${response.status} ${response.statusText} for ${reqUrl}`)
        throw new Error(`request failed with status ${response.status}`)
      }

      const payload = await response.json()
      if (!Array.isArray(payload?.data)) {
        console.error(`[auto-provider-models] invalid response body for ${reqUrl}: ${JSON.stringify(payload).slice(0, 200)}`)
        throw new Error("response is not an OpenAI-compatible models payload")
      }

      return payload.data
    } catch (error) {
      clearTimeout(timer)
      if (error.name === "AbortError") {
        console.error(`[auto-provider-models] timeout after ${timeoutMs}ms for ${reqUrl}`)
        throw new Error(`timeout after ${timeoutMs}ms`)
      }
      if (error instanceof TypeError) {
        console.error(`[auto-provider-models] network error for ${reqUrl}: ${error.message}`)
      }
      throw error
    }
  })()

  const result = promise.finally(() => {
    if (inflightRequests.get(cacheKey) === result) {
      inflightRequests.delete(cacheKey)
    }
  })

  inflightRequests.set(cacheKey, result)
  return result
}

function shouldKeepModel(modelId, pluginOptions) {
  const include = Array.isArray(pluginOptions.include)
    ? new Set(pluginOptions.include.filter((item) => typeof item === "string"))
    : null
  const exclude = Array.isArray(pluginOptions.exclude)
    ? new Set(pluginOptions.exclude.filter((item) => typeof item === "string"))
    : null

  if (include && !include.has(modelId)) return false
  if (exclude && exclude.has(modelId)) return false
  return true
}

function resolveProviderEntries(pluginOptions) {
  if (Array.isArray(pluginOptions.provider)) return pluginOptions.provider
  if (typeof pluginOptions.provider === "string") return [pluginOptions.provider]
  return []
}

function normalizeProviderEntry(entry) {
  if (typeof entry === "string") return { id: entry }
  if (entry && typeof entry === "object") {
    const id = typeof entry.id === "string" && entry.id.trim() ? entry.id.trim() : null
    if (!id) return null
    return {
      id,
      ...(entry.baseURL ? { baseURL: normalizeBaseUrl(entry.baseURL) } : {}),
      ...(entry.apiKey ? { apiKey: entry.apiKey.trim() } : {}),
      ...(entry.apiKeyEnv ? { apiKeyEnv: entry.apiKeyEnv } : {}),
    }
  }
  return null
}

// ---- v1 syncProvider ----

async function syncProviderV1(config, providerEntry, globalOpts) {
  const providerId = providerEntry.id

  const providerConfig = config?.provider?.[providerId]
  if (!providerConfig || typeof providerConfig !== "object") {
    console.warn(`[auto-provider-models] provider not found in config: ${providerId}`)
    return
  }

  const baseURL = providerEntry.baseURL || normalizeBaseUrl(providerConfig?.options?.baseURL || globalOpts.baseURL)
  if (!baseURL) {
    console.warn(`[auto-provider-models] missing baseURL for provider: ${providerId}`)
    return
  }

  const timeoutMs = Number.isFinite(globalOpts.timeout) ? globalOpts.timeout : DEFAULT_TIMEOUT
  const cacheTTL = Number.isFinite(globalOpts.cacheTTL) ? globalOpts.cacheTTL : 0

  const existingModels =
    providerConfig.models && typeof providerConfig.models === "object" ? providerConfig.models : {}

  try {
    const apiKey = getApiKey(providerConfig.options, providerEntry, globalOpts)

    const cacheKey = getCacheKey(baseURL, apiKey)
    const cached = cacheTTL > 0 ? modelCache.get(cacheKey) : null
    if (cached && Date.now() - cached.timestamp < cacheTTL) {
      providerConfig.models = cached.models
      return
    }

    let enrichCache = null
    if (globalOpts.enrich) {
      enrichCache = await fetchAndBuildEnrichCache({
        onError: (msg) => console.warn(`[auto-provider-models] enrich fetch failed: ${msg}`),
      })
    }

    const remoteModels = await fetchRemoteModels(baseURL, apiKey, timeoutMs)
    const nextModels = { ...existingModels }

    for (const remoteModel of remoteModels) {
      const modelId = normalizeModelId(remoteModel?.id)
      if (!modelId || !shouldKeepModel(modelId, globalOpts)) continue
      nextModels[modelId] = buildModelEntry(modelId, remoteModel, existingModels[modelId], enrichCache)
    }

    providerConfig.models = nextModels

    if (cacheTTL > 0) {
      modelCache.set(cacheKey, { timestamp: Date.now(), models: nextModels })
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    console.warn(`[auto-provider-models] failed to sync models for ${providerId}: ${message}`)
  }
}

// ---- v2 syncProvider ----

async function syncProviderV2(ctx, providerEntry, globalOpts) {
  const providerID = providerEntry.id

  let providerInfo
  try {
    providerInfo = await ctx.provider.get({ providerID })
  } catch (error) {
    console.warn(`[auto-provider-models] provider not found: ${providerID}`)
    return
  }

  console.log(`[auto-provider-models] providerInfo for ${providerID}:`, JSON.stringify(providerInfo, null, 2))
  console.log(`[auto-provider-models] ctx.config keys:`, ctx.config ? Object.keys(ctx.config) : 'undefined')
  console.log(`[auto-provider-models] ctx.config.providers keys:`, ctx.config?.providers ? Object.keys(ctx.config.providers) : 'undefined')

  const providerSettings = providerInfo?.settings || ctx.config?.providers?.[providerID]?.settings || {}
  const baseURL = providerEntry.baseURL
    || normalizeBaseUrl(providerSettings.baseURL || providerSettings.base_url || globalOpts.baseURL)
  if (!baseURL) {
    console.warn(`[auto-provider-models] missing baseURL for provider: ${providerID}`)
    return
  }

  const timeoutMs = Number.isFinite(globalOpts.timeout) ? globalOpts.timeout : DEFAULT_TIMEOUT
  const cacheTTL = Number.isFinite(globalOpts.cacheTTL) ? globalOpts.cacheTTL : 0

  try {
    const apiKey = getApiKey(providerSettings, providerEntry, globalOpts)

    const cacheKey = getCacheKey(baseURL, apiKey)
    const cached = cacheTTL > 0 ? modelCache.get(cacheKey) : null
    if (cached && Date.now() - cached.timestamp < cacheTTL) {
      syncedModels.set(providerID, cached.items)
      return
    }

    let enrichCache = null
    if (globalOpts.enrich) {
      enrichCache = await fetchAndBuildEnrichCache({
        onError: (msg) => console.warn(`[auto-provider-models] enrich fetch failed: ${msg}`),
      })
    }

    const remoteModels = await fetchRemoteModels(baseURL, apiKey, timeoutMs)
    const items = []

    let existingModels = new Map()
    try {
      const modelsResult = await ctx.model.list({ providerID })
      if (modelsResult?.data) {
        for (const model of modelsResult.data) {
          existingModels.set(model.id, model)
        }
      }
    } catch {
      // 忽略错误
    }

    for (const remoteModel of remoteModels) {
      const modelID = normalizeModelId(remoteModel?.id)
      if (!modelID || !shouldKeepModel(modelID, globalOpts)) continue

      const existing = existingModels.get(modelID)
      items.push({ modelID, patch: buildV2ModelPatch(modelID, remoteModel, existing, enrichCache) })
    }

    syncedModels.set(providerID, items)

    if (cacheTTL > 0) {
      modelCache.set(cacheKey, { timestamp: Date.now(), items })
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    console.warn(`[auto-provider-models] failed to sync models for ${providerID}: ${message}`)
  }
}

/**
 * v1 server 函数：供 OpenCode v1 调用
 */
async function server(_input, pluginOptions = {}) {
  return {
    config: async (config) => {
      const rawEntries = resolveProviderEntries(pluginOptions)
      if (rawEntries.length === 0) {
        console.warn("[auto-provider-models] missing required option: provider")
        return
      }

      const entries = rawEntries
        .map(normalizeProviderEntry)
        .filter((e) => e !== null)

      const startupTimeout = Number.isFinite(pluginOptions.startupTimeout)
        ? pluginOptions.startupTimeout
        : DEFAULT_STARTUP_TIMEOUT

      const syncPromise = Promise.all(
        entries.map((entry) => syncProviderV1(config, entry, pluginOptions))
      )

      const overallTimer = new Promise((resolve) => {
        setTimeout(() => {
          console.warn(`[auto-provider-models] startup timed out after ${startupTimeout}ms, models may be incomplete`)
          resolve()
        }, startupTimeout)
      })

      await Promise.race([syncPromise, overallTimer])
    },
  }
}

/**
 * v2 setup 函数：供 OpenCode v2 调用
 */
async function setup(ctx) {
  const rawEntries = resolveProviderEntries(ctx.options)
  if (rawEntries.length === 0) {
    console.warn("[auto-provider-models] missing required option: provider")
    return
  }

  const entries = rawEntries.map(normalizeProviderEntry).filter((entry) => entry !== null)

  await ctx.provider.transform((editor) => {
    for (const [providerID, items] of syncedModels) {
      for (const { modelID, patch } of items) {
        editor.models.update(providerID, modelID, (draft) => {
          for (const [key, value] of Object.entries(patch)) {
            if (value !== undefined) draft[key] = value
          }
        })
      }
    }
  })

  const startupTimeout = Number.isFinite(ctx.options.startupTimeout)
    ? ctx.options.startupTimeout
    : DEFAULT_STARTUP_TIMEOUT

  const syncPromise = Promise.all(entries.map((entry) => syncProviderV2(ctx, entry, ctx.options)))

  const overallTimer = new Promise((resolve) => {
    setTimeout(() => {
      console.warn(`[auto-provider-models] startup timed out after ${startupTimeout}ms, models may be incomplete`)
      resolve()
    }, startupTimeout)
  })

  await Promise.race([syncPromise, overallTimer])

  if (syncedModels.size > 0) {
    await ctx.provider.reload()
  }
}

// 同时支持 v1 和 v2
// - v1: 调用 server() 函数
// - v2: 读取 id + setup() 函数
export default {
  id: "opencode-auto-provider-models",
  server,
  setup,
}
