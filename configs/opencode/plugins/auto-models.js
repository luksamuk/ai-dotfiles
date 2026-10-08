// OpenCode v2 plugin: auto-sync models from OpenAI-compatible providers (llama-swap).
// Get {baseURL}/models per configured provider, then inject model entries via
// ctx.provider.transform — capabilities derived from the llama-swap payload,
// generic OpenAI fields as fallback for other providers (e.g. empero).

// Plain object shape works in v2 (loader requires: default export with id + setup).
// No npm dependencies needed — drop this file in ~/.config/opencode/plugins/ and go.
// (Equivalent sugar: `export default Plugin.define({ id: "auto-models", ... })`
//  from the @opencode/plugin package — same wire format.)

import { appendFileSync, readFileSync } from "node:fs"
import { homedir } from "node:os"

const DBG = "/tmp/auto-models-debug.log"
const dlog = (msg) => {
  try { appendFileSync(DBG, `${new Date().toISOString()} ${msg}\n`) } catch {}
  try { console.log("[auto-models]", msg) } catch {}
}

const DEFAULT_CONFIG = {
  // Provider IDs (from opencode.json "providers") to auto-sync.
  providers: ["local"],
  // Model IDs kept OUT of the catalog (embeddings, TTS, FIM, decision models).
  // Everything else /v1/models returns gets injected.
  exclude: [
    "d1-3b", "d1-omni-600m", "embeddinggemma-2", "julia-1", "laya",
    "lfm2.5-colbert-350m", "lfm2.5-embedding-350m", "lfm2.5-vl-3b",
    "nemotron-3-embed-1b", "qwen3-tts",
  ],
  // If set (array of IDs), ONLY these are injected — overrides `exclude`.
  include: null,
  // Per-provider baseURL override: { "local": "http://host:port/v1" }
  baseURL: {},
  timeoutMs: 3000,
}

function deriveModelPatch(remote, modelID) {
  const m = remote || {}
  const ls = m.meta?.llamaswap || {}
  const features = ls.features || {}

  const input = m.architecture?.input_modalities || m.input_modalities || m.modalities?.input || ["text"]
  const output = m.architecture?.output_modalities || m.output_modalities || m.modalities?.output || ["text"]

  const tools =
    m.capabilities?.function_calling === true ||
    (Array.isArray(m.supported_parameters) && m.supported_parameters.includes("tools")) ||
    features.tools === true

  const context =
    Number(m.context_window) || Number(m.meta?.llamaswap?.context) ||
    Number(m.context_length) || Number(m.max_input_tokens) || null

  // Fleet convention: 32K output budget for big-ctx models, ctx/4 otherwise.
  const outputLimit = context ? (context >= 131072 ? 32768 : Math.floor(context / 4)) : null

  const patch = {
    name: (typeof m.name === "string" && m.name.trim()) || modelID,
    capabilities: {
      tools: tools === true,
      input: Array.isArray(input) ? input.filter((x) => typeof x === "string") : ["text"],
      output: Array.isArray(output) ? output.filter((x) => typeof x === "string") : ["text"],
    },
  }
  if (context) patch.limit = { context, ...(outputLimit ? { output: outputLimit } : {}) }
  return patch
}

// Build a full Model.Info per the v2 schema (capabilities, variants, time,
// cost, status, enabled, limit are all required fields).
function makeModelInfo(providerID, id, patch) {
  return {
    id,
    modelID: id,
    providerID,
    name: patch.name || id,
    capabilities: patch.capabilities || { tools: true, input: ["text"], output: ["text"] },
    variants: [],
    time: { released: Date.now() },
    cost: [],
    status: "active",
    enabled: true,
    limit: patch.limit || { context: 32768, output: 8192 },
  }
}

export default {
  id: "auto-models",
  async setup(ctx) {
    try {
      return await _setup(ctx)
    } catch (e) {
      dlog(`setup FATAL: ${e?.stack || e?.message || e}`)
    }
  },
}

async function _setup(ctx) {
  dlog(`setup called; ctx keys: ${Object.keys(ctx || {}).join(",")}`)
  if (ctx?.provider) dlog(`ctx.provider keys: ${Object.keys(ctx.provider).join(",")}`)

  const opts = ctx?.options || {}
  const providerIDs = Array.isArray(opts.providers) ? opts.providers : DEFAULT_CONFIG.providers
  const exclude = Array.isArray(opts.exclude) ? opts.exclude : DEFAULT_CONFIG.exclude
  const include = Array.isArray(opts.include) ? opts.include : DEFAULT_CONFIG.include
  const baseURLs = opts.baseURL || DEFAULT_CONFIG.baseURL
  const timeoutMs = Number(opts.timeoutMs) || DEFAULT_CONFIG.timeoutMs
  const keep = (id) => (include ? include.includes(id) : !exclude.includes(id))

  // baseURL resolution: explicit option > runtime opencode.json (live registry
  // does not expose custom providers at setup time).
  const baseURLFor = (providerID) => {
    if (baseURLs[providerID]) return String(baseURLs[providerID]).replace(/\/+$/, "")
    try {
      const cfg = JSON.parse(readFileSync(`${homedir()}/.config/opencode/opencode.json`, "utf8"))
      const p = cfg?.providers?.[providerID]
      const b = p?.settings?.baseURL || p?.options?.baseURL
      return b ? String(b).replace(/\/+$/, "") : null
    } catch (e) {
      dlog(`baseURL read from config failed for ${providerID}: ${e?.message || e}`)
      return null
    }
  }

  dlog(`config resolved: providers=${JSON.stringify(providerIDs)}`)
  const synced = []

  for (const providerID of providerIDs) {
    const baseURL = baseURLFor(providerID)
    if (!baseURL) {
      dlog(`provider ${providerID}: no baseURL, skipping`)
      continue
    }

    try {
      const ctrl = new AbortController()
      const timer = setTimeout(() => ctrl.abort(), timeoutMs)
      const res = await fetch(`${baseURL}/models`, { signal: ctrl.signal })
      clearTimeout(timer)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const payload = await res.json()
      const models = Array.isArray(payload?.data) ? payload.data : []
      dlog(`provider ${providerID}: ${models.length} model(s) from ${baseURL}`)

      for (const rm of models) {
        const id = typeof rm?.id === "string" ? rm.id.trim() : ""
        if (!id || !keep(id)) continue
        synced.push({ providerID, id, patch: deriveModelPatch(rm, id) })
      }
    } catch (e) {
      dlog(`provider ${providerID}: sync failed: ${e?.message || e}`)
    }
  }

  dlog(`discovered ${synced.length} model(s) total; applying transform`)
  try {
    await ctx.provider.transform((editor) => {
      for (const providerID of providerIDs) {
        const items = synced.filter((s) => s.providerID === providerID)
        if (items.length === 0) continue
        try {
          const rec = editor.get(providerID)
          const existing = rec ? [...rec.models.values()].map((m) => ({ ...m })) : []
          dlog(`editor ${providerID}: existing=${existing.length} incoming=${items.length}`)
          const existingIDs = new Set(existing.map((m) => m.id))
          const toAdd = items
            .filter((it) => !existingIDs.has(it.id))
            .map((it) => makeModelInfo(providerID, it.id, it.patch))
          for (const it of items) {
            const cur = existing.find((m) => m.id === it.id)
            if (cur) {
              // existing manual entry wins; only fill missing fields
              cur.capabilities = cur.capabilities || it.patch.capabilities
              cur.limit = cur.limit || it.patch.limit
              cur.name = cur.name || it.patch.name
            }
          }
          if (toAdd.length > 0) {
            editor.models.set(providerID, [...existing, ...toAdd])
            dlog(`editor ${providerID}: set ${existing.length + toAdd.length} models (+${toAdd.length}: ${toAdd.map((m) => m.id).join(",")})`)
          }
        } catch (e) {
          dlog(`editor ${providerID} failed: ${e?.stack || e?.message || e}`)
        }
      }
    })
    dlog(`transform call returned`)
    if (synced.length > 0) {
      await ctx.provider.reload()
      dlog(`provider reload done`)
    }
    for (const { providerID, id } of synced) {
      dlog(`  + ${providerID}/${id}`)
    }
  } catch (e) {
    dlog(`transform failed: ${e?.stack || e?.message || e}`)
  }
  dlog(`setup finished`)
}