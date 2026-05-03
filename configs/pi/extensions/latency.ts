import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  // Per-task timing (user message → agent finishes everything)
  let taskStartTime = 0;
  let currentModel = "unknown";

  // Per-model stats (accumulated across the entire session)
  const stats: Record<string, { totalMs: number; count: number; min: number; max: number }> = {};

  // ── Task-level events ──────────────────────────────────────────
  // agent_start fires when the agent loop begins processing a user message.
  // agent_end fires when the agent finishes ALL turns (including tool calls).
  // This gives us the total wall-clock time from "agent starts thinking"
  // to "agent is done, user can type again".

  pi.on("agent_start", async (_event, _ctx) => {
    taskStartTime = Date.now();
  });

  pi.on("agent_end", async (_event, ctx) => {
    if (!taskStartTime) return;
    const totalMs = Date.now() - taskStartTime;
    const seconds = (totalMs / 1000).toFixed(1);

    // Accumulate per-model stats
    const model = currentModel;
    if (!stats[model]) {
      stats[model] = { totalMs: 0, count: 0, min: Infinity, max: 0 };
    }
    const s = stats[model];
    s.totalMs += totalMs;
    s.count++;
    if (totalMs < s.min) s.min = totalMs;
    if (totalMs > s.max) s.max = totalMs;

    const avg = (s.totalMs / s.count / 1000).toFixed(1);
    ctx.ui.setStatus("latency", `⏱ ${seconds}s total · ${shortModel(model)} avg ${avg}s (${s.count} tasks)`);
  });

  // ── Model capture ───────────────────────────────────────────────
  pi.on("before_provider_request", async (event, _ctx) => {
    const payload = event.payload as { model?: string };
    if (payload?.model) {
      currentModel = payload.model;
    }
  });

  // ── Stats command ───────────────────────────────────────────────
  pi.registerCommand("latency", {
    description: "Show per-model task time stats (user message → agent done)",
    handler: async (_args, ctx) => {
      const models = Object.keys(stats);
      if (models.length === 0) {
        ctx.ui.notify("No latency data yet", "info");
        return;
      }

      const lines = models
        .slice()
        .sort((a, b) => (stats[b].totalMs / stats[b].count) - (stats[a].totalMs / stats[a].count))
        .map((model) => {
          const s = stats[model];
          const avg = (s.totalMs / s.count / 1000).toFixed(1);
          const min = (s.min / 1000).toFixed(1);
          const max = (s.max / 1000).toFixed(1);
          return `${shortModel(model)}  avg ${avg}s  min ${min}s  max ${max}s  ×${s.count}`;
        });

      ctx.ui.setWidget("latency", lines);
    },
  });

  function shortModel(modelId: string): string {
    return modelId.replace(/:think$/, "+t").replace(/:cloud$/, "☁");
  }
}