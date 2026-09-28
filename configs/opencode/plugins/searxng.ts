import { Plugin } from "@opencode/plugin"

// SearXNG websearch provider — queries the self-hosted instance on the Sisyphus RPi.
// Returns results in the WebSearch.Result shape: { url, title, content, time }.
export default Plugin.define({
  id: "searxng",
  async setup(ctx) {
    await ctx.websearch.transform((editor) => {
      editor.add({
        id: "searxng",
        name: "SearXNG (Sisyphus)",
        execute: async ({ query }, { signal }) => {
          const url =
            "http://192.168.3.8:8888/search?format=json&q=" +
            encodeURIComponent(query)
          const response = await fetch(url, { signal })
          if (!response.ok) {
            throw new Error("SearXNG HTTP " + response.status)
          }
          const data = await response.json()
          return (data.results || []).slice(0, 10).map((r) => ({
            url: r.url,
            title: r.title,
            content: r.content || r.snippet || "",
            time: r.publishedDate ? { published: r.publishedDate } : {},
          }))
        },
      })
      editor.default.set("searxng")
    })
  },
})
