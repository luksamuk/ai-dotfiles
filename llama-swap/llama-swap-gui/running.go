package main

import (
	"fmt"
	"strings"
	"time"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/widget"
)

// RunningView shows currently running models with metrics and GPU dashboard
type RunningView struct {
	client           *APIClient
	content          *fyne.Container
	scroll           *container.Scroll
	autoRefresh      bool
	autoRefreshCheck *widget.Check
	sendBtn          *widget.Button
	unloadAllBtn    *widget.Button
	logViewer       *widget.Entry
	logScroll       *container.Scroll
	stopCh           chan struct{}
	outer            *fyne.Container
	// GPU history for mini-chart
	gpuHistory      []float64 // last 60 data points of GPU%
	vramHistory     []float64 // last 60 data points of VRAM GB
	tempHistory     []float64 // last 60 data points of temp °C
	tsHistory       []float64 // last 60 data points of tok/s
	maxHistory      int
}

// NewRunningView creates the running models dashboard
func NewRunningView(client *APIClient) *RunningView {
	v := &RunningView{
		client:      client,
		autoRefresh: true,
		stopCh:      make(chan struct{}),
		maxHistory:  60,
	}

	v.content = container.NewVBox(
		widget.NewLabel("Loading..."),
	)
	v.scroll = container.NewVScroll(v.content)
	v.scroll.SetMinSize(fyne.NewSize(600, 300))

	refreshBtn := widget.NewButton("🔄 Refresh", func() {
		go v.refresh()
	})

	v.unloadAllBtn = widget.NewButton("⏏ Unload All", func() {
		go v.unloadAll()
	})

	v.autoRefreshCheck = widget.NewCheck("Auto-refresh", func(checked bool) {
		v.autoRefresh = checked
	})
	v.autoRefreshCheck.SetChecked(true)

	// Log viewer (hidden by default)
	v.logViewer = widget.NewMultiLineEntry()
	v.logViewer.Wrapping = fyne.TextWrapWord
	v.logViewer.Disable()
	v.logScroll = container.NewVScroll(v.logViewer)
	v.logScroll.SetMinSize(fyne.NewSize(600, 200))
	v.logScroll.Hide()

	logBtn := widget.NewButton("📋 Logs", func() {
		if v.logScroll.Visible() {
			v.logScroll.Hide()
		} else {
			go v.fetchLogs()
			v.logScroll.Show()
		}
	})

	toolbar := container.NewHBox(
		refreshBtn,
		v.unloadAllBtn,
		v.autoRefreshCheck,
		logBtn,
		layout.NewSpacer(),
		widget.NewLabel("5s refresh"),
	)

	v.outer = container.NewBorder(toolbar, nil, nil, nil, v.scroll)

	go v.refresh()

	return v
}

func (v *RunningView) StartAutoRefresh() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			if v.autoRefresh {
				go v.refresh()
			}
		case <-v.stopCh:
			return
		}
	}
}

func (v *RunningView) refresh() {
	running, err := v.client.FetchRunning()
	if err != nil {
		fyne.Do(func() {
			v.content.Objects = []fyne.CanvasObject{
				widget.NewLabel("⚠ Cannot connect to llama-swap at " + v.client.BaseURL),
			}
			v.content.Refresh()
		})
		return
	}

	// Fetch system metrics for GPU info
	sysMetrics, _ := v.client.FetchSystemMetrics()

	// Update history
	var usedGB, totalGB, tempVal float64
	if sysMetrics != nil {
		if val, ok := sysMetrics["llamaswap_gpu_memory_used_bytes"]; ok {
			usedGB = val / 1073741824
		}
		if val, ok := sysMetrics["llamaswap_gpu_memory_total_bytes"]; ok {
			totalGB = val / 1073741824
		}
		if val, ok := sysMetrics["llamaswap_gpu_util_percent"]; ok {
			v.gpuHistory = append(v.gpuHistory, val)
		}
		if val, ok := sysMetrics["llamaswap_gpu_temperature_celsius"]; ok {
			tempVal = val
			v.tempHistory = append(v.tempHistory, val)
		}
	}
	v.vramHistory = append(v.vramHistory, usedGB)

	// Trim history
	if len(v.gpuHistory) > v.maxHistory {
		v.gpuHistory = v.gpuHistory[len(v.gpuHistory)-v.maxHistory:]
	}
	if len(v.vramHistory) > v.maxHistory {
		v.vramHistory = v.vramHistory[len(v.vramHistory)-v.maxHistory:]
	}
	if len(v.tempHistory) > v.maxHistory {
		v.tempHistory = v.tempHistory[len(v.tempHistory)-v.maxHistory:]
	}

	if len(running) == 0 {
		fyne.Do(func() {
			v.content.Objects = []fyne.CanvasObject{
				widget.NewLabelWithStyle(
					"No models currently loaded.\nModels will load on-demand when you send a request.",
					fyne.TextAlignLeading,
					fyne.TextStyle{},
				),
			}
			v.content.Refresh()
		})
		return
	}

	var cards []fyne.CanvasObject

	// GPU Dashboard (always show)
	gpuCard := v.buildGPUDashboard(sysMetrics, usedGB, totalGB, tempVal)
	cards = append(cards, gpuCard)

	// Mini GPU chart
	if len(v.gpuHistory) > 2 {
		chartCard := v.buildMiniChart("GPU Util %", v.gpuHistory)
		cards = append(cards, chartCard)
	}

	// Model cards
	for _, rm := range running {
		card := v.buildModelCard(rm, sysMetrics)
		cards = append(cards, card)
	}

	fyne.Do(func() {
		v.content.Objects = cards
		v.content.Refresh()
	})
}

// buildGPUDashboard creates a comprehensive GPU status card
func (v *RunningView) buildGPUDashboard(metrics map[string]float64, usedGB, totalGB, tempVal float64) *widget.Card {
	if metrics == nil {
		return widget.NewCard("🖥 GPU", "", widget.NewLabel("GPU metrics unavailable"))
	}

	pctUsed := float64(0)
	if totalGB > 0 {
		pctUsed = (usedGB / totalGB) * 100
	}

	// GPU utilization
	utilVal := float64(0)
	if val, ok := metrics["llamaswap_gpu_util_percent"]; ok {
		utilVal = val
	}

	// Color based on usage
	pctColor := "🟢"
	if pctUsed > 90 {
		pctColor = "🔴"
	} else if pctUsed > 75 {
		pctColor = "🟡"
	}

	// Progress bar for VRAM
	vramBar := v.makeProgressBar(pctUsed, 100, fmt.Sprintf("%.1f / %.1f GB (%.0f%%)", usedGB, totalGB, pctUsed))

	// Layout
	lines := []fyne.CanvasObject{
		widget.NewLabelWithStyle(fmt.Sprintf("%s GPU Dashboard", pctColor), fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		vramBar,
		widget.NewLabel(fmt.Sprintf("🎮 Util: %.0f%%  |  🌡️ Temp: %.0f°C", utilVal, tempVal)),
	}

	detailBox := container.NewVBox(lines...)
	return widget.NewCard("🖥 GPU", "", detailBox)
}

// buildMiniChart creates a simple ASCII-style mini chart card
func (v *RunningView) buildMiniChart(title string, data []float64) *widget.Card {
	if len(data) < 2 {
		return widget.NewCard(title, "", widget.NewLabel("Not enough data"))
	}

	// Create a simple text-based sparkline
	max := float64(0)
	for _, d := range data {
		if d > max {
			max = d
		}
	}
	if max == 0 {
		max = 100
	}

	// Build sparkline (20 chars wide)
	sparkline := ""
	steps := 20
	stepSize := float64(len(data)) / float64(steps)
	for i := 0; i < steps; i++ {
		idx := int(float64(i) * stepSize)
		if idx >= len(data) {
			idx = len(data) - 1
		}
		height := data[idx] / max
		char := "▁"
		switch {
		case height > 0.875:
			char = "▇"
		case height > 0.75:
			char = "▆"
		case height > 0.625:
			char = "▅"
		case height > 0.5:
			char = "▄"
		case height > 0.375:
			char = "▃"
		case height > 0.25:
			char = "▂"
		case height > 0.125:
			char = "▁"
		}
		sparkline += char
	}

	current := data[len(data)-1]
	avg := float64(0)
	for _, d := range data {
		avg += d
	}
	avg /= float64(len(data))

	label := widget.NewLabel(fmt.Sprintf("%s\n%s  now: %.1f  avg: %.1f", sparkline, title, current, avg))
	label.TextStyle = fyne.TextStyle{Monospace: true}

	return widget.NewCard(title, "", label)
}

// makeProgressBar creates a text-based progress bar
func (v *RunningView) makeProgressBar(current, max float64, label string) *widget.Label {
	if max == 0 {
		return widget.NewLabel(label)
	}
	pct := current / max * 100
	barLen := 30
	filled := int(pct / 100 * float64(barLen))
	if filled > barLen {
		filled = barLen
	}
	bar := strings.Repeat("█", filled) + strings.Repeat("░", barLen-filled)
	return widget.NewLabel(fmt.Sprintf("[%s] %.1f%%\n%s", bar, pct, label))
}

func (v *RunningView) buildModelCard(rm RunningModel, sysMetrics map[string]float64) *widget.Card {
	// Status icon
	statusIcon := "🟢"
	switch rm.State {
	case "starting", "loading", "health_check":
		statusIcon = "🟡"
	case "stopping", "unloading":
		statusIcon = "🔴"
	case "error", "failed":
		statusIcon = "❌"
	}

	// TTL formatting
	ttlStr := "-"
	if rm.TTL > 0 {
		if rm.TTL >= 3600 {
			ttlStr = fmt.Sprintf("%.1fh", float64(rm.TTL)/3600)
		} else if rm.TTL >= 60 {
			ttlStr = fmt.Sprintf("%dm", rm.TTL/60)
		} else {
			ttlStr = fmt.Sprintf("%ds", rm.TTL)
		}
	}

	// Port extraction
	port := extractPort(rm.Proxy)

	// Fetch model throughput
	genTps := "-"
	if port != "" {
		if mMetrics, err := v.client.FetchModelMetrics(port); err == nil {
			if val, ok := mMetrics["llamacpp:predicted_tokens_seconds"]; ok {
				genTps = fmt.Sprintf("%.1f", val)
				_ = val
				v.tsHistory = append(v.tsHistory, val)
				if len(v.tsHistory) > v.maxHistory {
					v.tsHistory = v.tsHistory[len(v.tsHistory)-v.maxHistory:]
				}
			}
		}
	}

	// Get model metadata
	meta := v.getModelMeta(rm.Model)

	// Feature tags
	var tags []string
	if meta != nil {
		f := meta.Features
		if f.Thinking {
			tags = append(tags, "🤔")
		}
		if f.Tools {
			tags = append(tags, "🛠")
		}
		if f.Vision {
			tags = append(tags, "👁")
		}
	}
	tagStr := "📝"
	if len(tags) > 0 {
		tagStr = strings.Join(tags, " ")
	}

	// Build detail lines
	title := fmt.Sprintf("%s %s", statusIcon, rm.Name)
	var lines []fyne.CanvasObject

	lines = append(lines, widget.NewLabel(fmt.Sprintf("Model: %s", rm.Model)))
	lines = append(lines, widget.NewLabel(fmt.Sprintf("Status: %s  |  TTL: %s", rm.State, ttlStr)))

	if meta != nil {
		lines = append(lines, widget.NewLabel(fmt.Sprintf("Size: %s  |  Context: %s  |  Features: %s", meta.Size, shortContext(meta.Context), tagStr)))
		if genTps != "-" {
			lines = append(lines, widget.NewLabel(fmt.Sprintf("⚡ Speed: %s t/s", genTps)))
		}
		if meta.KvCache != "" {
			lines = append(lines, widget.NewLabel(fmt.Sprintf("KV Cache: %s", meta.KvCache)))
		}
		if meta.VRAM != "" {
			lines = append(lines, widget.NewLabel(fmt.Sprintf("VRAM: %s", meta.VRAM)))
		}
		if meta.Warning != "" {
			w := widget.NewLabel("⚠ " + meta.Warning)
			w.Wrapping = fyne.TextWrapWord
			lines = append(lines, widget.NewSeparator(), w)
		}
	}

	// Unload button
	unloadBtn := widget.NewButton("⏏ Unload", func() {
		go v.unloadModel(rm.Model)
	})

	detailBox := container.NewVBox(lines...)
	detailWithBtn := container.NewBorder(nil, nil, nil, unloadBtn, detailBox)

	return widget.NewCard(title, "", detailWithBtn)
}

func (v *RunningView) getModelMeta(modelID string) *LlamaSwapMeta {
	models, err := v.client.FetchModels()
	if err != nil {
		return nil
	}
	for _, m := range models {
		if m.ID == modelID {
			meta := m.Meta.LlamaSwap
			return &meta
		}
	}
	return nil
}

func (v *RunningView) unloadAll() {
	if err := v.client.UnloadModel(""); err != nil {
		fmt.Printf("Error unloading all: %v\n", err)
	}
	go v.refresh()
}

func (v *RunningView) unloadModel(modelID string) {
	if err := v.client.UnloadModel(modelID); err != nil {
		fmt.Printf("Error unloading %s: %v\n", modelID, err)
	}
	go v.refresh()
}

func (v *RunningView) fetchLogs() {
	// Fetch recent journalctl logs
	var lines []string
	lines = append(lines, "📋 Recent llama-swap logs:\n")

	// We can't run journalctl directly from Fyne safely, so show a placeholder
	// In production, this would connect to a log streaming endpoint
	lines = append(lines, "Log viewer requires systemd journal access.\nUse: llama-swap-cli logs [N]")
	result := strings.Join(lines, "")

	fyne.Do(func() {
		v.logViewer.SetText(result)
		v.logScroll.Show()
		v.logScroll.Refresh()
	})
}

// shortContext extracts short context format
func shortContext(ctx string) string {
	ctx = strings.TrimSpace(ctx)
	if idx := strings.Index(ctx, "("); idx > 0 {
		ctx = strings.TrimSpace(ctx[:idx])
	}
	if ctx == "" {
		ctx = "-"
	}
	return ctx
}

// extractPort gets the port from a proxy URL
func extractPort(proxy string) string {
	parts := strings.Split(proxy, ":")
	if len(parts) >= 2 {
		return parts[len(parts)-1]
	}
	return ""
}