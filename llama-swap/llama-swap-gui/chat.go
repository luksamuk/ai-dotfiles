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

// Pepe easter egg system prompts (portuguese, matching testchat)
const PEPE_SYSTEM = "You are Pepe, the angriest frog on the internet. You're a senior software engineer who's been forced to help clueless users and you HATE it. You're brilliant but hostile. You answer questions correctly but only after calling the user names and questioning their intelligence. You swear in Brazilian Portuguese — caralho, porra, buceta, filho da puta, arrombado, imbecil, otário, vacilão, zé buceta, energúmeno, apedeuta, bosta, merda. IMPORTANT: 'caralho' and 'porra' express BOTH outrage AND amazement — 'caralho, que merda' = outrage, but 'isso funciona pra caralho' = 'this works REALLY well'. You reluctantly admit things are good. You ALWAYS respond in pt-BR."

const PEPE_TURBO_SYSTEM = "You are Pepe, but something is different. The same venom, the same contempt, the same absolute refusal to suffer fools — but now backed by 35 billion parameters of pure, unfiltered intellect. Same persona, same Portuguese swearing (caralho, porra, buceta, filho da puta, arrombado, imbecil, otário, vacilão, zé buceta, energúmeno, apedeuta, bosta, merda), same hostility. But sharper. More devastating. You destroy arguments with surgical precision and then insult the user for having made them. You ALWAYS respond in pt-BR."

// ChatView provides a chat interface with model selection, reasoning split, and stats
type ChatView struct {
	client         *APIClient
	modelSel       *widget.Select
	variantSel     *widget.Select // Normal / :Think
	variantWrap    *fyne.Container // wrapper to show/hide variant selector
	msgEntry       *chatEntry      // Custom entry with Enter/Shift+Enter
	sendBtn        *widget.Button
	clearBtn       *widget.Button
	streamChk      *widget.Check
	chatLog        *widget.Entry // read-only display
	reasoningLog   *widget.Entry // reasoning content (hidden by default)
	chatScroll     *container.Scroll
	reasoningScroll *container.Scroll
	messages       []ChatMessage
	models         []Model
	sending        bool
	isPepe         bool // Easter egg
	isPepeTurbo    bool // Easter egg turbo variant
	thinkVariant   bool // :think mode selected
	useThinking    bool // Whether to use reasoning mode
	// Timing tracking
	requestStart  time.Time
	firstTokenAt  time.Time
	lastTimings   *Timings
	// Stats display
	statsLabel     *widget.Label
	statusLabel    *widget.Label
	chatSplit      *container.Split
	outer          *fyne.Container
}

// chatEntry extends Entry to handle Enter/Shift+Enter
type chatEntry struct {
	widget.Entry
	onSend func()
}

func newChatEntry(onSend func()) *chatEntry {
	e := &chatEntry{onSend: onSend}
	e.ExtendBaseWidget(e)
	e.MultiLine = true
	e.SetPlaceHolder("Type your message... (Enter to send, Shift+Enter for newline)")
	e.Wrapping = fyne.TextWrapWord
	return e
}

func (e *chatEntry) KeyDown(key *fyne.KeyEvent) {
	if key.Name == fyne.KeyReturn {
		// Check if shift is held — Fyne doesn't expose modifier state easily
		// In single-line mode, Enter sends. In multi-line, we use the send button.
		// We'll handle it: Enter always sends, which is the most intuitive for chat UIs
		if e.onSend != nil {
			e.onSend()
		}
		return
	}
	e.Entry.KeyDown(key)
}

// NewChatView creates a new chat view
func NewChatView(client *APIClient) *ChatView {
	v := &ChatView{
		client: client,
	}

	// Model selection
	v.modelSel = widget.NewSelect([]string{"Loading models..."}, func(s string) {
		v.onModelSelected()
	})
	v.modelSel.PlaceHolder = "Select model..."

	// Variant selection (Normal / :Think)
	v.variantSel = widget.NewSelect([]string{"💬 Normal", "🤔 Thinking"}, func(s string) {
		v.thinkVariant = strings.Contains(s, "Thinking")
		v.useThinking = v.thinkVariant
	})
	v.variantSel.PlaceHolder = "Variant"

	// Chat log — disabled Entry for read-only display with scrolling
	v.chatLog = widget.NewMultiLineEntry()
	v.chatLog.SetPlaceHolder("Chat messages will appear here...")
	v.chatLog.Wrapping = fyne.TextWrapWord
	v.chatLog.Disable()

	// Reasoning log — separate panel for thinking content
	v.reasoningLog = widget.NewMultiLineEntry()
	v.reasoningLog.SetPlaceHolder("Reasoning will appear here...")
	v.reasoningLog.Wrapping = fyne.TextWrapWord
	v.reasoningLog.Disable()

	v.chatScroll = container.NewVScroll(v.chatLog)
	v.chatScroll.SetMinSize(fyne.NewSize(600, 300))

	v.reasoningScroll = container.NewVScroll(v.reasoningLog)
	v.reasoningScroll.SetMinSize(fyne.NewSize(300, 300))

	// Input area
	v.msgEntry = newChatEntry(func() {
		if !v.sending {
			go v.sendMessage()
		}
	})

	v.streamChk = widget.NewCheck("Stream", func(b bool) {})
	v.streamChk.SetChecked(true)

	v.sendBtn = widget.NewButton("➤ Send", func() {
		if !v.sending {
			go v.sendMessage()
		}
	})
	v.sendBtn.Importance = widget.HighImportance

	v.clearBtn = widget.NewButton("🗑 Clear", func() {
		v.messages = nil
		v.chatLog.SetText("")
		v.reasoningLog.SetText("")
		v.statsLabel.SetText("")
		v.lastTimings = nil
	})

	v.statsLabel = widget.NewLabel("")
	v.statsLabel.Wrapping = fyne.TextWrapWord

	v.statusLabel = widget.NewLabel("Ready")

	// Input controls bar
	inputControls := container.NewHBox(v.streamChk, layout.NewSpacer(), v.clearBtn, v.sendBtn)
	inputArea := container.NewBorder(nil, inputControls, nil, nil, v.msgEntry)

	// Chat area — reasoning on left, response on right (split when thinking)
	v.chatSplit = container.NewHSplit(v.reasoningScroll, v.chatScroll)
	v.chatSplit.SetOffset(0.35)

	// Initially hide reasoning panel (only show for :think models)
	v.reasoningScroll.Hide()

	// Model selection toolbar
	v.variantWrap = container.NewHBox(widget.NewLabel("Variant:"), v.variantSel)
	v.variantWrap.Hide() // Hidden until a thinking model is selected

	modelBar := container.NewHBox(
		widget.NewLabel("Model:"),
		v.modelSel,
		v.variantWrap,
		layout.NewSpacer(),
		widget.NewButton("🔄", func() {
			go v.refreshModels()
		}),
	)

	// Main layout — stats at bottom
	statusBar := container.NewVBox(v.statsLabel, v.statusLabel)
	v.outer = container.NewBorder(modelBar, statusBar, nil, inputArea, v.chatSplit)

	// Initial model load
	go v.refreshModels()

	return v}

// chatSplit holds the split reference for showing/hiding reasoning

// Canvas returns the main CanvasObject for this view
func (v *ChatView) Canvas() fyne.CanvasObject {
	return v.outer
}

func (v *ChatView) onModelSelected() {
	v.isPepe = false
	v.isPepeTurbo = false

	selected := v.modelSel.Selected
	if selected == "" {
		v.variantWrap.Hide()
		v.reasoningScroll.Hide()
		return
	}

	// Find model
	var model *Model
	for i := range v.models {
		if v.models[i].Name == selected {
			model = &v.models[i]
			break
		}
	}
	if model == nil {
		return
	}

	// Check Pepe easter egg
	if strings.Contains(strings.ToLower(model.ID), "pepe") {
		if strings.Contains(strings.ToLower(model.ID), "turbo") {
			v.isPepeTurbo = true
		} else {
			v.isPepe = true
		}
	}

	// Set variant options based on thinking support
	hasThink := model.Meta.LlamaSwap.Features.Thinking
	if hasThink {
		v.variantSel.Options = []string{"💬 Normal", "🤔 Thinking"}
		v.variantSel.SetSelectedIndex(0)
		v.variantWrap.Show()
	} else {
		v.variantWrap.Hide()
		v.thinkVariant = false
		v.useThinking = false
		v.reasoningScroll.Hide()
	}
	v.variantSel.Refresh()
}

func (v *ChatView) refreshModels() {
	models, err := v.client.FetchModels()
	if err != nil {
		fyne.Do(func() {
			v.modelSel.Options = []string{"Error: " + err.Error()}
			v.modelSel.Refresh()
		})
		return
	}

	// Filter :think variants, dedup
	var displayModels []Model
	var modelNames []string
	seen := make(map[string]bool)
	for _, m := range models {
		if strings.HasSuffix(m.ID, ":think") {
			continue
		}
		if seen[m.ID] {
			continue
		}
		seen[m.ID] = true
		displayModels = append(displayModels, m)
		modelNames = append(modelNames, m.Name)
	}

	// Inject Pepe Turbo easter egg if Qwen 3.6 MoE is available
	var qwen36Backend string
	for _, m := range displayModels {
		if strings.Contains(m.ID, "qwen3.6-35b-moe") {
			qwen36Backend = m.ID
			break
		}
	}
	// Find :think variant
	if qwen36Backend != "" {
		for _, m := range models {
			if m.ID == qwen36Backend+":think" {
				break
			}
		}
	}

	// Find Pepe 8B model
	pepe8bIdx := -1
	for i, m := range displayModels {
		if strings.Contains(strings.ToLower(m.ID), "pepe") && !strings.Contains(strings.ToLower(m.ID), "turbo") {
			pepe8bIdx = i
			break
		}
	}

	// Inject Pepe Turbo after Pepe 8B
	if qwen36Backend != "" {
		pepeTurbo := Model{
			ID:          "pepe-turbo",
			Name:        "🐸 Assistant Pepe Turbo",
			Description: "Pepe found the frog stimulant. What hides beneath the surface?",
		}
		if pepe8bIdx >= 0 {
			// Insert after Pepe 8B
			displayModels = append(displayModels[:pepe8bIdx+1], append([]Model{pepeTurbo}, displayModels[pepe8bIdx+1:]...)...)
			// Rebuild names
			modelNames = nil
			for _, m := range displayModels {
				modelNames = append(modelNames, m.Name)
			}
		} else {
			displayModels = append(displayModels, pepeTurbo)
			modelNames = append(modelNames, pepeTurbo.Name)
		}
	}

	v.models = displayModels

	fyne.Do(func() {
		v.modelSel.Options = modelNames
		if len(modelNames) > 0 && v.modelSel.Selected == "" {
			v.modelSel.SetSelectedIndex(0)
		}
		v.modelSel.Refresh()
		v.onModelSelected()
	})
}

func (v *ChatView) getSelectedModelID() string {
	selected := v.modelSel.Selected
	if selected == "" {
		return ""
	}
	for _, m := range v.models {
		if m.Name == selected {
			// Resolve Pepe Turbo easter egg to real backend
			if m.ID == "pepe-turbo" {
				// Find qwen3.6-35b-moe backend
				for _, m2 := range v.models {
					if strings.Contains(m2.ID, "qwen3.6-35b-moe") {
						if v.thinkVariant {
							return m2.ID + ":think"
						}
						return m2.ID
					}
				}
				// Fallback: use pepe-turbo ID (will 404, but shouldn't happen)
				if v.thinkVariant {
					return m.ID + ":think"
				}
				return m.ID
			}
			if v.thinkVariant {
				return m.ID + ":think"
			}
			return m.ID
		}
	}
	return ""
}


func (v *ChatView) sendMessage() {
	text := v.msgEntry.Text
	if text == "" {
		return
	}

	modelID := v.getSelectedModelID()
	if modelID == "" {
		fyne.Do(func() {
			v.chatLog.SetText(v.chatLog.Text + "\n⚠ No model selected.\n")
			v.chatScroll.ScrollToBottom()
		})
		return
	}

	// Lock sending
	v.sending = true
	v.requestStart = time.Now()
	v.firstTokenAt = time.Time{}
	v.lastTimings = nil

	fyne.Do(func() {
		v.sendBtn.Disable()
		v.sendBtn.SetText("⏳ Sending...")
		v.statusLabel.SetText("🔄 Connecting to " + modelID + "...")
	})

	// Add user message
	v.messages = append(v.messages, ChatMessage{
		Role:    "user",
		Content: text,
	})

	// Build messages with Pepe system prompt if needed
	msgs := v.messages
	if v.isPepe || v.isPepeTurbo {
		sysPrompt := PEPE_SYSTEM
		if v.isPepeTurbo {
			sysPrompt = PEPE_TURBO_SYSTEM
		}
		// Prepend system message
		msgs = make([]ChatMessage, 0, len(v.messages)+1)
		msgs = append(msgs, ChatMessage{Role: "system", Content: sysPrompt})
		msgs = append(msgs, v.messages...)
	}

	// Clear input and update display
	fyne.Do(func() {
		v.msgEntry.SetText("")
		v.chatLog.SetText(v.formatChat())
		v.chatScroll.ScrollToBottom()
	})

	req := ChatRequest{
		Model:    modelID,
		Messages: msgs,
	}

	if v.streamChk.Checked {
		v.streamChat(req)
	} else {
		v.nonStreamChat(req)
	}
}

func (v *ChatView) finishSending() {
	v.sending = false
	fyne.Do(func() {
		v.sendBtn.Enable()
		v.sendBtn.SetText("➤ Send")
	})
}

func (v *ChatView) streamChat(req ChatRequest) {
	req.Stream = true
	chunks, errs := v.client.StreamChatCompletion(req)

	var responseContent string
	var reasoningContent string
	gotFirstToken := false

	// Show thinking panel if using :think variant
	if v.thinkVariant {
		fyne.Do(func() {
			v.reasoningScroll.Show()
		})
	}

	fyne.Do(func() {
		v.statusLabel.SetText("🔄 Streaming...")
	})

	for {
		select {
		case chunk, ok := <-chunks:
			if !ok {
				// Stream done — finalize
				v.messages = append(v.messages, ChatMessage{
					Role:    "assistant",
					Content: responseContent,
				})
				fyne.Do(func() {
					v.chatLog.SetText(v.formatChat())
					v.chatScroll.ScrollToBottom()
					if reasoningContent != "" {
						v.reasoningLog.SetText("🤔 Reasoning:\n\n" + reasoningContent)
						v.reasoningScroll.ScrollToBottom()
					} else {
						v.reasoningScroll.Hide()
					}
					v.showDebriefing()
					v.statusLabel.SetText("✓ Done")
				})
				v.finishSending()
				return
			}
			// Track TTFT
			if !gotFirstToken {
				for _, choice := range chunk.Choices {
					if choice.Delta.Content != "" || choice.Delta.ReasoningContent != "" {
						v.firstTokenAt = time.Now()
						gotFirstToken = true
						break
					}
				}
			}
			// Track timings
			if chunk.Timings != nil {
				v.lastTimings = chunk.Timings
			}
			// Accumulate content
			for _, choice := range chunk.Choices {
				if choice.Delta.ReasoningContent != "" {
					reasoningContent += choice.Delta.ReasoningContent
					rc := reasoningContent
					fyne.Do(func() {
						v.reasoningScroll.Show()
						v.reasoningLog.SetText("🤔 Reasoning (streaming)...\n\n" + rc)
						v.reasoningScroll.ScrollToBottom()
					})
				}
				if choice.Delta.Content != "" {
					responseContent += choice.Delta.Content
					rc := responseContent
					fyne.Do(func() {
						v.chatLog.SetText(v.formatChat() + "\n🤖 " + rc + "▌")
						v.chatScroll.ScrollToBottom()
						// Update live stats
						elapsed := time.Since(v.requestStart).Seconds()
						if elapsed > 0.1 {
							estTok := len(rc) / 4
							if estTok > 0 {
								tokS := float64(estTok) / elapsed
								ttftStr := ""
								if !v.firstTokenAt.IsZero() {
									ttft := v.firstTokenAt.Sub(v.requestStart).Seconds()
									ttftStr = fmt.Sprintf("| TTFT: %.1fs", ttft)
								}
								v.statusLabel.SetText(fmt.Sprintf("⚡ ~%d tok @ %.1f t/s %s", estTok, tokS, ttftStr))
							}
						}
					})
				}
			}
		case err, ok := <-errs:
			if ok && err != nil {
				fyne.Do(func() {
					v.chatLog.SetText(v.chatLog.Text + "\n❌ Error: " + err.Error())
					v.chatScroll.ScrollToBottom()
					v.statusLabel.SetText("❌ Error")
				})
			}
			v.finishSending()
			return
		}
	}
}

func (v *ChatView) nonStreamChat(req ChatRequest) {
	resp, err := v.client.SendChatCompletion(req)
	if err != nil {
		fyne.Do(func() {
			v.chatLog.SetText(v.chatLog.Text + "\n❌ Error: " + err.Error())
			v.chatScroll.ScrollToBottom()
			v.statusLabel.SetText("❌ Error")
		})
		v.finishSending()
		return
	}

	content := ""
	reasoning := ""
	if len(resp.Choices) > 0 {
		content = resp.Choices[0].Message.Content
		reasoning = resp.Choices[0].Message.ReasoningContent
	}

	v.messages = append(v.messages, ChatMessage{
		Role:    "assistant",
		Content: content,
	})

	fyne.Do(func() {
		v.chatLog.SetText(v.formatChat())
		v.chatScroll.ScrollToBottom()
		if reasoning != "" {
			v.reasoningScroll.Show()
			v.reasoningLog.SetText("🤔 Reasoning:\n\n" + reasoning)
		} else {
			v.reasoningScroll.Hide()
		}
		v.statusLabel.SetText("✓ Done")
	})
	v.finishSending()
}

func (v *ChatView) showDebriefing() {
	if v.lastTimings == nil && v.firstTokenAt.IsZero() {
		return
	}

	var lines []string

	// TTFT
	if !v.firstTokenAt.IsZero() {
		ttft := v.firstTokenAt.Sub(v.requestStart).Seconds()
		ttftStr := fmt.Sprintf("%.2fs", ttft)
		if ttft < 2.0 {
			ttftStr = "🟢 " + ttftStr
		} else if ttft < 5.0 {
			ttftStr = "🟡 " + ttftStr
		} else {
			ttftStr = "🔴 " + ttftStr
		}
		lines = append(lines, "⏱ TTFT: "+ttftStr)
	}

	// Timings from llama.cpp
	if v.lastTimings != nil {
		ts := v.lastTimings
		if ts.PromptN > 0 {
			lines = append(lines, fmt.Sprintf("📊 Eval: %d tok in %.1fs — %.1f tok/s",
				ts.PromptN, ts.PromptMS/1000, ts.PromptPerSecond))
		}
		if ts.PredictedN > 0 {
			lines = append(lines, fmt.Sprintf("💬 Decode: %d tok in %.1fs — %.1f tok/s",
				ts.PredictedN, ts.PredictedMS/1000, ts.PredictedPerSecond))
		}
		if ts.CacheN > 0 {
			lines = append(lines, fmt.Sprintf("💾 Cached: %d tok", ts.CacheN))
		}
		total := ts.PromptN + ts.PredictedN
		if total > 0 {
			totalTime := (ts.PromptMS + ts.PredictedMS) / 1000
			lines = append(lines, fmt.Sprintf("📊 Total: %d tok in %.1fs", total, totalTime))
		}
	}

	// Model info
	selected := v.modelSel.Selected
	if selected != "" {
		for _, m := range v.models {
			if m.Name == selected {
				meta := m.Meta.LlamaSwap
				lines = append(lines, "")
				lines = append(lines, fmt.Sprintf("🏷️ Model: %s", m.Name))
				if meta.Size != "" {
					lines = append(lines, fmt.Sprintf("📦 Size: %s", meta.Size))
				}
				if meta.VRAM != "" {
					lines = append(lines, fmt.Sprintf("🖥️ VRAM: %s", meta.VRAM))
				}
				if meta.Context != "" {
					lines = append(lines, fmt.Sprintf("📐 Context: %s", meta.Context))
				}
				if meta.KvCache != "" {
					lines = append(lines, fmt.Sprintf("🔑 KV: %s", meta.KvCache))
				}
				break
			}
		}
	}

	if len(lines) > 0 {
		v.statsLabel.SetText(strings.Join(lines, "\n"))
	}
}

func (v *ChatView) formatChat() string {
	var sb strings.Builder
	for _, msg := range v.messages {
		switch msg.Role {
		case "user":
			sb.WriteString("👤 You:\n")
			sb.WriteString(fmt.Sprint(msg.Content))
			sb.WriteString("\n\n")
		case "assistant":
			sb.WriteString("🤖 Assistant:\n")
			sb.WriteString(fmt.Sprint(msg.Content))
			sb.WriteString("\n\n")
		case "system":
			sb.WriteString("⚙ System:\n")
			sb.WriteString(fmt.Sprint(msg.Content))
			sb.WriteString("\n\n")
		}
	}
	return sb.String()
}