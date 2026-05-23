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

// WorldsimView provides an interactive web world simulation interface
// Inspired by the worldsim CLI — agent + world model loop
type WorldsimView struct {
	client        *APIClient
	agentSel      *widget.Select
	worldSel      *widget.Select
	stepsEntry    *widget.Entry
	modeSel       *widget.Select // auto / manual
	runBtn        *widget.Button
	stepBtn       *widget.Button
	stopBtn       *widget.Button
	log           *widget.Entry
	stateDisplay  *widget.Entry
	reasonDisplay *widget.Entry
	actionDisplay *widget.Entry
	scroll        *container.Scroll
	outer         *fyne.Container

	// Simulation state
	running    bool
	stopCh     chan struct{}
	current    string // current page state
	stepCount  int
	taskPrompt string
}

// World templates matching the CLI
var worldTemplates = map[string]string{
	"search_portal": "RootWebArea 'Global Start - Your Daily Portal', focused\n\t[1] banner 'Top Header'\n\t\t[2] link 'Set as Homepage'\n\t\t[3] link 'Feedback'\n\t\t[5] region 'Weather Widget'\n\t\t\tStaticText 'New York, USA'\n\t\t\t[6] image 'Sunny'\n\t\t\tStaticText '24°C'\n\t\t[8] link 'Sign In'\n\t[10] region 'Search Area'\n\t\t[11] image 'Global Start Logo'\n\t\tStaticText 'Search the web'\n\t\t[12] tablist 'Search Engine'\n\t\t\t[13] tab 'Google', selected\n\t\t\t[14] tab 'Bing'\n\t\t\t[15] tab 'DuckDuckGo'\n\t\t[18] combobox 'Web Search'\n\t\t\t[19] textbox 'Type keywords or URL...'\n\t\t[20] button 'Search'\n\t[30] navigation 'Category Bar'\n\t\t[31] link 'Home'\n\t\t[32] link 'News'\n\t\t[33] link 'Video'\n\t\t[34] link 'Shopping'\n\t[50] main 'Site Directory'\n\t\t[51] region 'Top Recommended'\n\t\t\t[52] heading 'Most Popular'\n\t\t\t[54] link 'Facebook'\n\t\t\t[56] link 'YouTube'\n\t\t\t[58] link 'Amazon'\n\t\t\t[60] link 'Wikipedia'",

	"github_homepage": "RootWebArea 'GitHub', focused\n\t[1] banner 'Top Header'\n\t\t[2] link 'Pull requests'\n\t\t[3] link 'Issues'\n\t\t[4] link 'Actions'\n\t\t[5] textbox 'Search or jump to...'\n\t\t[6] button 'Sign in'\n\t[10] main 'Content'\n\t\t[11] heading 'Welcome to GitHub'\n\t\t[12] link 'Explore repositories'\n\t\t[13] link 'Trending'\n\t\t[14] link 'Marketplace'",

	"shopping_site": "RootWebArea 'TechStore - Electronics', focused\n\t[1] banner 'Navigation'\n\t\t[2] link 'Home'\n\t\t[3] link 'Laptops'\n\t\t[4] link 'Phones'\n\t\t[5] link 'Accessories'\n\t\t[6] link 'Cart (0)'\n\t\t[7] textbox 'Search products...'\n\t\t[8] button 'Search'\n\t\t[9] link 'Sign In'\n\t[10] main 'Products'\n\t\t[11] heading 'Featured Products'\n\t\t[12] list 'Product Grid'\n\t\t\t[13] link 'MacBook Pro 14\" - $1,999'\n\t\t\t[14] link 'iPhone 16 - $999'\n\t\t\t[15] link 'AirPods Pro - $249'\n\t\t\t[16] link 'iPad Air - $599'",

	"custom": "",
}

var worldTasks = map[string]string{
	"search_portal":  "Find news about AI technology using the search",
	"github_homepage": "Search for Python repositories and find the most starred one",
	"shopping_site":  "Find and add a MacBook Pro to the cart",
}

// Agent and World system prompts
const AGENT_SYSTEM = `You are a web navigation agent. Your goal is to complete tasks on websites.

RULES:
1. You see page states in A11y Tree format. Elements have IDs in square brackets like [5], [13], etc.
2. Use these IDs in your actions — always with brackets, e.g. click([13]), fill([7], "text")
3. Available actions:
   - click([id]) — Click an element
   - fill([id], "text") — Type text into a field
   - keyboard_press("Enter") — Press a key
   - goto("url") — Navigate to URL
   - scroll(dx, dy) — Scroll the page
   - go_back() — Go back in browser history
4. If the task is complete or impossible, respond with: DONE <reason>
5. Output ONLY one action per turn. No explanations.
6. Look at the page state carefully — check cart counts, form values, page titles for progress.`

const WORLD_SYSTEM = `You are a web world model. I will provide you with an initial page state and a sequence of actions. For each action, predict the resulting page state.
Strictly maintain the original format. Output only the full page state without explanations, code, or truncation.`

// NewWorldsimView creates a new world simulation view
func NewWorldsimView(client *APIClient) *WorldsimView {
	v := &WorldsimView{
		client: client,
		stopCh: make(chan struct{}),
	}

	worlds := []string{}
	for k := range worldTemplates {
		worlds = append(worlds, k)
	}

	v.worldSel = widget.NewSelect(worlds, func(s string) {
		if task, ok := worldTasks[s]; ok {
			v.taskPrompt = task
		}
	})
	v.worldSel.PlaceHolder = "Select world template..."

	v.agentSel = widget.NewSelect([]string{"Loading..."}, nil)
	v.agentSel.PlaceHolder = "Select agent model..."

	v.stepsEntry = widget.NewEntry()
	v.stepsEntry.SetPlaceHolder("Max steps (default: 10)")
	v.stepsEntry.SetText("10")

	v.modeSel = widget.NewSelect([]string{"🤖 Auto", "👆 Manual"}, nil)
	v.modeSel.SetSelectedIndex(0)

	v.runBtn = widget.NewButton("▶ Run Simulation", func() {
		go v.runSimulation()
	})
	v.runBtn.Importance = widget.HighImportance

	v.stepBtn = widget.NewButton("⏭ Step", func() {
		go v.stepSimulation()
	})
	v.stepBtn.Disable()

	v.stopBtn = widget.NewButton("⏹ Stop", func() {
		v.running = false
		v.stopCh <- struct{}{}
	})
	v.stopBtn.Disable()

	// Log display
	v.log = widget.NewMultiLineEntry()
	v.log.SetPlaceHolder("Simulation log will appear here...")
	v.log.Wrapping = fyne.TextWrapWord
	v.log.Disable()

	v.stateDisplay = widget.NewMultiLineEntry()
	v.stateDisplay.SetPlaceHolder("Current page state...")
	v.stateDisplay.Wrapping = fyne.TextWrapWord
	v.stateDisplay.Disable()

	v.reasonDisplay = widget.NewMultiLineEntry()
	v.reasonDisplay.SetPlaceHolder("World model reasoning...")
	v.reasonDisplay.Wrapping = fyne.TextWrapWord
	v.reasonDisplay.Disable()

	v.actionDisplay = widget.NewMultiLineEntry()
	v.actionDisplay.SetPlaceHolder("Agent actions...")
	v.actionDisplay.Wrapping = fyne.TextWrapWord
	v.actionDisplay.Disable()

	stateCard := widget.NewCard("🌐 Page State", "", v.stateDisplay)
	reasonCard := widget.NewCard("🤔 Reasoning", "", v.reasonDisplay)

	leftPanel := container.NewVSplit(stateCard, reasonCard)
	leftPanel.SetOffset(0.6)

	logCard := widget.NewCard("📋 Log", "", v.log)
	actionCard := widget.NewCard("🤖 Agent Actions", "", v.actionDisplay)
	rightPanel := container.NewVSplit(logCard, actionCard)
	rightPanel.SetOffset(0.5)

	mainSplit := container.NewHSplit(leftPanel, rightPanel)
	mainSplit.SetOffset(0.55)

	// Toolbar
	toolbar := container.NewHBox(
		widget.NewLabel("🌐 World:"),
		v.worldSel,
		widget.NewLabel("🤖 Agent:"),
		v.agentSel,
		widget.NewLabel("Steps:"),
		v.stepsEntry,
		v.modeSel,
		layout.NewSpacer(),
		v.stopBtn,
		v.stepBtn,
		v.runBtn,
	)

	v.outer = container.NewBorder(toolbar, nil, nil, nil, mainSplit)

	// Load models
	go v.loadModels()

	return v
}

// Canvas returns the main CanvasObject for this view
func (v *WorldsimView) Canvas() fyne.CanvasObject {
	return v.outer
}

func (v *WorldsimView) loadModels() {
	models, err := v.client.FetchModels()
	if err != nil {
		fyne.Do(func() {
			v.agentSel.Options = []string{"Error: " + err.Error()}
			v.agentSel.Refresh()
		})
		return
	}

	var names []string
	// Filter for thinking-capable models (good agents)
	for _, m := range models {
		if m.Meta.LlamaSwap.Features.Thinking || m.Meta.LlamaSwap.Features.Tools {
			names = append(names, m.Name)
		}
	}
	// If no thinking models, show all
	if len(names) == 0 {
		for _, m := range models {
			names = append(names, m.Name)
		}
	}

	fyne.Do(func() {
		v.agentSel.Options = names
		if len(names) > 0 {
			v.agentSel.SetSelectedIndex(0)
		}
		v.agentSel.Refresh()
	})
}

func (v *WorldsimView) getSelectedAgentID() string {
	selected := v.agentSel.Selected
	if selected == "" {
		return ""
	}
	models, err := v.client.FetchModels()
	if err != nil {
		return ""
	}
	for _, m := range models {
		if m.Name == selected {
			// Use :think variant if available
			if m.Meta.LlamaSwap.Features.Thinking {
				return m.ID + ":think"
			}
			return m.ID
		}
	}
	return ""
}

func (v *WorldsimView) runSimulation() {
	world := v.worldSel.Selected
	if world == "" {
		fyne.Do(func() { v.log.SetText(v.log.Text + "\n⚠ Select a world template first") })
		return
	}

	agentID := v.getSelectedAgentID()
	if agentID == "" {
		fyne.Do(func() { v.log.SetText(v.log.Text + "\n⚠ Select an agent model") })
		return
	}

	maxSteps := 10
	if v.stepsEntry.Text != "" {
		fmt.Sscanf(v.stepsEntry.Text, "%d", &maxSteps)
	}

	v.running = true
	v.stepCount = 0
	fyne.Do(func() {
		v.runBtn.Disable()
		v.stepBtn.Disable()
		v.stopBtn.Enable()
		v.log.SetText(fmt.Sprintf("🌍 Starting worldsim: %s\n🤖 Agent: %s\n📋 Task: %s\n", world, agentID, v.taskPrompt))
	})

	// Set initial state
	v.current = worldTemplates[world]
	fyne.Do(func() {
		v.stateDisplay.SetText(v.current)
	})

	// Auto mode: run all steps
	if v.modeSel.Selected == "🤖 Auto" {
		for v.stepCount < maxSteps && v.running {
			v.stepSimulation()
			time.Sleep(500 * time.Millisecond) // Brief pause between steps
		}
	} else {
		// Manual mode: one step at a time
		fyne.Do(func() {
			v.stepBtn.Enable()
			v.log.SetText(v.log.Text + "\n👆 Manual mode — click Step for each action")
		})
		return
	}

	fyne.Do(func() {
		v.runBtn.Enable()
		v.stopBtn.Disable()
		v.log.SetText(v.log.Text + fmt.Sprintf("\n✅ Simulation complete (%d steps)", v.stepCount))
	})
	v.running = false
}

func (v *WorldsimView) stepSimulation() {
	if v.current == "" {
		return
	}

	agentID := v.getSelectedAgentID()
	if agentID == "" {
		return
	}

	v.stepCount++
	stepNum := v.stepCount

	fyne.Do(func() {
		v.log.SetText(v.log.Text + fmt.Sprintf("\n--- Step %d ---", stepNum))
	})

	// Build agent prompt
	agentMessages := []ChatMessage{
		{Role: "system", Content: AGENT_SYSTEM},
		{Role: "user", Content: fmt.Sprintf("Task: %s\n\nCurrent page state:\n%s\n\nWhat action should I take next?", v.taskPrompt, v.current)},
	}

	// Call agent
	fyne.Do(func() { v.log.SetText(v.log.Text + "\n🤖 Agent thinking...") })

	req := ChatRequest{
		Model:       agentID,
		Messages:    agentMessages,
		MaxTokens:   512,
		Temperature: 0.0,
	}

	resp, err := v.client.SendChatCompletion(req)
	if err != nil {
		fyne.Do(func() { v.log.SetText(v.log.Text + "\n❌ Agent error: " + err.Error()) })
		v.running = false
		return
	}

	action := strings.TrimSpace(resp.Choices[0].Message.Content)

	// Check for DONE
	if strings.HasPrefix(strings.ToUpper(action), "DONE") {
		fyne.Do(func() {
			v.actionDisplay.SetText(v.actionDisplay.Text + fmt.Sprintf("\nStep %d: %s", stepNum, action))
			v.log.SetText(v.log.Text + fmt.Sprintf("\n✅ Agent completed task: %s", action))
			v.runBtn.Enable()
			v.stopBtn.Disable()
		})
		v.running = false
		return
	}

	fyne.Do(func() {
		v.actionDisplay.SetText(v.actionDisplay.Text + fmt.Sprintf("\nStep %d: %s", stepNum, action))
	})

	// Call world model to simulate the result
	worldMessages := []ChatMessage{
		{Role: "system", Content: WORLD_SYSTEM},
		{Role: "user", Content: fmt.Sprintf("Current state:\n%s\n\nAction taken: %s\n\nPredict the resulting page state:", v.current, action)},
	}

	fyne.Do(func() { v.log.SetText(v.log.Text + "\n🌐 World model predicting...") })

	worldResp, err := v.client.SendChatCompletion(ChatRequest{
		Model:     agentID, // Use same model as world model
		Messages: worldMessages,
		MaxTokens: 2048,
	})
	if err != nil {
		fyne.Do(func() { v.log.SetText(v.log.Text + "\n❌ World model error: " + err.Error()) })
		v.running = false
		return
	}

	newState := strings.TrimSpace(worldResp.Choices[0].Message.Content)

	// Extract reasoning if present
	reasonText := ""
	if idx := strings.Index(newState, "<reason>"); idx >= 0 {
		if endIdx := strings.Index(newState, "</reason>"); endIdx > idx {
			reasonText = strings.TrimSpace(newState[idx+8 : endIdx])
			newState = strings.TrimSpace(newState[:idx] + newState[endIdx+9:])
		}
	}

	v.current = newState

	fyne.Do(func() {
		v.stateDisplay.SetText(v.current)
		if reasonText != "" {
			v.reasonDisplay.SetText(v.reasonDisplay.Text + fmt.Sprintf("\n--- Step %d ---\n%s", stepNum, reasonText))
		}
		v.log.SetText(v.log.Text + fmt.Sprintf("\n✅ Step %d complete", stepNum))
	})

	// Auto mode: check if we should continue
	if !v.running {
		fyne.Do(func() {
			v.runBtn.Enable()
			v.stopBtn.Disable()
		})
	}
}