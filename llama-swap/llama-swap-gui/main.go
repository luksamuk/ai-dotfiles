package main

import (
	"image/color"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/theme"
	"fyne.io/fyne/v2/widget"
)

// darkTheme implements a custom dark theme matching Catppuccin Mocha
type darkTheme struct{}

func (d *darkTheme) Color(name fyne.ThemeColorName, variant fyne.ThemeVariant) color.Color {
	switch name {
	case theme.ColorNameBackground:
		return color.NRGBA{R: 30, G: 30, B: 46, A: 255}
	case theme.ColorNameButton:
		return color.NRGBA{R: 49, G: 50, B: 68, A: 255}
	case theme.ColorNameForeground:
		return color.NRGBA{R: 205, G: 214, B: 244, A: 255}
	case theme.ColorNameInputBackground:
		return color.NRGBA{R: 24, G: 24, B: 37, A: 255}
	case theme.ColorNameInputBorder:
		return color.NRGBA{R: 88, G: 91, B: 112, A: 255}
	case theme.ColorNamePrimary:
		return color.NRGBA{R: 137, G: 180, B: 250, A: 255}
	case theme.ColorNameHover:
		return color.NRGBA{R: 69, G: 71, B: 90, A: 255}
	case theme.ColorNamePlaceHolder:
		return color.NRGBA{R: 108, G: 112, B: 134, A: 255}
	case theme.ColorNameSeparator:
		return color.NRGBA{R: 49, G: 50, B: 68, A: 255}
	case theme.ColorNameDisabled:
		return color.NRGBA{R: 108, G: 112, B: 134, A: 255}
	case theme.ColorNameDisabledButton:
		return color.NRGBA{R: 49, G: 50, B: 68, A: 255}
	case theme.ColorNameSelection:
		return color.NRGBA{R: 137, G: 180, B: 250, A: 80}
	case theme.ColorNameScrollBar:
		return color.NRGBA{R: 88, G: 91, B: 112, A: 200}
	case theme.ColorNameHyperlink:
		return color.NRGBA{R: 137, G: 180, B: 250, A: 255}
	case theme.ColorNameMenuBackground:
		return color.NRGBA{R: 30, G: 30, B: 46, A: 255}
	case theme.ColorNamePressed:
		return color.NRGBA{R: 49, G: 50, B: 68, A: 255}
	case theme.ColorNameSuccess:
		return color.NRGBA{R: 166, G: 227, B: 161, A: 255}
	case theme.ColorNameWarning:
		return color.NRGBA{R: 249, G: 226, B: 175, A: 255}
	case theme.ColorNameError:
		return color.NRGBA{R: 243, G: 139, B: 168, A: 255}
	default:
		return theme.DefaultTheme().Color(name, variant)
	}
}

func (d *darkTheme) Font(style fyne.TextStyle) fyne.Resource {
	return theme.DefaultTheme().Font(style)
}

func (d *darkTheme) Icon(name fyne.ThemeIconName) fyne.Resource {
	return theme.DefaultTheme().Icon(name)
}

func (d *darkTheme) Size(name fyne.ThemeSizeName) float32 {
	switch name {
	case theme.SizeNamePadding:
		return 6
	case theme.SizeNameText:
		return 13
	default:
		return theme.DefaultTheme().Size(name)
	}
}

func main() {
	a := app.NewWithID("com.luksamuk.llama-swap-gui")
	a.Settings().SetTheme(&darkTheme{})
	w := a.NewWindow("llama-swap-gui")
	w.Resize(fyne.NewSize(960, 680))

	// API client
	client := NewAPIClient("http://localhost:12434")

	// Create views
	modelsView := NewModelsView(client)
	runningView := NewRunningView(client)
	chatView := NewChatView(client)
	worldsimView := NewWorldsimView(client)

	tabs := container.NewAppTabs(
		container.NewTabItem("Models", modelsView.Canvas()),
		container.NewTabItem("Running", runningView.outer),
		container.NewTabItem("Chat", chatView.Canvas()),
		container.NewTabItem("Worldsim", worldsimView.Canvas()),
	)

	// Status bar
	statusBar := widget.NewLabel("Connecting to llama-swap...")

	w.SetContent(container.NewBorder(nil, statusBar, nil, nil, tabs))

	// Check connection on startup
	go func() {
		if _, err := client.FetchModels(); err != nil {
			fyne.Do(func() {
				statusBar.SetText("⚠ Cannot connect to llama-swap at " + client.BaseURL)
			})
		} else {
			fyne.Do(func() {
				statusBar.SetText("✓ Connected to llama-swap")
			})
		}
	}()

	// Auto-refresh running models
	go runningView.StartAutoRefresh()

	w.ShowAndRun()
}