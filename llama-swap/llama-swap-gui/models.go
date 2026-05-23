package main

import (
	"fmt"
	"sort"
	"strings"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/widget"
)

// ModelsView shows all available models with details
type ModelsView struct {
	client   *APIClient
	list     *widget.List
	models   []Model
	search   *widget.Entry
	filtered []Model
	cardBox  *fyne.Container
	scroll   *container.Scroll
	outer    fyne.CanvasObject
}

// NewModelsView creates a new models view
func NewModelsView(client *APIClient) *ModelsView {
	v := &ModelsView{
		client: client,
	}

	// Search bar
	v.search = widget.NewEntry()
	v.search.SetPlaceHolder("Search models...")
	v.search.OnChanged = v.onSearchChanged

	// Model list (left sidebar)
	v.list = widget.NewList(
		func() int { return len(v.filtered) },
		func() fyne.CanvasObject {
			return widget.NewLabel("Model")
		},
		func(i widget.ListItemID, obj fyne.CanvasObject) {
			if i < len(v.filtered) {
				label := obj.(*widget.Label)
				m := v.filtered[i]
				label.SetText(m.Name)
			}
		},
	)
	v.list.OnSelected = v.onModelSelected

	// Detail card (right side)
	v.cardBox = container.NewVBox()
	v.scroll = container.NewVScroll(v.cardBox)
	v.scroll.SetMinSize(fyne.NewSize(450, 400))

	// Initial load
	go v.refresh()

	// left panel: search + list
	left := container.NewBorder(v.search, nil, nil, nil, v.list)

	v.outer = container.NewHSplit(left, v.scroll)

	return v
}

// Canvas returns the main CanvasObject for this view
func (v *ModelsView) Canvas() fyne.CanvasObject {
	return v.outer
}

func (v *ModelsView) refresh() {
	models, err := v.client.FetchModels()
	if err != nil {
		v.filtered = nil
		fyne.Do(func() {
			v.cardBox.Objects = []fyne.CanvasObject{
				widget.NewLabel("Error: " + err.Error()),
			}
			v.cardBox.Refresh()
			v.list.Refresh()
		})
		return
	}

	// Filter out :think variants for display
	var displayModels []Model
	for _, m := range models {
		if !strings.HasSuffix(m.ID, ":think") {
			displayModels = append(displayModels, m)
		}
	}

	// Sort by name
	sort.Slice(displayModels, func(i, j int) bool {
		return displayModels[i].Name < displayModels[j].Name
	})

	v.models = displayModels
	v.filtered = displayModels

	fyne.Do(func() {
		v.list.Refresh()
		// Show placeholder
		v.cardBox.Objects = []fyne.CanvasObject{
			widget.NewLabel("Select a model to view details"),
		}
		v.cardBox.Refresh()
	})
}

func (v *ModelsView) onSearchChanged(text string) {
	text = strings.ToLower(text)
	if text == "" {
		v.filtered = v.models
	} else {
		var filtered []Model
		for _, m := range v.models {
			if strings.Contains(strings.ToLower(m.Name), text) ||
				strings.Contains(strings.ToLower(m.ID), text) ||
				strings.Contains(strings.ToLower(m.Description), text) {
				filtered = append(filtered, m)
			}
		}
		v.filtered = filtered
	}
	v.list.Refresh()
	v.list.ScrollToTop()
}

func (v *ModelsView) onModelSelected(i widget.ListItemID) {
	if i >= len(v.filtered) {
		return
	}
	m := v.filtered[i]

	meta := m.Meta.LlamaSwap
	features := meta.Features

	// Build feature tags
	var tags []string
	if features.Thinking {
		tags = append(tags, "🤔 Thinking")
	}
	if features.Tools {
		tags = append(tags, "🛠 Tools")
	}
	if features.Vision {
		tags = append(tags, "👁 Vision")
	}
	if len(tags) == 0 {
		tags = append(tags, "📝 Text only")
	}

	// Build card content
	objects := []fyne.CanvasObject{
		widget.NewCard(m.Name, "", nil),
	}

	// Feature tags
	tagLabel := widget.NewLabel(strings.Join(tags, "  "))
	tagLabel.TextStyle = fyne.TextStyle{Bold: true}
	objects = append(objects, tagLabel)

	// Description
	if m.Description != "" {
		descLabel := widget.NewLabel(m.Description)
		descLabel.Wrapping = fyne.TextWrapWord
		objects = append(objects, descLabel)
	}

	// Separator
	objects = append(objects, widget.NewSeparator())

	// Metadata fields
	detailFields := []struct{ label, value string }{
		{"Model ID", m.ID},
		{"Size", meta.Size},
		{"Context", meta.Context},
		{"KV Cache", meta.KvCache},
		{"VRAM", meta.VRAM},
		{"Source", meta.Source},
	}

	for _, f := range detailFields {
		if f.value == "" {
			continue
		}
		hbox := container.NewHBox(
			widget.NewLabelWithStyle(f.label+":", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
			widget.NewLabel(f.value),
		)
		objects = append(objects, hbox)
	}

	// Warning
	if meta.Warning != "" {
		warnLabel := widget.NewLabel("⚠ " + meta.Warning)
		warnLabel.Wrapping = fyne.TextWrapWord
		objects = append(objects, widget.NewSeparator(), warnLabel)
	}

	// Think variant info
	if features.Thinking {
		thinkID := m.ID + ":think"
		thinkInfo := widget.NewLabel(fmt.Sprintf("💡 Thinking variant: %s", thinkID))
		objects = append(objects, widget.NewSeparator(), thinkInfo)
	}

	v.cardBox.Objects = objects
	v.cardBox.Refresh()
	v.scroll.ScrollToTop()
}